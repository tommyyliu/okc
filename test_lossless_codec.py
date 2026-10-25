"""Bit-exact format tests; CPU tests need only Python's standard library.

  python test_lossless_codec.py
  python test_lossless_codec.py --gpu

The GPU option also compares the submission's actual Triton pack/decode
kernels with this independent host encoder, including all 65,536 BF16 bit
patterns, mixed compact/raw blocks, signed zeros and NaN payloads. Matrix
correctness and fresh-input graph replay belong to validate.py.
"""

import argparse
import importlib.util
from pathlib import Path
import random
import unittest


def encode(bits):
    if len(bits) % 256:
        raise ValueError("expected whole blocks of 256 weights")
    payload = bytearray()
    offsets, bases = [], []
    for start in range(0, len(bits), 256):
        block = bits[start:start + 256]
        if any(not 0 <= b < 65536 for b in block):
            raise ValueError("not a 16-bit pattern")
        exponents = [(b >> 7) & 255 for b in block]
        low, high = min(exponents), max(exponents)
        compact = high - low <= 15
        offsets.append(len(payload))
        bases.append(low if compact else 256)
        payload.extend((b & 127) | ((b >> 8) & 128) for b in block)
        if compact:
            deltas = [e - low for e in exponents]
            payload.extend(deltas[i] | (deltas[i + 1] << 4) for i in range(0, 256, 2))
        else:
            payload.extend(exponents)
    return bytes(payload), offsets, bases


def decode(payload, offsets, bases):
    bits = []
    for offset, base in zip(offsets, bases):
        for i in range(256):
            sm = payload[offset + i]
            if base == 256:
                exponent = payload[offset + 256 + i]
            else:
                byte = payload[offset + 256 + i // 2]
                exponent = base + ((byte >> (4 * (i % 2))) & 15)
            bits.append((sm & 127) | ((sm & 128) << 8) | (exponent << 7))
    return bits


def mixed_patterns():
    ordered = list(range(65536))
    shuffled = ordered.copy()
    random.Random(80137).shuffle(shuffled)
    result = []
    for start in range(0, 65536, 256):
        result.extend(ordered[start:start + 256])
        result.extend(shuffled[start:start + 256])
    return result


class LosslessFormatTests(unittest.TestCase):
    def assert_round_trip(self, bits):
        encoded = encode(bits)
        self.assertEqual(decode(*encoded), bits)
        return encoded

    def test_every_bf16_bit_pattern_compact(self):
        payload, _, bases = self.assert_round_trip(list(range(65536)))
        self.assertTrue(all(base < 256 for base in bases))
        self.assertEqual(len(payload), 65536 * 3 // 2)

    def test_every_bf16_bit_pattern_raw(self):
        bits = list(range(65536))
        random.Random(80137).shuffle(bits)
        payload, _, bases = self.assert_round_trip(bits)
        self.assertTrue(all(base == 256 for base in bases))
        self.assertEqual(len(payload), 65536 * 2)

    def test_mixed_blocks_and_variable_offsets(self):
        _, offsets, bases = self.assert_round_trip(mixed_patterns())
        self.assertEqual(set(bases[1::2]), {256})
        self.assertTrue(all(base < 256 for base in bases[::2]))
        self.assertEqual(offsets[:4], [0, 384, 896, 1280])

    def test_exponent_range_boundary(self):
        for low in (0, 1, 100, 239, 240):
            for span in (0, 1, 15):
                block = [((low + (i % (span + 1))) << 7) | (i & 127) |
                         ((i % 2) << 15) for i in range(256)]
                payload, _, bases = self.assert_round_trip(block)
                self.assertEqual(bases, [low])
                self.assertEqual(len(payload), 384)
        block = [((i % 17) << 7) | (i & 127) for i in range(256)]
        self.assertEqual(self.assert_round_trip(block)[2], [256])

    def test_signs_subnormals_infinities_and_nan_payloads(self):
        # All encodings round-trip as integer bits, including signaling NaNs.
        patterns = [0, 0x8000, 1, 0x807f, 0x7f80, 0xff80, 0x7f81, 0x7fff]
        self.assert_round_trip(patterns * 32)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            encode([0])
        with self.assertRaises(ValueError):
            encode([65536] * 256)


def check_gpu():
    import torch
    import triton
    import triton.language as tl

    if not torch.cuda.is_available():
        raise SystemExit("--gpu requires PyTorch, Triton and a CUDA GPU.")
    path = Path(__file__).with_name("submission_triton_lossless.py")
    spec = importlib.util.spec_from_file_location("lossless_candidate", path)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    # Bind the JIT helper directly so the compiler sees it as a global symbol.
    global decode_kernel_helper
    decode_kernel_helper = candidate._decode_lossless

    @triton.jit
    def unpack(Payload, Offsets, Bases, Output):
        group = tl.program_id(0)
        lanes = tl.arange(0, 256)
        weights = decode_kernel_helper(Payload, Offsets, Bases, group, lanes, True)
        bits = weights.to(tl.uint16, bitcast=True).to(tl.int32)
        tl.store(Output + group * 256 + lanes, bits)

    for label, bits in (("compact", list(range(65536))), ("mixed", mixed_patterns())):
        expected_payload, expected_offsets, expected_bases = encode(bits)
        original_bits = torch.tensor(bits, device="cuda", dtype=torch.int32)
        weights = original_bits.to(torch.int16).view(torch.bfloat16).reshape(-1, 256)
        payload, offsets, bases, ratio = candidate.compress_weight(weights)
        torch.cuda.synchronize()
        assert bytes(payload.cpu().tolist()) == expected_payload, f"{label}: payload mismatch"
        assert offsets.cpu().tolist() == expected_offsets, f"{label}: offset mismatch"
        assert bases.cpu().tolist() == expected_bases, f"{label}: base mismatch"
        recovered = torch.empty_like(original_bits)
        unpack[(len(bits) // 256,)](payload, offsets, bases, recovered, num_warps=4)
        torch.cuda.synchronize()
        assert torch.equal(recovered, original_bits), f"{label}: GPU bit reconstruction mismatch"
        print(f"PASS GPU {label}: {len(bits):,} patterns, byte ratio {ratio:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    tests = unittest.main(argv=[__file__], exit=False)
    if not tests.result.wasSuccessful():
        raise SystemExit(1)
    if args.gpu:
        check_gpu()
