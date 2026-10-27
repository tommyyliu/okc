"""Workload dimensions; importable without Torch or Triton."""

SHAPES = {
    "qkv_gate": (2304, 8192),
    "o": (8192, 1024),
    "pooler": (4096, 8192),
    "unpooler": (8192, 4096),
    "router": (256, 8192),
}
SCORED_TOKENS = (1, 8, 16, 32, 128, 256)
TOKEN_BUCKETS = (1, 8, 16, 32, 64, 128, 256)


def shape_config(layer, m):
    n, k = SHAPES[layer]
    if layer == "unpooler" and m <= 8:
        n = 1024
    return dict(layer=layer, m=m, n=n, k=k)


def token_bucket(m):
    return next((upper for upper in TOKEN_BUCKETS if m <= upper), None)
