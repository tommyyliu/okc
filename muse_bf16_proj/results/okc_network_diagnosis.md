# OKC command connectivity — resolved October 26

**Resolved in the current session.** Agent-run `okc whoami`, `problems`,
`leaderboard muse_bf16_proj`, `status`, and remote `check` requests now succeed.
The authenticated account is `tommyyliu`. The agent has completed new H200
checks; see [the October 26 experiments](round7_experiment.md). No local
permission or security configuration was edited to obtain access.

The investigation below records the **earlier restricted session**, not the
current permission profile or network behavior.

Investigated October 26, 2026, without reading credential values or launching
another GPU job.

**Confirmed:** this session's shell commands have networking disabled by the
Codex execution sandbox. This is independent of OKC's remote GPU worker sandbox.

Observed evidence:

| Probe | Result |
|---|---|
| `command -v okc` | `/home/tommy/.okc/bin/okc` |
| `okc --version` | `okc 0.7.1` |
| `okc --help`, `okc api` | Work locally; API base is `https://openkernelchallenge.dev/api/v1` |
| `okc problems --json` | `can't reach openkernelchallenge.dev: no such host` |
| `CODEX_SANDBOX_NETWORK_DISABLED` | `1` |
| `CODEX_PERMISSION_PROFILE` | `:workspace` |
| Python DNS lookup of OKC, PyPI and example.com | All fail with temporary DNS resolution failure |
| Direct internet TCP / local DNS UDP probes | `PermissionError: [Errno 1] Operation not permitted` |

The active session instructions also declare restricted networking and granular
approval with sandbox escalation disabled. There is no eligible escalation prompt
for this session. No escalation was attempted and rejected; it is disabled by
the supplied policy. The evidence establishes a network restriction before an
HTTP request/authentication response, not a bad token, exhausted check quota,
missing CLI, or lack of a local GPU. Authentication cannot be verified until
connectivity is available.

The local user Codex config has model/project/UI settings but no network or
permission override. The repository `.codex/` directory is empty and protected
by the session filesystem policy. The visible `:workspace` selection therefore
comes from effective session configuration/defaults rather than an explicit
network setting in either inspected config file. The exact host-side launcher
or policy owner is not exposed here.

Web search remains available through a separate tool path. Official OpenAI
documentation explicitly distinguishes command networking from web search,
apps and MCP access; shell networking does not follow from having a web tool.
See the [OpenAI configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

## How the host can enable the workflow

The application/session owner needs to enable outbound command networking for
this workspace/session, with DNS and HTTPS access to `openkernelchallenge.dev`.
The CLI then submits and polls remote H200 jobs; no local GPU is required for
`okc check`. Keep the filesystem restrictions if desired.

For a normal Codex CLI session under the user's control, the documented named
permission-profile form is:

```toml
default_permissions = "okc-network"

[permissions.okc-network]
extends = ":workspace"

[permissions.okc-network.network]
enabled = true
```

This is a proposed host configuration, **not applied**. Do not combine named
profiles with legacy `sandbox_mode` / `[sandbox_workspace_write]` settings.
For a legacy workspace-write configuration, the separate documented knob is
`sandbox_workspace_write.network_access = true`. Managed requirements or an
outer sandbox can still prevent these local settings from taking effect.

If restricting egress to OKC, use a supported network-proxy allowlist at the
host layer. Profile domain rules alone are not enforced without the Codex
network proxy or managed networking enabled. No broad sandbox-disable flag is
needed to solve this specific network restriction.

These configuration semantics are documented in the
[official OpenAI reference](https://learn.chatgpt.com/docs/config-file/config-reference).
After the host changes the policy, first verify `okc problems --json`; then
submit a prepared candidate. No local security configuration was changed here.

## Separate cluster-kernel failure

The user-launched cluster check `2d1ffa9571c8437d` reached the remote worker and
failed at router M8 during Triton frontend compilation. The unsupported
expression was `C in (2, 4, 8)` inside a Gluon static assertion. An analogous
BM assertion had the same operator. Both have been replaced with explicit
equality expressions and the candidate rebuilt; GPU validation remains pending.

The subsequent response in `build/router-cluster/check.jso`, job
`791d4d4fa2884dd5`, is a duplicate of that same failed source, not a new run.
The original result/log and duplicate response are retained in
`results/router_cluster_failed.json`, `router_cluster_failed.log`, and
`router_cluster_duplicate.json`. The exact old candidate is preserved as
`build/router-cluster/failed-candidate.py`.
