# Security Policy

openfusion is a proxy that holds upstream provider API keys and forwards
prompts. Please treat key handling and prompt/secret leakage as the primary
threat model.

## Reporting a vulnerability

**Do not open a public issue for security problems.**

Report privately via GitHub's
[private vulnerability reporting](https://github.com/shahar-dagan/openfusion/security/advisories/new)
("Report a vulnerability" under the repo's Security tab). Please include:

- a description and impact,
- steps to reproduce (a minimal config or request if relevant),
- any known mitigations.

We aim to acknowledge within a few days and will coordinate a fix and
disclosure timeline with you.

## What we care about most

- **Upstream key exfiltration** — provider keys must come from config/env only,
  never from client headers or request bodies.
- **Secret / prompt leakage in logs or metrics** — logs carry metadata and usage
  only; `/metrics` exposes labels and numbers only. A path that emits an
  `Authorization` value, `api_key`, prompt, or response body is a vulnerability.
- **SSRF via `base_url`** — config is operator-controlled and trusted; a way for
  a *client* to influence upstream URLs would not be.
- **Auth bypass** — the optional `OPENFUSION_API_KEYS` / `gateway.api_keys`
  allowlist should not be bypassable when set.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ("Security considerations")
for the full table of concerns and mitigations.

## Operator guidance

- Keep `openfusion.yaml` out of version control (it is gitignored) and
  `chmod 600` it — it contains expanded provider keys.
- `/metrics` is unauthenticated; bind it to a trusted interface only.
- Set `cost_controls` token ceilings and provider-side budgets to bound spend.

## Supported versions

openfusion is pre-1.0; only the latest `main` is supported. Fixes land on `main`
and the next release.
