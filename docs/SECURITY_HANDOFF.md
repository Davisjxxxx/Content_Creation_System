# Security Review Handoff

## Trust boundaries
1. Internet/retrieved sources: untrusted
2. Agent/model output: untrusted until validated
3. Local job workspace: semi-trusted, path constrained
4. Source-of-truth DB: trusted only through validated writes
5. Provider credentials: secret
6. Publishing credentials: highest sensitivity, separate from research credentials
7. Human approval token: privileged capability

## Security review checklist
- prompt injection isolation and regression tests
- SSRF controls on source fetcher
- URL scheme allowlist
- archive/PDF decompression limits
- malware handling for downloaded assets
- path traversal prevention
- secrets scanning and `.gitignore`
- OAuth scope minimization
- token encryption at rest
- credential rotation/revocation
- per-provider rate limits
- egress allowlisting where practical
- request/response logging with secret redaction
- dependency vulnerability scanning
- supply-chain pinning/lockfiles
- sandboxing for media processors
- FFmpeg input hardening and time/resource limits
- image/video parser resource limits
- webhook authenticity checks if used
- CSRF/session controls for desktop-local web UI
- local network binding review
- approval-token replay prevention
- publish idempotency
- per-platform emergency disable switch
- audit log immutability/tamper evidence

## Required security gates before live credentials
SEC-01 prompt-injection tests pass
SEC-02 source fetcher SSRF tests pass
SEC-03 secrets never written to logs or Git
SEC-04 publishing credentials isolated
SEC-05 approval tokens scoped and single-use
SEC-06 publishing connector supports dry run
SEC-07 asset provenance enforced
SEC-08 desktop app binds safely and authenticates privileged actions
