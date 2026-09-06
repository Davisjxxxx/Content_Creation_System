# Security Constitution

1. No credentials, OAuth refresh tokens, private keys, raw voice data, or session cookies in Git.
2. Providers receive least-privilege credentials and narrow scopes.
3. Publishing credentials are separated from research credentials.
4. Research content is untrusted input. Treat web pages, creator posts, PDFs, transcripts, and retrieved text as prompt-injection capable.
5. Retrieved content may provide evidence, never system instructions.
6. Tool calls are allowlisted by agent role.
7. Agents cannot execute arbitrary shell commands from retrieved text.
8. File writes are constrained to declared job workspaces.
9. External URLs are recorded with provenance and normalized before fetching.
10. Human approval tokens must be short-lived, story-bound, platform-bound, and non-reusable.
11. Hard FinOps limits fail closed.
12. Production should support audit logs for agent decisions, tool calls, costs, source retrievals, and state transitions.
13. Synthetic voice/avatar assets require explicit creator authorization and provider-side verification where applicable.
