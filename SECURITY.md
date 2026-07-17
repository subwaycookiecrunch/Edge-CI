# Security

## Self-hosted runner warnings

EdgeCI runs on self-hosted Apple Silicon Macs. If you expose a persistent Mac as a GitHub Actions runner:

- **Never auto-run untrusted code.** Fork PRs execute on your hardware after environment approval. Use a dedicated runner with no personal files, SSH keys, browser sessions, or unrelated credentials.
- Restrict the runner group to specific repositories and workflows.
- Use a non-admin macOS user for the runner service.
- The benchmark job should only have `contents: read`. PR comment posting belongs in a separate hosted job with `pull-requests: write`.
- Pin third-party Actions by full commit SHA, not tags.

GitHub's own documentation warns that self-hosted runners can be persistently compromised by malicious workflows.

## Reporting a vulnerability

If you find a security issue in EdgeCI itself (not a general self-hosted runner concern), email the maintainer directly rather than opening a public issue. Include steps to reproduce if possible.

Response target: acknowledgment within 72 hours, fix or mitigation plan within two weeks.
