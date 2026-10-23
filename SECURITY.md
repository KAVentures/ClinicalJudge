# Security and privacy policy

Do not file public issues containing patient data, API keys, private clinical records or identifying clinician annotations. v0.1.0 is an experimental local research package, not certified PHI-processing infrastructure. It refuses remote endpoints by default, and only authorizes remote HTTPS endpoints with an explicit flag. Any external service may retain prompts and must be reviewed by the researcher.

Case content and model responses are treated as untrusted data, not tool instructions. The judge's text is never executed. No clinical judgment is made by Codex/Claude Code without an explicit independent backend. The initial parser requires exact JSON; it intentionally does not attempt to auto-repair or execute completions.

External contributors' SKILL.md changes and software dependencies must be reviewed before use. Only install trusted commits of this repo; skill instructions are executable agent workflow guidance.
