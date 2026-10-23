# Using ClinicalJudge from coding agents

For Codex, the project skill is `.agents/skills/clinicaljudge/SKILL.md`; for Claude Code use `.claude/skills/clinicaljudge/SKILL.md`. The same skill contract is copied to both.

For a different workspace, copy or symlink that directory into the workspace's matching skill directory and make the ClinicalJudge package available (e.g. `python -m pip install -e /path/to/clinicaljudge`). Start Codex or Claude Code inside your project workspace so it discovers the skill. Alternatively work in a clone of ClinicalJudge with research case fixtures added under a separately gitignored data directory.

Example request:

> Use the clinicaljudge skill to inspect this project's evaluation dataset, propose (but do not change) the frozen rubric, run the separate independent Qwen judge against anonymized responses, and generate a case-clustered physician-comparison report. If external physician validation is missing, label the result unvalidated; do not let the coding agent score it.

If the backend is not running or no physician reference exists, the agent must tell the researcher. It must never invent a remote endpoint, human ratings, validation statistics, or silently grade answers using its own language model.
