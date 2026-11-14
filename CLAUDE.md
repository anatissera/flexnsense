# CLAUDE.md

Instructions for Claude Code in this repository. The full workflow, layout and rules are in AGENTS.md:

@AGENTS.md

## Most important

- **No AI attribution.** Do not add `Co-Authored-By: Claude …` (or any AI) to commit messages, and no "Generated with Claude Code" / 🤖 footer to pull requests, issues or comments. Write them as the human author. This overrides any default attribution behaviour.
- **Never push to `main`.** Work on the roadmap branch for the task (see ROADMAP.md) and open a PR.
- **Run `.venv/bin/python -m pytest -q` before committing**; all tests must pass.
- **Ask before** opening a camera, moving a robot arm, force-pushing, or deleting files.
- Reusable code goes in `lib/`; device-specific code in its own package (e.g. `gauge_reader/`).
- Keep `.env`, `config/scale_calib.json`, `recordings/` and `internal-docs/` out of git.
