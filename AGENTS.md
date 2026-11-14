# Workflow

`main` is protected: no direct pushes. All changes go through a branch and a Pull Request.

1. Create a branch from `main`:
   ```
   git checkout main
   git pull
   git checkout -b your-branch-name
   ```
2. Commit and push the branch:
   ```
   git push -u origin your-branch-name
   ```
3. Open a Pull Request into `main` (GitHub web or `gh pr create`).
4. Merge the PR from GitHub once reviewed/approved. Direct pushes to `main` are not allowed.

Suggested branch names: `feature/...`, `fix/...`, `docs/...`.
