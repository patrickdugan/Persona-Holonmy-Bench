# Publish this starter repository to GitHub

This archive contains the source tree. Publishing requires GitHub authentication on your own machine.

1. Extract the archive and open a terminal in the `persona-holonomy-bench` directory.
2. Install [GitHub CLI](https://cli.github.com/) if needed, and run `gh auth login` once.
3. Run:

```bash
git init -b main
git add .
git commit -m "Initial Persona Holonomy Bench v0.1"
gh repo create patrickdugan/persona-holonomy-bench --public --source . --remote origin --push
```

If the remote repo already exists, skip `gh repo create` and use:

```bash
git remote add origin https://github.com/patrickdugan/persona-holonomy-bench.git
git push -u origin main
```

A GitHub account with write access to `patrickdugan` is required. Review the source before publishing. The `runs/` directory is ignored, and there are no API keys in this archive.
