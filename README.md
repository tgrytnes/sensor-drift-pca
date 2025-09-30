# MLOps Project Template

This repository is a reusable MLOps template for AI and data science projects.  
It provides a config-driven, hardware-independent pipeline structure with clear separation of source code, configs, data, and reports.  
The goal is to standardize project setup, ensure reproducibility, and accelerate development across different ML projects.

## How to use this template
- Do not develop directly in this repository.
- Click “Use this template” on GitHub to create a fresh project repo.
- Initialize and rename the new project with the initializer script (see below).
- Keep this repo unchanged as your base template.

## Environment & infrastructure
For environment setup and infrastructure instructions (Docker images, RunPod templates, etc.), please see the dedicated infrastructure repository.

---

## Project initialization (rename & scaffold update)

Use the initializer to set your import package, distribution name, human-readable title, and optionally register a Jupyter kernel. Run it once, right after creating your new repo from this template.

### Requirements
- Python 3.10+ available as `python3`
- Clean working tree (no uncommitted changes)
- Run from the project root (where `pyproject.toml` and `scripts/` live)

### Usage (CLI)
    python3 scripts/init_project.py --help

### Options (reflects the committed script)
| Option | Required | Default | Description |
|-------|----------|---------|-------------|
| `--package NAME` | yes | — | New Python import package under `src/NAME` (renames `src/yourproj` and rewrites imports/placeholders). Lowercase, letters/numbers/underscore. |
| `--dist-name NAME` | no | same as `--package` | Distribution/project name written to packaging metadata (`pyproject.toml`). Can include dashes. |
| `--title TEXT` | no | Titleized `dist-name` | Human-readable project title injected into README/notebooks/report stubs. |
| `--kernel-name NAME` | no | (skip) | If provided, registers a Jupyter kernel (via `ipykernel`) for this project’s venv. |
| `--author TEXT` | no | (leave as template) | Optional author override for metadata files if present. |
| `--repo-url URL` | no | (leave as template) | Optional repository URL for metadata badges. |
| `--force` | no | false | Proceed even if non-empty target paths exist (use with care). |

### What it changes
- Renames the template package: `src/yourproj` → `src/<package>` and fixes internal `yourproj` imports.
- Updates packaging metadata in `pyproject.toml`: project name (`dist-name`), optional author/URL, and console-script placeholder if present.
- Rewrites placeholders (project title, import path) in:
  - `README.md` header stubs and any `docs/` stubs found,
  - `scripts/` helpers referencing `yourproj`,
  - `configs/` that include module paths,
  - example notebooks in `notebooks/` (lightweight text replace in headings/metadata if applicable).
- Optionally registers a Jupyter kernel named `--kernel-name`.
- Leaves git history intact and does not touch your data/ or reports/.

### Safety checks
- Aborts if there are uncommitted changes unless `--force` is set.
- Validates `--package` is a valid Python identifier.
- Creates missing `src/<package>/__init__.py` if needed.

### Examples

Minimal (rename only)
    python3 scripts/init_project.py --package hcr

Explicit dist name and title
    python3 scripts/init_project.py --package hcr --dist-name hcr --title "Hotel Cancellation Risk"

Register a Jupyter kernel too
    python3 scripts/init_project.py --package hcr --kernel-name hcr-venv

Tip: Run this on a clean working tree. If you need to re-run, use `git restore -SW :/` (or `git reset --hard` if you’re sure) to revert changes first.

---

## Quickstart (after initialization)

Once your development environment is configured (e.g., using the Docker image from the infra repo), run the training pipeline with a config file:

Run the full pipeline using the baseline experiment config
    bash scripts/run_train.sh configs/exp_baseline.yaml

View logs and outputs
    tree -L 2 reports/ runs/

Common iterative workflow
    # 1) Edit/clone a config under configs/
    # 2) Commit the config change
    # 3) Launch: bash scripts/run_train.sh configs/your_experiment.yaml
    # 4) Compare metrics/artifacts in reports/ and runs/

---

## Data & large files
- Do not commit large artifacts to Git. Use object storage or DVC-style remotes.
- `data/` stays out of version control (a `.gitignore` is provided).
- Prefer stable, versioned datasets and record data provenance in `reports/` or `runs/`.

---

## Reproducibility
- Prefer config-only changes over code changes for experiments.
- Always set/record seeds in configs; the pipeline logs seeds and environment details.
- Capture package versions (`pip freeze` or `uv export`) into `reports/env.txt` on each run.

---

## Repository layout (post-init)
    .
    ├── configs/                 # Experiment configs (YAML)
    ├── data/                    # Local data (ignored by Git)
    ├── notebooks/               # Optional EDA / reports (title updated by init)
    ├── reports/                 # Metrics, charts, artifacts written by runs
    ├── runs/                    # Per-run logs, params, env snapshots
    ├── scripts/
    │   ├── init_project.py      # Initializer you ran
    │   └── run_train.sh         # Pipeline entrypoint
    ├── src/
    │   └── <package>/           # Your import package (renamed from yourproj)
    ├── tests/                   # Unit/integration tests
    ├── pyproject.toml           # Packaging & tool config (name updated)
    └── README.md                # This file (title updated)

---

## Troubleshooting

Initializer refuses to run on dirty working tree
    git status
    git commit -m "WIP" || git stash
    # or re-run with --force if you understand the impact

Imports still reference `yourproj`
    rg -n "yourproj" -S
    # If anything remains, adjust manually or re-run the initializer

Kernel not visible in Jupyter
    python3 -m ipykernel install --user --name <kernel-name>
    # Then select it in your notebook UI

---

## License
This template is distributed under the license specified in `LICENSE` in this repository.
