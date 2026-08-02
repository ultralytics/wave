# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

Ultralytics WAVE (WAveform Vector Exploitation, AGPL-3.0) is the reference implementation for [arXiv:1811.05875](https://arxiv.org/abs/1811.05875): deep networks that read out and reconstruct signals from full-waveform time-of-flight particle detectors, regressing interaction position and time directly from digitized waveform pairs. It is research code — small standalone training scripts in PyTorch and TensorFlow, no package and no tests.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
python -m pip install -U -r requirements.txt # numpy, scipy, torch, tensorflow, plotly
python train.py --epochs 5000 --var 3        # PyTorch training; --var selects the model (0=WAVE, 2/3/4=WAVE2/3/4)
python train_tf.py                           # TensorFlow 1.x equivalent, requires eager-execution TF 1.x
```

`train.py` downloads `data/wavedata25ns.mat` with `wget` from `https://storage.googleapis.com/ultralytics/` on first run and writes `results/<name>.mat`. There is no test suite; CI is `.github/workflows/format.yml` (Ruff, docformatter, Prettier, codespell auto-applied to PR branches) and `cla.yml`.

## Architecture

- `train.py` is the PyTorch entry point and holds the four model variants. Inputs are `nx512` (two concatenated 256-sample waveforms), outputs are the first two of `[position (mm), time (ns), PE, E (MeV)]`. Each row of `x` is normalized independently and each output column is normalized across the dataset, then split 70/15/15 without shuffling. `WAVE` is a 3-layer tanh MLP; `WAVE2`/`WAVE3`/`WAVE4` reshape the input to `[bs, 2, 256]` and use strided 1D-style convolutions. Training runs full-batch-ish with Adam, `ReduceLROnPlateau`, and the `patienceStopper` early stop.
- `utils/utils.py` holds the framework-agnostic pieces shared by all three scripts: `normalize`, `splitdata`, `shuffledata`, `model_info`, `stdpt`/`stdtf`, and `patienceStopper` (tracks the best model and prints the epoch table). `utils/torch_utils.py` has `init_seeds` and `select_device`.
- `train_tf.py` is a TensorFlow 1.x eager-mode port of the same experiment (`tf.enable_eager_execution`, `tf.set_random_seed`) and plots with Plotly. It will not run on TensorFlow 2.x.
- `gcp/wave_pytorch_gcp.py` is an older standalone PyTorch variant used for cloud sweeps, with `gcp/vminstall.sh` and `gcp/vmstart.sh` provisioning a GPU VM. It duplicates much of `train.py`; prefer changing `train.py` and only touch the GCP copy when the cloud sweep itself changes.

## Conventions

- Every Python and shell file starts with the `Ultralytics 🚀 AGPL-3.0 License` header — Ultralytics Actions adds it automatically; don't add or revert them manually.
- Model result tables are kept as trailing comment blocks at the bottom of the training scripts; when a model's numbers change, update the block rather than deleting it.
- The three training scripts are intentionally independent; share code through `utils/` rather than importing one script from another.
