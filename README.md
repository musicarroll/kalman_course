# Kalman Filter Theory and Applications

This repository contains the presentation sources, published PDF slide decks,
and numerical examples for Michael L. Carroll's Kalman filter course. The
material covers the heuristic predictor/corrector view, equations and notation,
state-space models, navigation, and implementation examples.

## Repository layout

- `latex/` — curated Beamer source used for the course modules.
- `src/` — Python demonstrations and legacy MATLAB reference programs.
- `student_contrib/` — contributed examples and their generated figures.
- Top-level `*.pdf` files — published presentation artifacts.
- Historical working directories — older source, graphics, references, and
  experiments retained in the local workspace while they are selectively
  migrated into the curated layout.

## Python setup

Use an isolated environment for the numerical examples:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Most examples are standalone scripts. Run them from their containing directory
so that imports such as `collate` and `movavgs` resolve correctly:

```bash
cd src
python constant_velocity.py
```

## Building presentations

The presentations use LaTeX Beamer. Compile a driver from its containing
directory so its relative `\input` and graphics paths are preserved:

```bash
cd path/to/the/driver
pdflatex -interaction=nonstopmode -halt-on-error DRIVER.tex
pdflatex -interaction=nonstopmode -halt-on-error DRIVER.tex
```

The repaired Day 3 master deck has a repeatable two-pass build that writes all
intermediate files under the ignored `build/` directory:

```bash
make day3
```

The YouTube-first assured-PNT series begins with a 16:9 slide deck and a
recording script whose cues follow the slide builds. Build and publish both
PDFs with:

```bash
make assured-pnt-part1-all
```

The command writes temporary LaTeX files below
`build/assured_pnt_part1/` and publishes the two PDFs beside their sources in
`basic_nav_concepts/`.

The current source archive includes drivers with machine-specific paths and a
curated `latex/` tree whose shared includes and graphics have not all been
migrated yet. Treat those as migration work: collect dependencies into the
repository, replace absolute paths with repository-relative paths, and verify
the resulting PDF before committing it.

## Version-control workflow

The `main` branch tracks
[`musicarroll/kalman_course`](https://github.com/musicarroll/kalman_course).
Commit source changes together with any intentionally refreshed course PDF.
LaTeX intermediate files and Python caches are ignored. Historical local files
that have not yet been curated should be reviewed deliberately before they are
added; do not bulk-delete or bulk-add them.

Use the repository push helper to verify SSH access or publish reviewed work:

```bash
./gitpush.sh --check
./gitpush.sh "Describe changes to tracked files"
./gitpush.sh "Publish a new deck" -- path/to/new-source.tex path/to/new-deck.pdf
```

Changes to existing tracked files are included automatically. New, untracked
files must be named after `--`. This differs intentionally from helpers that
run `git add -A`: the historical workspace contains many files that are not yet
approved for publication. The helper fetches and rebases on the matching
remote branch before pushing.

See [WORKSPACE_STATUS.md](WORKSPACE_STATUS.md) for the current migration
baseline and likely restart points.

## License

The repository is distributed under the [MIT License](LICENSE).
