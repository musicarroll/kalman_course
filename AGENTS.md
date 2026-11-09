# Codex Repository Guide

## Purpose

This repository develops PDF presentations and supporting Python/Matlab
examples for a Kalman filter course. Preserve the mathematical intent and the
connection between source examples, figures, TeX slides, and published PDFs.
Read `WORKSPACE_STATUS.md` before starting structural or migration work.

## Safety and scope

- Never use or recommend `git clean`. This workspace contains valuable
  historical material that is not yet tracked.
- Do not delete, rename, or reorganize historical files unless the user asks
  for a specific migration.
- Do not bulk-add the entire legacy workspace. Review additions by topic and
  distinguish authored source from generated output and third-party reference
  material.
- Do not commit or push unless the user explicitly asks.
- Never commit credentials, tokens, private correspondence, or licensed source
  material without confirming that it belongs in the public repository.

## Canonical layout

- `latex/`: curated Beamer sources.
- `src/`: maintained Python examples; `src/matlab/` contains Matlab references.
- `student_contrib/`: contributed programs and figures.
- Top-level PDFs: intentional published course artifacts.
- Other top-level and topic directories: historical working material pending
  selective review and migration.

## Working conventions

- Make focused edits and preserve the existing instructional voice and slide
  sequence.
- Prefer repository-relative TeX paths; replace old absolute Windows/Cygwin
  paths when touching a driver.
- Build TeX drivers from their containing directory because many use relative
  `\input` and graphics paths.
- Keep generated PDFs only when they are intended deliverables. Do not add
  `.aux`, `.log`, `.nav`, `.out`, `.snm`, `.toc`, or SyncTeX files.
- Keep Python examples runnable as scripts unless a broader packaging change is
  requested. Use four-space indentation and descriptive snake_case names for
  new code.
- When a Python example generates a figure consumed by TeX, document or retain
  the relationship in the script and verify both the program and presentation.

## Validation

- Python: run the changed script with a non-interactive Matplotlib backend when
  practical, for example `MPLBACKEND=Agg python example.py`.
- TeX: run `pdflatex -interaction=nonstopmode -halt-on-error DRIVER.tex` twice
  from the driver's directory, then inspect the PDF for warnings or regressions.
- Day 3 master deck: run `make day3`; outputs stay under ignored `build/day3/`.
- Before handoff, use `git status --short` and report tracked changes separately
  from historical untracked files.
