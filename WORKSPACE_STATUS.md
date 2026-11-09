# Workspace Status

This note records the baseline established on 2026-10-16 when the historical
working folder was connected directly to the GitHub repository.

## Git baseline

- Remote: `git@github.com:musicarroll/kalman_course.git`
- Branch: `main`, tracking `origin/main`
- Baseline commit: `5dcb815` (`Presentation for Zyfer`, 2024-11-11)
- Remote content at initialization: 158 tracked files in the curated `latex/`,
  `src/`, and `student_contrib/` trees plus published top-level PDFs.
- The remote checkout had no path collisions with the pre-existing local files,
  so no historical file was replaced.

## Historical local material

The workspace also contains a much larger pre-Git archive. At initialization it
contained about 1,850 files (160 MB). After standard build artifacts were
ignored, roughly 1,575 files remained available for deliberate review and selective
addition. They include full course-day source trees, graphics, Matlab code,
reference papers, old archives, and multiple generations of PDF output.

Do not bulk-add or bulk-delete this material. Some files appear to be generated,
duplicated, externally sourced, private, or historical backups. Migrate one
topic at a time and review both provenance and usefulness.

## Last visible work before the pause

File timestamps indicate these recent work areas:

- 2024-11-12: `basic_nav_concepts/nav_technologies.tex` and its PDF build.
- 2024-11-11: `gps_kalman_master_day3.tex` and its PDF build.
- 2024-10-16: state-space driver 4 and a transcript in
  `day1_seg4_state_space/`.
- 2024-08-30: the root `slide_template.tex`.
- 2024-07: substantial state-space presentation work, including drivers 1–4,
  dynamics examples, predator/prey material, and a DC motor example.

## Current production sequence

1. Review and record `basic_nav_concepts/assured_pnt_part1.pdf` using its
   matched voice-track transcript.
2. Develop Part 2 around alternative aiding sources and terrain-referenced
   navigation, with one worked measurement-model example.
3. Follow with Part 3 on sensor quality, clocks, calibration, and mission-level
   performance trades.
4. Reuse useful material from the broad 2024 `nav_technologies.tex` deck rather
   than replacing that historical source.
5. Commit each episode as a reviewed source, transcript, and published-PDF set.

## Initialization checks

- The tracked Python files in `src/` and `student_contrib/` pass Python bytecode
  compilation with the available Python 3.8.8 interpreter.
- `gps_kalman_master_day3.tex` was repaired for current Beamer/hyperref by
  replacing metadata-unsafe title formatting. It now completes a two-pass build
  with 291 pages and no unresolved references. Run `make day3` to reproduce it.
- The rebuilt `gps_kalman_master_day3.pdf` was published on 2026-10-16. The
  prior 2024 PDF is preserved as
  `archive/gps_kalman_master_day3_2024-11-11.pdf`.
- `basic_nav_concepts/assured_pnt_part1.tex` is a new 16:9, YouTube-first deck
  with 15 logical slides and 44 incremental-build pages. Its seven-page
  transcript includes exact build cues and a 17-minute-20-second timing plan.
  Run `make assured-pnt-part1-all` to reproduce and publish both PDFs.
