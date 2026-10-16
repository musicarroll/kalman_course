PDFLATEX ?= pdflatex
BUILD_DIR ?= build/day3
DAY3_TEX := gps_kalman_master_day3.tex
DAY3_PDF := $(BUILD_DIR)/gps_kalman_master_day3.pdf
DAY3_JOB := day3_build
DAY3_WORK_PDF := $(BUILD_DIR)/$(DAY3_JOB).pdf

PNT_BUILD_DIR ?= build/assured_pnt_part1
PNT_DECK_TEX := basic_nav_concepts/assured_pnt_part1.tex
PNT_DECK_JOB := assured_pnt_part1_build
PNT_DECK_WORK_PDF := $(PNT_BUILD_DIR)/$(PNT_DECK_JOB).pdf
PNT_DECK_PDF := basic_nav_concepts/assured_pnt_part1.pdf
PNT_TRANSCRIPT_TEX := basic_nav_concepts/assured_pnt_part1_transcript.tex
PNT_TRANSCRIPT_JOB := assured_pnt_part1_transcript_build
PNT_TRANSCRIPT_WORK_PDF := $(PNT_BUILD_DIR)/$(PNT_TRANSCRIPT_JOB).pdf
PNT_TRANSCRIPT_PDF := basic_nav_concepts/assured_pnt_part1_transcript.pdf

.PHONY: day3 assured-pnt-part1 assured-pnt-part1-transcript assured-pnt-part1-all
day3:
	@mkdir -p "$(BUILD_DIR)"
	@$(PDFLATEX) -jobname="$(DAY3_JOB)" -file-line-error \
		-interaction=nonstopmode -halt-on-error \
		-output-directory="$(BUILD_DIR)" "$(DAY3_TEX)" \
		>"$(BUILD_DIR)/pass1.txt" 2>&1 || \
		{ tail -50 "$(BUILD_DIR)/pass1.txt"; exit 1; }
	@$(PDFLATEX) -jobname="$(DAY3_JOB)" -file-line-error \
		-interaction=nonstopmode -halt-on-error \
		-output-directory="$(BUILD_DIR)" "$(DAY3_TEX)" \
		>"$(BUILD_DIR)/pass2.txt" 2>&1 || \
		{ tail -50 "$(BUILD_DIR)/pass2.txt"; exit 1; }
	@cp "$(DAY3_WORK_PDF)" "$(DAY3_PDF)"
	@echo "Built $(DAY3_PDF)"

assured-pnt-part1:
	@mkdir -p "$(PNT_BUILD_DIR)"
	@$(PDFLATEX) -jobname="$(PNT_DECK_JOB)" -file-line-error \
		-interaction=nonstopmode -halt-on-error \
		-output-directory="$(PNT_BUILD_DIR)" "$(PNT_DECK_TEX)" \
		>"$(PNT_BUILD_DIR)/deck-pass1.txt" 2>&1 || \
		{ tail -50 "$(PNT_BUILD_DIR)/deck-pass1.txt"; exit 1; }
	@$(PDFLATEX) -jobname="$(PNT_DECK_JOB)" -file-line-error \
		-interaction=nonstopmode -halt-on-error \
		-output-directory="$(PNT_BUILD_DIR)" "$(PNT_DECK_TEX)" \
		>"$(PNT_BUILD_DIR)/deck-pass2.txt" 2>&1 || \
		{ tail -50 "$(PNT_BUILD_DIR)/deck-pass2.txt"; exit 1; }
	@cp "$(PNT_DECK_WORK_PDF)" "$(PNT_DECK_PDF)"
	@echo "Built $(PNT_DECK_PDF)"

assured-pnt-part1-transcript:
	@mkdir -p "$(PNT_BUILD_DIR)"
	@$(PDFLATEX) -jobname="$(PNT_TRANSCRIPT_JOB)" -file-line-error \
		-interaction=nonstopmode -halt-on-error \
		-output-directory="$(PNT_BUILD_DIR)" "$(PNT_TRANSCRIPT_TEX)" \
		>"$(PNT_BUILD_DIR)/transcript-pass1.txt" 2>&1 || \
		{ tail -50 "$(PNT_BUILD_DIR)/transcript-pass1.txt"; exit 1; }
	@$(PDFLATEX) -jobname="$(PNT_TRANSCRIPT_JOB)" -file-line-error \
		-interaction=nonstopmode -halt-on-error \
		-output-directory="$(PNT_BUILD_DIR)" "$(PNT_TRANSCRIPT_TEX)" \
		>"$(PNT_BUILD_DIR)/transcript-pass2.txt" 2>&1 || \
		{ tail -50 "$(PNT_BUILD_DIR)/transcript-pass2.txt"; exit 1; }
	@cp "$(PNT_TRANSCRIPT_WORK_PDF)" "$(PNT_TRANSCRIPT_PDF)"
	@echo "Built $(PNT_TRANSCRIPT_PDF)"

assured-pnt-part1-all: assured-pnt-part1 assured-pnt-part1-transcript
