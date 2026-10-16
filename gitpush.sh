#!/usr/bin/env bash
# Commit reviewed files, rebase on the matching origin branch, and push.
#
# Usage:
#   ./gitpush.sh --check
#   ./gitpush.sh "Commit message"
#   ./gitpush.sh "Commit message" -- path/to/new-file another/new-file
#
# Existing tracked files are staged automatically. Because this workspace has a
# large historical archive, untracked files are included only when named after
# "--". This intentionally avoids the unconditional `git add -A` used by some
# smaller repositories.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./gitpush.sh --check
  ./gitpush.sh "Commit message"
  ./gitpush.sh "Commit message" -- path/to/new-file [more-new-files ...]

With a commit message, all modifications and deletions to already tracked files
are staged. Name each reviewed untracked file after "--" to include it.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"

[[ -n "$REPO_ROOT" ]] || die "gitpush.sh is not inside a Git repository."
[[ "$SCRIPT_DIR" == "$REPO_ROOT" ]] || die "gitpush.sh must remain at the repository root."

cd "$REPO_ROOT"

ORIGIN_URL="$(git remote get-url origin 2>/dev/null || true)"
[[ -n "$ORIGIN_URL" ]] || die "the origin remote is not configured."

case "$ORIGIN_URL" in
  git@*:musicarroll/kalman_course.git|ssh://git@*/musicarroll/kalman_course.git) ;;
  *) die "origin is not the SSH remote for musicarroll/kalman_course: $ORIGIN_URL" ;;
esac

if ! BRANCH="$(git symbolic-ref --quiet --short HEAD)"; then
  die "cannot push from a detached HEAD; check out a branch first."
fi

check_connection() {
  echo "Repository: $REPO_ROOT"
  echo "Branch:     $BRANCH"
  echo "Origin:     $ORIGIN_URL"

  echo "Checking SSH read access ..."
  git ls-remote --exit-code origin HEAD >/dev/null
  echo "SSH authentication and GitHub repository read access are working."
}

if [[ "${1:-}" == "--check" ]]; then
  [[ $# -eq 1 ]] || die "--check does not accept additional arguments."
  check_connection
  echo "Checking SSH push permission (dry run) ..."
  git push --dry-run origin "$BRANCH" >/dev/null
  echo "GitHub push permission is working."
  exit 0
fi

[[ $# -ge 1 ]] || { usage; exit 1; }
COMMIT_MSG="$1"
shift
[[ -n "$COMMIT_MSG" ]] || die "the commit message cannot be empty."

NEW_PATHS=()
if [[ $# -gt 0 ]]; then
  [[ "$1" == "--" ]] || { usage; die "put reviewed untracked paths after --"; }
  shift
  [[ $# -gt 0 ]] || die "name at least one path after --."
  NEW_PATHS=("$@")
fi

for candidate in "${NEW_PATHS[@]}"; do
  case "$candidate" in
    .|./|"$REPO_ROOT")
      die "refusing the repository root as a path; name reviewed files explicitly."
      ;;
  esac
  [[ -e "$candidate" || -L "$candidate" ]] || die "path does not exist: $candidate"
done

check_connection

echo "Staging changes to tracked files ..."
git add -u

if [[ ${#NEW_PATHS[@]} -gt 0 ]]; then
  echo "Staging ${#NEW_PATHS[@]} reviewed untracked path(s) ..."
  git add -- "${NEW_PATHS[@]}"
fi

UNTRACKED_COUNT="$(git ls-files --others --exclude-standard | wc -l)"
if [[ "$UNTRACKED_COUNT" -gt 0 ]]; then
  echo "Leaving $UNTRACKED_COUNT other untracked file(s) out of this commit."
fi

if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  echo "Files selected for commit:"
  git diff --cached --name-status
  echo "Committing ..."
  git commit -m "$COMMIT_MSG"
fi

echo "Fetching origin ..."
git fetch origin

if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  echo "Rebasing on origin/$BRANCH ..."
  if ! git rebase "origin/$BRANCH"; then
    echo >&2
    echo "Rebase has conflicts. Resolve them, then run:" >&2
    echo "  git add <fixed-files> && git rebase --continue" >&2
    echo "After the rebase completes, run:" >&2
    echo "  git push -u origin $BRANCH" >&2
    exit 1
  fi
else
  echo "origin/$BRANCH does not exist; this will be its first push."
fi

echo "Pushing $BRANCH ..."
git push -u origin "$BRANCH"
echo "Success."
