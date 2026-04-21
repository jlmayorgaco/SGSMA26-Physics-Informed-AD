#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-.}"
OUT="${2:-project_context_report.txt}"

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: root folder does not exist: $ROOT" >&2
  exit 1
fi

ROOT="$(cd "$ROOT" && pwd)"

: > "$OUT"

section() {
  {
    printf "\n"
    printf "============================================================\n"
    printf "%s\n" "$1"
    printf "============================================================\n"
  } >> "$OUT"
}

is_text_like() {
  local f="$1"
  case "${f,,}" in
    *.py|*.txt|*.md|*.rst|*.json|*.yaml|*.yml|*.toml|*.ini|*.cfg|*.conf|*.csv|*.ts|*.tsx|*.js|*.jsx|*.java|*.c|*.cpp|*.h|*.hpp|*.cs|*.go|*.rs|*.sh|*.bash|*.ps1|*.sql|*.tex|*.html|*.css|*.scss|*.xml|*.dockerfile|*dockerfile|*.gitignore)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

file_size_bytes() {
  wc -c < "$1" | tr -d ' '
}

append_file_preview() {
  local f="$1"
  local rel="$2"
  local size
  size="$(file_size_bytes "$f")"

  {
    printf "\n------------------------------------------------------------\n"
    printf "FILE: %s\n" "$rel"
    printf "SIZE_BYTES: %s\n" "$size"
    printf "------------------------------------------------------------\n"
  } >> "$OUT"

  if (( size > 300000 )); then
    printf "[SKIPPED CONTENT PREVIEW: file too large]\n" >> "$OUT"
    return
  fi

  awk 'NR<=160 { printf "%5d | %s\n", NR, $0 } NR==161 { print "[TRUNCATED AFTER 160 LINES]"; exit }' "$f" >> "$OUT" || {
    printf "[FAILED TO READ FILE AS TEXT]\n" >> "$OUT"
  }
}

section "PROJECT SCAN METADATA"
{
  printf "ROOT: %s\n" "$ROOT"
  printf "GENERATED_AT_UTC: "
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  printf "HOST_OS: Windows via Git Bash / MINGW / MSYS\n"
  printf "IGNORED: .git/\n"
} >> "$OUT"

section "FULL DIRECTORY AND FILE TREE"
(
  cd "$ROOT"
  find . -path './.git' -prune -o -mindepth 1 -print | LC_ALL=C sort
) >> "$OUT"

section "DIRECTORIES ONLY"
(
  cd "$ROOT"
  find . -path './.git' -prune -o -type d -print | LC_ALL=C sort
) >> "$OUT"

section "FILES ONLY"
(
  cd "$ROOT"
  find . -path './.git' -prune -o -type f -print | LC_ALL=C sort
) >> "$OUT"

section "FILES WITH SIZE IN BYTES"
while IFS= read -r -d '' f; do
  rel="${f#$ROOT/}"
  size="$(file_size_bytes "$f")"
  printf "%12s  %s\n" "$size" "$rel"
done < <(find "$ROOT" -path "$ROOT/.git" -prune -o -type f -print0 | sort -z) >> "$OUT"

section "TEXT / CODE / CONFIG FILE CONTENT PREVIEW"
while IFS= read -r -d '' f; do
  rel="${f#$ROOT/}"
  if is_text_like "$f"; then
    append_file_preview "$f" "$rel"
  fi
done < <(find "$ROOT" -path "$ROOT/.git" -prune -o -type f -print0 | sort -z)

section "SUMMARY"
dir_count="$(find "$ROOT" -path "$ROOT/.git" -prune -o -type d -print | wc -l | tr -d ' ')"
file_count="$(find "$ROOT" -path "$ROOT/.git" -prune -o -type f -print | wc -l | tr -d ' ')"
{
  printf "TOTAL_DIRECTORIES: %s\n" "$dir_count"
  printf "TOTAL_FILES: %s\n" "$file_count"
  printf "REPORT: %s\n" "$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
} >> "$OUT"

echo "Done. Report created at: $OUT"