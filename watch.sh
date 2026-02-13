#!/usr/bin/env bash
# Watch `data.tex` and rebuild `paper.tex` using pLaTeX + dvipdfmx.
# Uses inotifywait if available, otherwise falls back to a 1s polling loop.
set -euo pipefail
cd "$(dirname "$0")"

TARGET_TEX=paper.tex
WATCH_FILE=data.tex

build() {
	echo "[watch] Building $TARGET_TEX ..."
	platex -interaction=nonstopmode "$TARGET_TEX"
	dvipdfmx "${TARGET_TEX%.tex}.dvi"
	echo "[watch] Built ${TARGET_TEX%.tex}.pdf"
}

build

if command -v inotifywait >/dev/null 2>&1; then
	echo "[watch] Watching $WATCH_FILE (inotifywait)..."
	while inotifywait -e close_write,modify "$WATCH_FILE" >/dev/null 2>&1; do
		build
	done
else
	echo "[watch] inotifywait not found; falling back to polling (1s)."
	last_mod=$(stat -c %Y "$WATCH_FILE")
	while true; do
		sleep 1
		new_mod=$(stat -c %Y "$WATCH_FILE")
		if [ "$new_mod" != "$last_mod" ]; then
			last_mod=$new_mod
			build
		fi
	done
fi
