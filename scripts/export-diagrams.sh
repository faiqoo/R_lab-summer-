#!/usr/bin/env bash
#
# export-diagrams.sh — render the C4 architecture model to PDF (and SVG).
#
# Pipeline:  workspace.dsl
#              → structurizr export (PlantUML)   [docker: structurizr/structurizr]
#              → plantuml -tsvg                   [docker: plantuml/plantuml]
#              → rsvg-convert -f pdf              [local:  rsvg-convert]
#
# Produces one PDF per view plus a combined multi-page PDF, written to docs/export/.
#
# Usage:
#   scripts/export-diagrams.sh [--clean] [--workspace PATH] [--out DIR]
#
#   --clean            remove the .puml/.svg intermediates, keep only PDFs
#   --workspace PATH   DSL file to export   (default: docs/workspace.dsl)
#   --out DIR          output directory     (default: docs/export)
#
# Requirements: docker, rsvg-convert (brew install librsvg).

set -euo pipefail

# ── Resolve paths relative to this script, so it runs from anywhere ──────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKSPACE="$REPO_ROOT/docs/workspace.dsl"
OUT_DIR="$REPO_ROOT/docs/export"
CLEAN=0

# Combined-PDF page order (logical C4 order); falls back to all views found.
VIEW_ORDER=(structurizr-SystemContext structurizr-Containers structurizr-DrupalComponents)
COMBINED_NAME="islandora-architecture-c4.pdf"

# ── Parse args ──────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
  case "$1" in
    --clean)     CLEAN=1; shift ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --out)       OUT_DIR="$2"; shift 2 ;;
    -h|--help)   sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# ── Preflight ───────────────────────────────────────────────────────────────
command -v docker >/dev/null        || { echo "ERROR: docker not found." >&2; exit 1; }
command -v rsvg-convert >/dev/null  || { echo "ERROR: rsvg-convert not found (brew install librsvg)." >&2; exit 1; }
[ -f "$WORKSPACE" ]                 || { echo "ERROR: workspace not found: $WORKSPACE" >&2; exit 1; }

WORKSPACE_DIR="$(cd "$(dirname "$WORKSPACE")" && pwd)"
WORKSPACE_FILE="$(basename "$WORKSPACE")"
mkdir -p "$OUT_DIR"

echo "Workspace : $WORKSPACE"
echo "Output    : $OUT_DIR"
echo

# ── 1. DSL → PlantUML ───────────────────────────────────────────────────────
echo "[1/3] Structurizr DSL → PlantUML"
docker run --rm -v "$WORKSPACE_DIR:/work" -w /work \
  structurizr/structurizr export -workspace "$WORKSPACE_FILE" -format plantuml -output /work/.diagrams-tmp >/dev/null
# Move generated puml into the output dir
mv "$WORKSPACE_DIR/.diagrams-tmp"/*.puml "$OUT_DIR"/
rmdir "$WORKSPACE_DIR/.diagrams-tmp" 2>/dev/null || true

# ── 2. PlantUML → SVG ───────────────────────────────────────────────────────
echo "[2/3] PlantUML → SVG"
docker run --rm -v "$OUT_DIR:/data" plantuml/plantuml -tsvg "/data/*.puml" >/dev/null

# ── 3. SVG → PDF ────────────────────────────────────────────────────────────
echo "[3/3] SVG → PDF"
pdfs=()
# Render the main views in the preferred order first…
for name in "${VIEW_ORDER[@]}"; do
  if [ -f "$OUT_DIR/$name.svg" ]; then
    rsvg-convert -f pdf -o "$OUT_DIR/$name.pdf" "$OUT_DIR/$name.svg"
    pdfs+=("$OUT_DIR/$name.svg")
    echo "      $name.pdf"
  fi
done
# …then any other (non-legend) views the model may contain.
for svg in "$OUT_DIR"/*.svg; do
  base="$(basename "$svg" .svg)"
  case "$base" in *-key) continue ;; esac          # skip auto-generated legends
  [ -f "$OUT_DIR/$base.pdf" ] && continue           # already done above
  rsvg-convert -f pdf -o "$OUT_DIR/$base.pdf" "$svg"
  pdfs+=("$svg")
  echo "      $base.pdf"
done

# Combined multi-page PDF (all views in one file)
if [ ${#pdfs[@]} -gt 0 ]; then
  rsvg-convert -f pdf -o "$OUT_DIR/$COMBINED_NAME" "${pdfs[@]}"
  echo "      $COMBINED_NAME (combined, ${#pdfs[@]} pages)"
fi

# ── Optional cleanup ────────────────────────────────────────────────────────
if [ "$CLEAN" = 1 ]; then
  rm -f "$OUT_DIR"/*.puml "$OUT_DIR"/*.svg
  echo
  echo "Cleaned intermediates (.puml/.svg)."
fi

echo
echo "Done. PDFs in: $OUT_DIR"
