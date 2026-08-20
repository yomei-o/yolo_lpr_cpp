#!/bin/sh
# Rebuild every dataset this project trains on. `data/` is deliberately NOT in git — the synthetic
# parts are a pure function of a seed (committing 30000 PNGs would just be a slow copy of this file),
# and the real parts are other people's photos, which we are not in a position to redistribute.
#
#   sh tools/make_data.sh            # everything (needs ../alpr_jp for the real-plate steps)
#   sh tools/make_data.sh synth      # just the recognizer sets
#   sh tools/make_data.sh det        # just the detector sets
#
# The seeds below are the ones the shipped models were actually trained on. Change them and you get a
# different dataset, which is fine — but then the numbers in README no longer describe your models.
# Both languages generate identical bytes for the same seed (tools/parity/gen.py), so
# `python tools/gen.py` can be substituted for `./jlpr gen` anywhere here.
set -e
cd "$(dirname "$0")/.."
JLPR=${JLPR:-./jlpr}
[ -x "$JLPR" ] || JLPR=./jlpr.exe
# --fonts-strict = only the faces in spec/fonts.txt, i.e. exactly what the shipped models were trained
# with. Without it, a machine that also has the Windows faces installed silently produces a different
# dataset from the same seed (same labels, different pixels), and the published numbers stop meaning
# anything. Drop it if you deliberately want more letter shapes — but then re-measure everything.
STRICT=${STRICT:---fonts-strict}
WHAT=${1:-all}

# The real data. 720 cropped plates with region labels, plus vehicle photos and plate-free negatives.
# Public repository, but not vendored here: it is not ours, and plate photos are the kind of data that
# should stay at its source. Everything below that mentions ../alpr_jp needs this clone.
if [ ! -d ../alpr_jp ]; then
  echo "note: ../alpr_jp is missing -> real plates, pseudo-labels and the recognizer's real split are skipped"
  echo "      git clone --depth 1 https://github.com/dyama/alpr_jp.git ../alpr_jp"
fi
REAL_PLATES='../alpr_jp/自家用,../alpr_jp/自家用(軽),../alpr_jp/事業用,../alpr_jp/事業用(軽)'

if [ "$WHAT" = all ] || [ "$WHAT" = synth ]; then
  echo "== recognizer: 30000 train + 2000 val crops"
  $JLPR gen --out data/synth     --count 30000 --seed 90210 $STRICT --quiet
  $JLPR gen --out data/synth_val --count 2000  --seed 555001 $STRICT --quiet
  # Names the real data has none of get oversampled into the SAME directory (--start appends), which
  # keeps one dataset and one weight: 30000 uniform + 6000 of the five 2025 additions puts them at
  # ~17% of samples instead of 3.6%. They start from zero weights, so they need the extra exposure —
  # measured: at 3.6% 江戸川 stayed unreachable, at 10% it became top-1 on 1 of 4 crops.
  echo "== recognizer: +6000 crops of the 2025 additions (regions 133-137)"
  $JLPR gen --out data/synth --count 6000 --start 30000 --seed 90210 --region 133-137 $STRICT --quiet
  echo "== region coverage test set: 2 crops per region name (tools/check_regions.py)"
  $JLPR gen --out data/region_sweep --count 276 --region sweep --seed 4242 $STRICT --quiet
fi

if [ "$WHAT" = all ] || [ "$WHAT" = det ]; then
  if [ -d ../alpr_jp ]; then
    # Composites that paste *real* cropped plates onto real backgrounds 60% of the time. Drawn plates
    # alone taught the detector to look for the drawing, not for a plate (README データ戦略).
    echo "== detector: 9000 train + 800 val frames, 60% real plates"
    $JLPR gen-det --out data/det_train2 --count 9000 --seed 8888 --imgsz 640 \
      --bg ../alpr_jp/train/neg --real-plates "$REAL_PLATES" --real-pct 60 $STRICT --quiet
    $JLPR gen-det --out data/det_val2  --count 800  --seed 9911 --imgsz 640 \
      --bg ../alpr_jp/train/neg --real-plates "$REAL_PLATES" --real-pct 60 --quiet
    # Pseudo-labels: the borrowed PlateYOLO-JP detector labels real vehicle photos for us. 1309 photos
    # in, 588 kept at conf>=0.5 — the frames where a teacher we did not train is confident.
    echo "== detector: pseudo-labelled real vehicle photos (teacher = models/plate_det_pyj320.onnx)"
    python tools/pseudo_label.py --src ../alpr_jp/train/sample --out data/det_real --imgsz 640 --conf 0.5
  else
    echo "== detector: fully synthetic fallback (no real plates available)"
    $JLPR gen-det --out data/det_train --count 8000 --seed 4242 --imgsz 640 --quiet
    $JLPR gen-det --out data/det_val   --count 800  --seed 991  --imgsz 640 --quiet
  fi
  echo "== detector eval set (tools/eval_det.py: recall bucketed by plate size)"
  if [ -d ../alpr_jp ]; then
    $JLPR gen-det --out data/det_eval --count 150 --seed 20250819 --imgsz 640       --bg ../alpr_jp/train/neg --real-plates "$REAL_PLATES" --real-pct 60 --quiet
  else
    $JLPR gen-det --out data/det_eval --count 150 --seed 20250819 --imgsz 640 --quiet
  fi
fi

echo
echo "done. sizes:"
du -sh data/* 2>/dev/null | sort -k2

# Optional extra: the same coverage set rendered in the proprietary Windows faces, which no model here
# has ever trained on. Measured 2026-08-19: v4 scores 48.9% on the training faces and 47.5% on these,
# so the typeface gap is 1.4 points — the region head's problem is exposure, not letter shapes. Needs
# `python tools/fetch_fonts.py --include-system` on a Windows box, so it can never be part of the
# published, reproducible numbers.
if [ "$WHAT" = unseen ]; then
  $JLPR gen --out data/region_sweep_unseen --count 276 --region sweep --seed 4242 --quiet
  echo "now: python tools/check_regions.py --data data/region_sweep_unseen --ocr models/plate_ocr_v6_last.onnx"
fi
