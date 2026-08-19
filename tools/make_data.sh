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
  $JLPR gen --out data/synth     --count 30000 --seed 90210 --quiet
  $JLPR gen --out data/synth_val --count 2000  --seed 555001 --quiet
  echo "== region coverage test set: 2 crops per region name (tools/check_regions.py)"
  $JLPR gen --out data/region_sweep --count 276 --region sweep --seed 4242 --quiet
fi

if [ "$WHAT" = all ] || [ "$WHAT" = det ]; then
  if [ -d ../alpr_jp ]; then
    # Composites that paste *real* cropped plates onto real backgrounds 60% of the time. Drawn plates
    # alone taught the detector to look for the drawing, not for a plate (README データ戦略).
    echo "== detector: 9000 train + 800 val frames, 60% real plates"
    $JLPR gen-det --out data/det_train2 --count 9000 --seed 8888 --imgsz 640 \
      --bg ../alpr_jp/train/neg --real-plates "$REAL_PLATES" --real-pct 60 --quiet
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
