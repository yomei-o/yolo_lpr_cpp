# ゼロから学習まで（clone 2 本だけで完結する手順）

このリポジトリと `alpr_jp` を取ってくれば、他に何もダウンロードせずにデータ生成と学習ができる。
書体は再配布可能な 2 つを同梱済み、認識器の出発点 ONNX もリポジトリに入っている。

```sh
git clone https://github.com/yomei-o/yolo_lpr_cpp.git
git clone --depth 1 https://github.com/dyama/alpr_jp.git      # yolo_lpr_cpp と同じ階層に置く
cd yolo_lpr_cpp
```

置き場所は `../alpr_jp`（＝リポジトリの隣）が既定。別の場所なら各コマンドの `--alpr` を変える。

| 要るもの | 無いとどうなる |
|---|---|
| C++20 コンパイラ（g++ / MSVC。`build/gcc.sh` は vcvars 不要） | C++ レーンが使えない。Python レーンだけなら不要 |
| Python 3.10+ ＋ `pip install -r requirements.txt` | Python レーンが使えない。C++ レーンだけなら不要 |
| ネットワーク（**検出器の学習だけ**） | `ultralytics` が `yolov8n.pt` を取れない。認識器・4隅・生成・推論は全部オフラインで完結 |

## 1. ビルド（C++ レーンを使うなら）

```sh
sh build/gcc.sh pure/jlpr.cpp -o jlpr.exe            # mingw g++
sh build/cc.sh  pure/jlpr.cpp -o jlpr.exe            # MSVC（vcvars を自前で叩かない）
EXTRA="-fopenmp" sh build/gcc.sh pure/jlpr.cpp -o jlpr.exe   # 生成が数倍速くなる
```

`./jlpr.exe labels --dump` が 138 地名を吐けば通っている。

## 2. データを作る

```sh
sh tools/make_data.sh                # 全部（合成 3.3 万枚 ＋ 検出用 9800 フレーム ＋ 評価セット）
sh tools/make_data.sh synth          # 認識器のデータだけ
```

- 種は README の数字を出したときと同じもの。**C++ と Python のどちらの生成器でも 1 バイト単位で同じ物が出る**
  （`python tools/parity/gen.py` で検証できる）
- 書体集合はデータ定義の一部（顔の選択が rng の 29 番目の draw）。`spec/fonts.txt` が契約で、
  ディレクトリが違うと生成器が `note: font set differs from spec/fonts.txt` と言う。
  `make_data.sh` は `--fonts-strict`（= `spec/fonts.txt` の 2 書体だけ）で作るので、
  Windows 商用書体を入れてあるマシンでも**公開されている数字と同じデータ**になる
- Windows の商用書体（`python tools/fetch_fonts.py --include-system`）は**未見書体での評価**に使う:
  `sh tools/make_data.sh unseen` → `data/region_sweep_unseen`。学習には使わない（再配布できないので
  他のマシンで再現できず、公開する数字の根拠にできない）。実測した差は 1.4 ポイントしかない
- 目安（8 コア、OpenMP 有効）: 合成 3.3 万枚で 10 分前後、検出用 9800 フレームで 15 分前後

## 3. 学習する

同じことが両言語でできる。速いのは Python（GPU が使える）、依存が無いのは C++。

```sh
# 認識器（11 head 分類）— GPU なら 12 分、CPU なら数時間
python tools/train_ocr.py --synth data/synth --alpr ../alpr_jp --steps 3000 --batch 64 --lr 4e-4 \
  --real-weight 0.4 --workers 4 --eval-every 250 --eval-limit 144 \
  --save ocr.pt --export models/plate_ocr_new.onnx --export-last models/plate_ocr_new_last.onnx

./jlpr.exe train --init models/plate_ocr.onnx --synth data/synth --alpr ../alpr_jp \
  --steps 3000 --batch 64 --lr 4e-4 --export models/plate_ocr_new.onnx

# 4隅回帰
python tools/train_corner.py --synth data/synth --steps 3000 --batch 64 --export models/plate_corner_new.onnx

# 4隅回帰（C++。--init random なら出発点の ONNX も C++ が書くので Python は要らない）
./jlpr.exe train --model corner --synth data/synth --steps 3000 --batch 64 \
  --init random --width 24 --export models/plate_corner_new.onnx

# 出発点のモデルを C++ だけで作る（Python も学習済み重みも要らない）
./jlpr.exe init-det --out models/scratch_320.onnx --arch n --nc 1 --imgsz 320
./jlpr.exe init-det --out models/tr_320.onnx --arch n --nc 1 --imgsz 320 --from-pt yolov8n.pt
./jlpr.exe init-ocr --out models/scratch_ocr.onnx          # 認識器（head 幅は spec から）
./jlpr.exe train --model corner --init random ...          # 4隅（前からある）

# 転移で始めるときは勾配クリップを付ける。事前学習の胴体に真新しい cls head が乗るので、
# 既定 lr のままだと cls が発散する（実測: step3 で 3.6e11 -> NaN。--clip 10 で安定）
./jlpr.exe train --model det --init models/tr_320.onnx --data data/det_train2 --clip 10

# 検出器（Ultralytics。yolov8n.pt を自動取得。T4 で 28 epoch 約 70 分）
python tools/train_det.py --data data/det_train2 --data data/det_real --val data/det_val2 \
  --epochs 28 --imgsz 640 --batch 32 \
  --export models/plate_det_new.onnx --export-imgsz 320 --export-imgsz 640

# 検出器（C++。既存の ONNX をそのまま学習する。mosaic/affine/HSV/反転・SGD+warmup・EMA 付き。
# CPU 320px batch2 で約 3.5 秒/step なので本番量は上の Python/GPU、こちらは追加学習と検算用）
./jlpr.exe train --model det --init models/plate_det_v8n_320.onnx --data data/det \
  --val data/det_val --val-every 200 --epochs 3 --batch 8 \
  --export-best models/plate_det_new_320.onnx --export models/plate_det_new_320_last.onnx
./jlpr.exe train --model det --data data/det --check-aug 4 --batch 4    # 箱が画素と一緒に動くか

# 長時間走らせるとき（3 段すべて共通）
./jlpr.exe train --model det --data data/det --steps 20000 --ckpt run.ckpt --ckpt-every 500   --log run.csv --patience 10 --val data/det_val --val-every 500
./jlpr.exe train --model det --data data/det --steps 20000 --stop-at 4000 --ckpt run.ckpt  # 途中で止める
./jlpr.exe train --model det --data data/det --steps 20000 --resume run.ckpt               # 続きから
```

学習の設計（どのデータでどの head を学習し、どれをマスクするか）は `pure/train_ocr.hpp` の冒頭コメントと
`tools/train_ocr.py` の docstring に書いてある。**地名 head は実データ専任、ただし実データに 1 枚も無い
地名だけ合成で教える**——ここが 2025 追加地名（十勝/日光/江戸川/安曇野/南信州）の生命線。

## 4. 測る（学習したら必ず全部通す）

```sh
python tools/parity/labels.py && python tools/parity/infer.py            # 仕様と推論の C++⇔Python 一致
python tools/parity/gen.py    && python tools/parity/train.py            # 生成と学習 1 step の一致
./jlpr.exe train --model det --gradcheck                                 # 検出損失の勾配（データ不要）
./jlpr.exe train --model det --data data/det --steps 1 --batch 2 --dump-fixture scratch/det_fix.bin
python tools/parity/train_det.py --fixture scratch/det_fix.bin           # 検出損失 vs ultralytics
./jlpr.exe train --model corner --synth data/synth --init models/plate_corner.onnx --steps 1 \
  --batch 8 --dump-fixture scratch/crn_fix.bin
python tools/parity/train_corner.py --fixture scratch/crn_fix.bin --onnx models/plate_corner.onnx
python tools/parity/init_det.py --pt yolov8n.pt --imgsz 320   # C++ が書いたグラフ vs ultralytics の export
./jlpr.exe val --data ../alpr_jp --kind alpr --holdout --ocr models/plate_ocr_new.onnx   # 実データ hold-out
python tools/check_regions.py --data data/region_sweep --ocr models/plate_ocr_new.onnx --alpr ../alpr_jp
./jlpr.exe check-regions --data data/region_sweep --ocr models/plate_ocr_new.onnx --alpr ../alpr_jp  # 同じ数値
./jlpr.exe val --data ../alpr_jp --kind alpr --holdout --tta --ocr models/plate_ocr_new.onnx         # 6クロップTTA
./jlpr.exe context-test --img assets/kei-commercial-yokohama480ri4567.jpg --det models/plate_det_new_320.onnx
python tools/eval_det.py --data data/det_eval --det models/plate_det_new_320.onnx --det-kind v8 --fmt cxcywh
./jlpr.exe val --model det --data data/det_eval --det models/plate_det_new_320.onnx --fmt cxcywh   # 同じ数値
python tools/parity/eval_det.py --data data/det_eval --det models/plate_det_new_320.onnx          # 突き合わせ
./jlpr.exe detect --img assets/kei-commercial-yokohama480ri4567.jpg --det models/plate_det_v8n_320.onnx \
  --det-kind v8 --fmt cxcywh --corner models/plate_corner.onnx        # 実写 1 枚（最後は必ずこれ）
```

合成の指標だけで採否を決めないこと。4隅回帰 v2 は合成誤差が 1.93% → 1.89% と改善したのに、
実写では地名確信度が 0.94 → 0.75 に落ちて不採用になった（README の落とし穴の表）。

## 5. WASM デモ

```sh
sh build/emcc.sh                     # wasm/jlpr.js, wasm/jlpr.wasm を作り直す
node wasm/test_node.js               # 実写フィクスチャで CLI と同じ読みになるか
python -m http.server 8000           # http://localhost:8000/wasm/ を開く
```

モデルは `wasm/index.html` 冒頭の `DETECTORS` / `OCR` / `CORNER` が指すファイルを実行時に取りに行くので、
`models/` を差し替えれば再ビルドは要らない。

## Kaggle の無料 GPU を使う場合

`kaggle_server_cpp`（kbridge）経由で回せる。手順は README の「Kaggle の GPU で学習する」と、
あちらのリポジトリの `FOR_AGENTS.md`。学習ジョブの先頭は必ず
`git checkout -- models/; git clean -fdq models/; git pull` にする（未追跡の onnx で pull が止まる）。
