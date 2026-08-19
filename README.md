# yolo_lpr_cpp — 日本のナンバープレート認識を C++ で（検出 YOLOv8 → 4隅補正 → 位置別クラス分類）

日本の車両ナンバープレートを **検出して読む** システム。モデルは全段 **ONNX**、推論は
**依存ライブラリなしの C++**（MSVC / mingw / Emscripten）で、ブラウザ (WASM) デモまで含む。
学習は Colab / Kaggle の無料枠で回せる範囲に収める（既存重みからのファインチューニング＋合成データ）。

**設計の核: Python と C++ が対等。** データ生成・学習・推論・評価・ONNX 入出力のすべてを
**両言語で実装し、同じ CLI・同じラベル定義・同じ数値**が出ることをテストで縛る。
どちらか一方しかできない機能は作らない（詳細は下の「Python と C++ の対等性」）。

姉妹リポジトリ: 検出エンジン [yolov8_cpp](https://github.com/yomei-o/yolov8_cpp) ／
9-head 分類器 [lpr_cpp](https://github.com/yomei-o/lpr_cpp) ／ [yolox_cpp](https://github.com/yomei-o/yolox_cpp)。
本リポジトリはこの2つを**製品として使える1本のパイプライン**にまとめ直すもの。

## ▶ ブラウザで試す（送信なし・完全ローカル）

**https://yomei-o.github.io/yolo_lpr_cpp/wasm/**

カメラか画像ファイルを渡すと、その場で読みます。**「自動スキャン」を押すとカメラを見続け、プレートが
現れたら自動で検出して読みます**（追跡しながらフレーム方向に確率を足し込むので、見続けるほど安定して
確定します。0.44–1.04 秒/フレーム、推論は Web Worker）。**検出器も認識器も自前で学習したモデル**
（`models/plate_det_v8n_320.onnx` + `models/plate_ocr_v7_bal.onnx`、実データ hold-out の地域名 **97.9%**）。
**認識器も 2 つから選べます**: 「実データ重視」（地名 97.9%／2025 年追加の 5 地名は 28%）と
「新地名対応」（十勝・日光・江戸川・安曇野・南信州を **76%**／地名 95.1%）。
検出器はページ上で 3 つから切り替えられます:

| 検出器 | 得意 | 苦手 | `data/det_eval` recall |
|---|---|---|---|
| **自前 yolov8n 320**（既定） | **プレート単体・近接**（実写の黒ナンバーで det 0.93、車体が写っていなくても取れる） | 画面 10% 前後の遠景を落とす | 93.7% |
| 自前 yolov8n 640 | 遠くの小さいプレートも拾う（画面 11% のバスで 0.41） | 検出だけで 4 倍重い | **97.1%** |
| PlateYOLO-JP 320（借り物・AGPL） | 車ごと写った遠景（同じバスで **0.80**） | **近接は検出ゼロ**（20% で 0.07、30% 超はゼロ） | 13.7%（このセットは車体なしが主なので不利） |

**実写の黒ナンバー（車体なし・机の上）で、切り出し窓を振って測った結果**
（`assets/kei-commercial-yokohama480ri4567.jpg`、`tools/context_test.py`）:

| プレートが画面幅に占める割合 | 5% | 8% | 12% | 20% | 30% | 45% | 60% | 80% | 95% |
|---|---|---|---|---|---|---|---|---|---|
| 借り物 PlateYOLO-JP | 0.76 | 0.83 | 0.70 | 0.07 | なし | なし | なし | なし | なし |
| **自前 yolov8n 320（再学習後）** | 0.32 | 0.82 | 0.57 | **0.88** | **0.91** | **0.91** | **0.92** | **0.94** | **0.95** |

元画像そのまま（プレートが画面幅の 24%）では、**借り物は検出ゼロ・自前は det 0.93 で検出して
「横浜 480 り 4567」と正しく読めた**（地名の確信度 0.94）。3 つは相補的なので切り替えられるようにしてある。

近接が苦手な検出器を選んだときは「**枠を手で指定**」でドラッグしてください（認識器は近接でも読めます）。

**現在の状態: 3 段すべて自前の学習済みモデルで動く。** 残りは 2025 追加地名の仕上げ（十勝）と
plate_kind / legible head の学習、ブラウザ実機での再確認。進捗と残作業は [RESUME.md](RESUME.md)。

## clone だけで学習できる（ゼロからの手順）

このリポジトリと [dyama/alpr_jp](https://github.com/dyama/alpr_jp) の 2 本を clone すれば、
他に何も落とさずにデータ生成・学習・評価・WASM デモまで通る。書体（再配布可能な 2 つ）と
認識器の出発点 ONNX は同梱済み。手順は **[SETUP.md](SETUP.md)** に全部書いた。

```sh
git clone https://github.com/yomei-o/yolo_lpr_cpp.git
git clone --depth 1 https://github.com/dyama/alpr_jp.git    # 隣に置く
cd yolo_lpr_cpp && sh build/gcc.sh pure/jlpr.cpp -o jlpr.exe && sh tools/make_data.sh
```

## 動かす

```sh
sh build/gcc.sh pure/jlpr.cpp -o jlpr.exe          # または sh build/cc.sh (MSVC, vcvars 不要)

./jlpr.exe detect --img assets/tokyu-bus-yokohama200ka3591.jpg \
                  --det models/plate_det_yolox.onnx --ocr models/plate_ocr.onnx --out out.png
#   plates: 2 (conf>=0.15, 6 raw detections over all classes)
#     [0] box (250,407)-(337,465) det 0.82  crops 6
#          横浜 200 か 3591
#          conf 0.57 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00

python tools/jlpr.py detect --img assets/tokyu-bus-yokohama200ka3591.jpg   # Python 版（同じフラグ）
python tools/parity/labels.py                       # ラベル表の C++⇔Python 一致 (M1)
python tools/parity/infer.py --ref <lpr_cpp>/pure/ref   # 推論の C++⇔Python 一致 (M3)
./jlpr.exe parity-ocr --ocr models/plate_ocr.onnx --ref <lpr_cpp>/pure/ref
#   exported ONNX vs original-ONNX fixture: worst 3.299e-05   argmax 9/9   MATCH
```

ブラウザ（WASM、送信なし・完全ローカル）:

```sh
sh build/emcc.sh wasm/jlpr_wasm.cpp -o wasm/jlpr.js
python -m http.server 8000     # リポジトリのルートから → http://localhost:8000/wasm/
node wasm/test_node.js         # ヘッドレスの回帰テスト（CLI と同じ画素で同じ文字列を assert）
```

実測（実写 960×640、CPU 1 スレッド）: CLI **2.2 秒** / WASM(node, SIMD) **1.9 秒** で
`横浜 200 か 3591`（地域名の信頼度 0.96）。内訳は検出 320px と 6 クロップ分の認識。

**まだ暫定なところ**: 検出器は [PlateYOLO-JP](https://github.com/Kazuhito00/PlateYOLO-JP-Prototype)
（YOLO12・AGPL-3.0）の NMS を剥がしたものを借りている（`tools/strip_nms.py`）。自前 yolov8n nc=1 に
差し替えるのは M7 で、**それが超えるべき基準がこの暫定検出器**。4隅補正（M6）も未実装。

参考までに、検出器を替えただけで認識側が受ける影響（同じ写真・同じ認識器）:

| 検出器 | 誤検出 | 地域名 conf | CLI 時間 |
|---|---|---|---|
| YOLOX-tiny 416（`lpr_cpp` 由来・最初の暫定） | 1 件 (det 0.24) | **0.57** | 5.6 秒 |
| PlateYOLO-JP 320（NMS 剥がし・現行） | 0 件 | **0.92** | 2.4 秒 |

box が良くなるだけで地域名の信頼度が 0.57 → 0.92 に動く。**入力の正規化が精度を支配している**という
仮説の裏付けで、4隅補正（M6）を入れる理由がそのまま数字で出ている。

## いま出せている数字（2026-08-19）

| | 数字 | どう測ったか |
|---|---|---|
| 認識器（地域名） | **97.9%** | 実データ 144 枚 hold-out、margin 0.03 固定、TTA なし（`jlpr val --holdout`）。誤りは 3 枚で、いずれも学習 12-25 枚の地名 |
| 認識器（2025 追加の 5 地名） | 1% → **28%**（精度を落とさず）／**76%**（精度 95.1% を許すなら） | 各名 20 枚・計 100 枚（`--region 133-137`）。朝は確率 1e-09 で構造的に出せなかった |
| 認識器（合成データの数字系 head） | 87-90% | 合成 200 枚 |
| 認識器（地名を138名すべて言えるか） | 15.9% → **50.7%** | `tools/check_regions.py`（地名1つずつレンダして読ませる）。v2 → v5。2025 追加の 5 名も 4/5 が読めるようになった（安曇野 2/2、江戸川・日光・南信州 1/2、十勝 0/2） |
| 書体のドメインギャップ | **1.4 ポイント**だけ | 学習と同じ 2 書体で 48.9%、未見の Windows 商用書体 3 つで 47.5%（v4、138名レンダ） |
| 4隅補正 | 近接で地域名 0.42 → **0.85** | 実写の黒ナンバー。90px 未満の box では使わない（実測で悪化するため） |
| 検出器 学習時 val | **mAP50 0.985 / mAP50-95 0.947**（P 0.999 R 0.987） | 合成 val 800 フレーム 964 個（28 epoch、Kaggle T4 71 分） |
| 検出器 recall（自前 320） | **93.7%** | `data/det_eval` 150 フレーム 175 個、conf 0.25 / IoU 0.5 |
| 検出器 recall（自前 640） | **97.1%** | 同じセット。小さいプレート（画面 5-10%）で 86.8% → 94.7%。480 は 96.6% と 640 に肉薄するが実写の遠景を落とすので採らない |
| 検出器 recall（借り物 PlateYOLO-JP） | 13.7% | 同じセット。**このセットは車体が写っていない絵が主なので借り物に不利**（借り物が得意なのは車ごと写す構図） |
| 誤検出 | プレートが無い 12 フレームで 0 個 | 同上（借り物は 1 個） |
| トラッキング | 0.56-0.61 秒/フレーム、3 フレームで確定 | 実写 640×480、node/SIMD |
| C++ ⇔ Python | 推論は文字列・argmax 一致、学習は step1 の loss 差 1e-6 | `tools/parity/*.py` |

## 実写での検出結果

手元の実写 1 枚をそのまま入れた結果（`tools/infer.py --out` が描いた画像）。
枠が検出位置、上のラベルが読んだ文字列と検出スコア。

### 軽自動車の事業用（黒ナンバー）を近接で — 車体はフレームに無い

![黒ナンバーの認識結果](docs/result_kei.jpg)

`横浜 480 り 4567` を正解、検出 0.93 / 地名の確信度 0.83（自前 yolov8n 320）。
**プレート単体でも検出できる**のがこのモデルを自分で学習した理由で、
借り物の検出器（PlateYOLO-JP）はこの写真では 1 個も検出できない（0 detections）。
再学習前は同じ写真で 0.80 ＋ モニタの縁に誤検出 0.50 が出ていた。今はその誤検出も消えている。

### バスのプレートを遠景で — 画面幅の 11%

![バスの認識結果](docs/result_bus.jpg)

`横浜 200 か 3591` を正解、検出 0.41（自前 yolov8n **640**）。
この写真は自前 320 だと落とす（0 detections）ので、遠景は 640 か借り物（0.80）を使う。
3 つのモデルは補完関係にあり、デモではプルダウンで切り替えられる。

## パイプライン（3段）

```
入力画像
  │
  ├─[1] 検出   yolov8n  nc=1 (plate)   640×640   → box
  │        （小さいプレート用に COCO yolov8n で車を切ってから再検出する2パスを任意で）
  ├─[2] 矯正   corner-net 64×64 → 4隅 8座標      → 透視変換で 128×128 に正規化
  └─[3] 認識   dwsep-ResNet 128ch 128×128        → 11 head の argmax
                                                    → 「品川 371 ら 100」＋ 種別/信頼度
```

**なぜ 2段目を足すのか。** `lpr_cpp` の実測で、地域名 head（133クラス）は**クロップ余白 数%で
奄美 / 横浜 / 練馬 が入れ替わる**。同リポは 6クロップの TTA（確率和）で緩和しているが、これは
入力正規化がブレていることの症状であって原因治療ではない。4隅を推定して射影補正し**固定マージンで
128×128 を作る**のが正攻法で、合成データなら4隅の正解がタダで得られる。ここが「少データで高性能」の要。

## 確定した方針（2026-08-19）

| 論点 | 決定 | 理由 |
|---|---|---|
| 学習の実装 | **Python と C++ の両方で対等に**（成果物・数値ともに一致させる） | ここが本リポの主眼。無料枠での長時間学習は速い方（Python/CUDA）で回してよいが、C++ 単独でも同じ学習・同じ ONNX 出力ができる状態を維持する |
| モデル配布形式 | **ONNX（全3段）** | C++ / WASM 双方が同じファイルを直読み。姉妹リポの ONNX リーダを流用 |
| 対応範囲 | 中型(330×165)・**大型(440×220)**・軽、白/黄/緑/黒、**地方版図柄入り** | 種別は3段目の head として明示的に出す。二輪・2段組みは初版対象外 |
| 矯正段 | **入れる**（4隅回帰 → 透視変換） | 上記のとおり地域名 head の余白依存を根本から潰す |
| 事前学習重み | **AGPL 許容**（Ultralytics `yolov8n.pt` から転移） | 少データ前提で検出精度が最優先。姉妹リポと同じ扱い |
| 推論の依存 | なし（自前 ONNX ランナ＋stb） | MSVC / mingw / emcc の3系統で同一ソース。`-DUSE_EIGEN` で CPU 高速化 |

## Python と C++ の対等性

両言語に同じ機能を持たせ、**同じサブコマンド名**で叩けるようにする。片方だけの近道は作らない。

| 機能 | Python (`tools/`) | C++ (`pure/`) | パリティの条件 |
|---|---|---|---|
| ラベル表（地名138/かな/分類番号/種別） | `labels.py` ✅ | `spec.hpp` ✅ | 唯一の定義 `spec/labels.txt` を**両方が実行時にパース**（生成物ではなく同じ入力）。dump と埋め込みヘッダがバイト一致 |
| 合成データ生成 | `gen.py` | `gen.cpp` | 同じ seed で**文字列・4隅・変換パラメータ・色・種別が完全一致**。画像はラスタライザ差（PIL/FreeType vs stb_truetype）が出るので、一致は幾何とラベルまでとし、画像は統計と目視で同等を確認 |
| 学習（認識器） | `train_ocr.py`（PyTorch 移植、ONNX 重み読み込み） ✅ | `jlpr train`（**ONNX グラフを直接学習**） ✅ | 同じ ONNX・同じ seed・同じ batch で **step1 の loss 差 1e-6**、以降は相対 1% 以内（実測、`tools/parity/train.py`） |
| 学習（検出器） | `train_det.py`（Ultralytics） ✅ | `jlpr train --model det`（**ONNX グラフを直接学習**、mosaic/affine/HSV/反転・SGD+warmup・EMA・学習中 val 付き） ✅ | 同じ head テンソルを両者に食わせて loss 3 項が相対 **1e-05 以内**、勾配が **3e-06 以内**（実測、`tools/parity/train_det.py`）。拡張は箱が画素と一緒に動くことを `--check-aug` で検算 |
| 学習（4隅） | `train_corner.py` ✅ | `jlpr train --model corner`（**ONNX を直接学習**、BN は学習モード） ✅ | 同じ ONNX・同じバッチで **loss 2.7e-06 / 勾配 1.6e-05**（`tools/parity/train_corner.py`）。`--init random` なら出発点の ONNX も C++ が書く |
| head の拡張（地名 133→138） | `ocr_model.py` の重み読み込み時 ✅ | `onx::widen_heads` ✅ | 同じ ONNX から始めて**学習前の精度が一致**（追加クラス bias -10 で 92.4%、0 で 85.4%） |
| チェックポイント選択 | `--export` / `--export-balanced` / `--export-last` ✅ | 同じ 3 フラグ ✅ | 実データ最良／実データを 1pt 以内に抑えて合成地名最大／最終step。選択規則が同一 |
| 生成の地名指定 | `--region <n>\|sweep\|a-b` ✅ | 同じ ✅ | labels/meta がバイト一致（`--region sweep` で確認） |
| 書体集合の固定 | `--fonts-strict` ✅ | 同じ ✅ | `spec/fonts.txt` の 2 書体だけで生成し、どのマシンでも同じ画になる |
| 推論 | `infer.py`（onnxruntime） ✅ | 自前 ONNX ランナ ✅ | 同一画像で**同一文字列・同一 argmax**、box 差 0.10px / det 差 0.0000（実測）。自作側の float 加算誤差は認識器 3.3e-05・検出器 3e-03 なので、閾値ぎりぎりの box は割れる |
| 評価（mAP / 全文一致 / 色別内訳） | `eval_det.py` / `eval_ocr.py` ✅ | `jlpr val --model det` / `jlpr val` ✅ | 同じ dir・同じ ONNX で mAP50・mAP50-95・P/R/F1・バケット別 recall が一致（実測差 **5e-07 以下**、`tools/parity/eval_det.py`）。`compute_ap` は ultralytics と**差 0** |
| ONNX 出力 | `torch.onnx.export` | 自前エクスポータ | 出力 ONNX を相互に読み込んで **forward が 1e-5 以内** |
| 疑似ラベル | `pseudo_label.py` ✅ | `jlpr pseudo-label` ✅ | 同じ検出器で同じ YOLO 形式を書く（書いたラベルを `jlpr val --model det` で読み返して検算） |
| 地名の到達性 | `check_regions.py` ✅ | `jlpr check-regions` ✅ | 実測**完全一致**（v7_bal 42.0%、2025 追加 1/5、一度も読めない名前 80 個） |
| 診断（窓サイズ掃引 / 色替え / NMS 剥がし） | `context_test.py` / `recolor_test.py` / `strip_nms.py` ✅ | `jlpr context-test` / `recolor-test` / `strip-nms` ✅ | context-test は det も IoU も plate px も**完全一致**。strip-nms の出力は onnxruntime が読める |
| WASM | — | `emcc`（C++ 側の役目） | ブラウザで CLI と同じ文字列が出る |

**Python は必須ではなく、速いだけ。** 時間に余裕があるなら **C++ だけで全部完結できる**
（合成データ生成 → 学習 → ONNX 出力 → 推論 → WASM デモ）。逆に急ぐなら本番学習だけ Python/CUDA に投げる。
どちらの経路でも同じ ONNX が出て、同じ数値が出る。

**速度は対等ではない**（そこは割り切る）。姉妹リポの実測で C++/CUDA 経路は COCO128/640/batch4 で
約 4.6 分/epoch = 100 epoch なら約 8 時間、CPU なら 640px で丸一日規模。無料枠で本番学習を回すときは
Python/CUDA が現実的で、C++ 側は**同じ結果に到達できることを小規模で証明し、追加学習にも使える**状態を保つ。

## モデル仕様

### [1] 検出器 — yolov8n / nc=1
- 入力 640×640 letterbox（WASM 用に 416 / 512 も出す）、出力は素の YOLOv8 head（DFL 16bin）
- ONNX は **NMS 前**まで。decode + NMS は C++ 側（`yolov8_cpp/pure/infer.hpp` 由来、nc 任意に一般化）
- 学習は `yolov8n.pt` から転移。mosaic / mixup / HSV / affine ＋ プレート色の均衡と夜間・逆光・ブラー
- 評価は mAP@0.5 と**プレート色別 recall を別掲**（下の「既知の落とし穴」参照）

### [2] 4隅回帰器 — corner-net（新規・小型）
- 入力: 検出 box を 1.25 倍に広げた crop を 64×64
- 出力: 4隅の (x,y) × 4 = 8 値（crop 正規化座標）。損失は Wing / smooth-L1
- 学習: 合成データが主（4隅は生成時に既知）。実データは正面向きを弱ラベルとして併用
- 想定サイズ 0.2–0.5 MB。ここで得た4隅で 128×128 に warp（マージンは実験で決めて記録する）

### 学習の実測（M5、2026-08-19）

出発点は出荷済みの重み（`models/plate_ocr.onnx`）。実データ 720 枚を 576 train / 144 hold-out に
分けて（両言語で同じ分割）、合成データと混ぜて 600 step 学習した結果:

| | 実データ hold-out region top1 |
|---|---|
| 出荷済みの重み | 91.7% |
| **600 step 学習後（`plate_ocr_v2.onnx`）** | **95.1%** |

学習前に見つけて直したこと（数字が出たので分かった）:
- **region を 133→138 に広げるとき、新クラスの bias が 0 だと既存クラスに勝ってしまう**
  （logit 0 は負の logit に勝つ）。学習前から 91.7% → 85.4% に落ちていた。bias を −10 で初期化して解決。
- **合成データで region head を学習させると実データ精度が落ちる**（78% → 67%）。合成サンプルは region をマスク。
- **BN 統計を凍結**し、backbone は lr×0.1。合成が多いバッチで BN 統計が動くと実データ精度が下がる。

### [3] 認識器 — depthwise-separable ResNet 128ch（`lpr_cpp` の構造を継承）
既存 ONNX（実データ学習済み）の重みを初期値にする。構造は `lpr_cpp/pure/ref/ARCH.md` の通り:
stem `Conv4×4/s4` → **Conv→ReLU→BN**（BN eps 1e-3）で分岐A(6ブロック)・分岐B(5ブロック) →
GAP → Dense+softmax。head を 9 → **11** に拡張する:

| head | クラス数 | 備考 |
|---|---|---|
| region（地域名） | **138+** | 既存 133 に **十勝・日光・江戸川・安曇野・南信州**（2025-05 追加）を足す。現行リストを確定して固定する |
| class_num_01 / 02 / 03（分類番号） | 10 / 20 / 22 | 既存踏襲（2桁目以降にアルファベット、3桁目に空白あり） |
| hiragana | 53 | 既存踏襲（`あ〜ろ` ＋ `ABCEHKLMTYV`）。お・し・へ・ん は実在しないので生成時に除外 |
| plate_num_01..04（一連番号） | 11 / 11 / 11 / 10 | 値 10 = 空白（`・` 表示側） |
| **plate_kind**（新規） | 8 | 自家用普通 / 事業用普通 / 軽自家用 / 軽事業用 / 大型 / 図柄入り / その他 / 不明 |
| **legible**（新規） | 2 | 読めるか否か。運用時の棄却閾値に使う |

head 追加は分岐の GAP 出力に Dense を足すだけなので、既存 backbone の初期値をそのまま活かせる。

## データ戦略（実データは少なく、生成で埋める）

### リポジトリに入っていないもの（と、その理由）
| | git に入れているか | 理由 / 代わりの手段 |
|---|---|---|
| `data/`（合成 3 万枚ほか） | **入れない** | 種の関数でしかないので、置いても `tools/make_data.sh` の遅い複製になる。`sh tools/make_data.sh` で全部作り直せる（種は実際に出荷モデルを学習したもの。C++/Python どちらの生成器でも 1 バイト単位で同じ物が出る） |
| `data/`（実データ） | **入れない** | 他人が撮った写真で、再配布する立場にない。`git clone https://github.com/dyama/alpr_jp.git ../alpr_jp` を各自で |
| `fonts/GenSenRounded2-B.ttc` | **入れた**（17MB） | SIL OFL 1.1 で再配布可。clone しただけで生成が走り、パリティの基準書体が確実に揃う |
| `fonts/meiryo.ttc` 等 | 入れない | Windows 付属の商用フォント。`python tools/fetch_fonts.py --include-system` で各自の PC からコピー（生成器は使った書体名を meta.txt に記録する） |
| `models/*.onnx` | **入れた** | 合計 40MB。これが無いとデモが動かないので、リポジトリの一部として扱う |
| `fonts/DroidSansFallbackFull.ttf` | **入れた**（4MB） | Apache-2.0（(c) Google, Android 由来）。GenSen と合わせた 2 書体が出荷モデルの学習に使った集合そのもの（`spec/fonts.txt`） |


### 集める（実データ）
| 出所 | 内容 | ライセンス | 用途 |
|---|---|---|---|
| [dyama/alpr_jp](https://github.com/dyama/alpr_jp) | プレート crop 1,249枚（うち **720枚が `自家用/<地名>/` 構造＝地域名ラベル付き**, 66地名）＋ ネガ 4,420枚 | MIT（撮影者に著作権） | 認識器の実データ微調整・地域名 head の検証・検出のハードネガ |
| [Number Plate in Japan (Roboflow, 2,196枚)](https://universe.roboflow.com/moriken/number-plate-in-japan), [license-plate-japan](https://universe.roboflow.com/new-workspace-vijtn/license-plate-japan) | 日本車の全景＋プレート box | **要確認**（RESUME の未確認事項） | 検出器の学習・評価 |
| Open Images V7 `Vehicle registration plate` | 全世界のプレート box（大量） | 画像 CC BY 2.0 / 注釈 CC BY 4.0 | 検出器の量稼ぎ（日本以外も形状は近い） |
| 自前撮影 | 黒・黄プレート、近接、夜間 | 自前 | 弱点の埋め合わせと最終評価 |
| [PlateYOLO-JP](https://github.com/Kazuhito00/PlateYOLO-JP-Prototype) の検出器 | ラベルではなく**教師**（車写真に自動で box を付ける） | AGPL-3.0 | 検出データの自動ラベル付け。Python 側だけで動かす（NMS 入りグラフのため） |
| 同リポの EkMixer 認識器 | 疑似ラベルの**第二意見** | AGPL-3.0 | 自前モデルと一致した読みだけ採用し、不一致だけ人が見る |

### 生成する（合成）
生成器は Python と C++ の両方に置く（`tools/gen.py` / `pure/gen.cpp`）。プレート**画像そのもの**と**車体に貼った全景**の両方を出す。
- テンプレート: 中型 330×165 / 大型 440×220、白地緑字・黄地黒字・緑地白字・黒地黄字、図柄入りは背景に模倣柄
- 書体: [FGゼロラバウル](https://coliss.com/articles/freebies/font-fgzerorabaul.html)（プレート書体を再現した商用可フリーフォント、漢字かな入り）を軸に複数フォントを混ぜ、字幅・太さ・字間をジッタ（1書体に寄せると本物で滑る）
- 実在ルール: 地名138、分類番号の桁数とアルファベット規則、ひらがなの使用可集合、一連番号の `・` 詰め
- 物理: ボルト・封印、汚れ・錆・水滴、フレーム、影・反射・ヘッドライト、パース(yaw±35° / pitch±20°)、モーションブラー、JPEG劣化、幅 24–200px の低解像度
- 規模の目安: 認識 20–40万枚、検出コンポジット 2–5万枚（実データはその 5% 程度）
- 合成では **4隅・全head ラベルが正解付きで得られる**ので、2段目と3段目はここで作り込む

### Kaggle の GPU で学習する（kbridge 経由）

姉妹リポジトリ [kaggle_server_cpp](https://github.com/yomei-o/kaggle_server_cpp)（kbridge）で、Kaggle の
無料 GPU をローカルの HTTP API として使える。Colab notebook を開かなくても、エージェントや `curl` から
ビルド・生成・学習・回収まで通せる。長時間の学習は `/exec` ではなく `/job`（Kaggle 側で切り離して起動し、
ログをファイルに落とす）を使う。

```sh
# ローカル: kbridge を起動して Kaggle セッションに繋ぐ（URL は Notebook の "VSCode Compatible URL"）
./kbridge_server.exe --port 8787
curl -s -X POST localhost:8787/session -d '{"url":"https://kkb-production.jupyter-proxy.kaggle.net/k/.../proxy"}'
curl -s localhost:8787/gpu            # -> Tesla T4 x2, torch 2.10+cu128

# Kaggle 側: 環境を作る（このリポを clone してビルド、フォントと実データを取る）
curl -s -X POST localhost:8787/job -d '{"name":"prep","cmd":"cd /kaggle/working &&   git clone -q --depth 1 https://github.com/yomei-o/yolo_lpr_cpp.git && cd yolo_lpr_cpp &&   python tools/fetch_fonts.py --include-system &&   g++ -std=c++20 -O2 -fopenmp -Ipure -Ipure/third_party pure/jlpr.cpp -o jlpr &&   git clone -q --depth 1 https://github.com/dyama/alpr_jp.git ../alpr_jp &&   ./jlpr gen --out data/synth --count 30000 --seed 90210 --quiet"}'

# 学習（GPU）。ログは何度でも増分で読める
curl -s -X POST localhost:8787/job -d '{"name":"ocr","cmd":"cd /kaggle/working/yolo_lpr_cpp &&   python tools/train_ocr.py --synth data/synth --alpr ../alpr_jp --steps 4000 --batch 64   --workers 4 --export models/plate_ocr_v2.onnx"}'
curl -s "localhost:8787/job/<id>/log?offset=0"

# 成果物を回収
curl -s "localhost:8787/download?path=yolo_lpr_cpp/models/plate_ocr_v2.onnx" -o models/plate_ocr_v2.onnx
```

Kaggle 側の実測（無料枠 T4×2 / 4 vCPU）: 合成生成 **0.06 秒/枚**（4 コア）、認識器の学習は GPU なら
ローカル CPU の 50 倍以上速い。`--workers 4` を付けないと GPU が画像デコード待ちになる。

### 学習の段取り
1. 合成のみで 3 モデルを事前学習（Colab T4 で検出 1.5–3h / 認識 1–2h / 4隅 30分 が目安）
2. 認識器は既存 ONNX 重みからの転移 ＋ 実データ（alpr_jp 720枚 + 自前）で低 LR 微調整（head→backbone の段階解凍）
3. 既存モデルで実データに疑似ラベル → TTA 一致分だけ採用 → 不一致だけ人手修正（能動学習）
4. 評価は **実データ hold-out のみ**で、色別・地名別・解像度別に出す

目標値: 検出 mAP@0.5 ≥ 0.95（色別 recall も ≥ 0.90）、認識は**プレート全文一致 ≥ 95%**、region ≥ 98%（TTA なし）。

### 検出器も C++ で学習できる（M7b、2026-08-19 夜）

`jlpr train --model det` が入り、**検出器の学習も Python と対等**になった。認識器と同じ方式
＝ **ONNX グラフをそのまま学習する**。RESUME には「v8 の損失は生の head 出力を必要とし、それは
グラフの外側にあるので 1,100 行の移植が要る」と書いてあったが、これは**誤りだった**: Ultralytics の
ONNX には per-level の head conv がそのまま残っている（`/model.22/cv2.<l>/cv2.<l>.2/Conv_output_0`
＝ box 分布 64ch、`/model.22/cv3.<l>/cv3.<l>.2/Conv_output_0` ＝ クラス logit）。この 6 本で `stop`
すれば、その先（Reshape/Concat/DFL/anchor 復号）は**推論用の尻尾**でしかない。移植したのは損失側だけ
（`pure/train_det.hpp` 455 行 ＝ TAL 割り当て・CIoU・DFL・BCE）。head の名前ではなく
**グラフの形**（DFL の Softmax と cls の Sigmoid から逆にたどる）で見つけるので、module 番号が違う
エクスポートでも動く。

```sh
./jlpr.exe gen-det --out data/det_smoke --count 8 --imgsz 320 --seed 5     # 数枚だけ作る
./jlpr.exe train --model det --data data/det_smoke --limit 2 --batch 2 --steps 40 --lr 3e-4 \
                 --export scratch/plate_det_two.onnx
./jlpr.exe train --model det --gradcheck          # 解析勾配 vs 中心差分（データもモデルも要らない）
./jlpr.exe train --model det --data data/det_smoke --steps 1 --batch 2 --dump-fixture scratch/det_fix.bin
python tools/parity/train_det.py --fixture scratch/det_fix.bin   # vs ultralytics の v8DetectionLoss
```

実測（`models/plate_det_v8n_320.onnx` から、CPU・320px・batch 2 で約 3.5 秒/step）:

| 確認 | 結果 |
|---|---|
| 2 枚に固定して 40 step | total 2.256 → **1.520**、box 0.247 → **0.024**、cls 0.144 → **0.038** |
| gradcheck（142 箇所） | 最悪の相対誤差 **2.7e-03** ＝ float32 中心差分の分解能どおり。割り当てが切り替わる 8 箇所は比較対象外 |
| Ultralytics との loss 一致 | box **2.5e-07** / cls 9.7e-06 / dfl 2.3e-07 / 合計 3.1e-06（相対、ultralytics 8.4.104） |
| Ultralytics との勾配一致 | box head **2.7e-06** / cls head **3.1e-06**（max\|差\| / max\|勾配\|） |
| 学習後の ONNX | `jlpr detect` がそのまま読んで実写を検出（det 0.93 → 0.98。ただし合成 8 枚に 80 step 当てた副作用で誤検出が 1 件出る） |

dfl 項が 0.70 前後で下げ止まるのは正常。DFL の下限は目標値の**エントロピー**（隣り合う 2 bin に理想的な
重みを置いたときの値で、gain 1.5 込みの最大が 1.5·ln2 = 1.04）なので、**box と cls が落ちていれば効いている**。

まだ入っていないもの: mosaic などの拡張、EMA・SGD スケジュール、C++ 側の mAP 評価。
本番の学習量（合成 9000 枚 × 28 epoch）は CPU では現実的でないので、そこは引き続き Python/GPU
（`tools/train_det.py`）で回し、C++ 側は**同じ損失に到達できることの証明と追加学習**に使う。

### 評価も 4隅の学習も C++ で（2026-08-19 夜、対等性の残り 2 つ）

**`jlpr val --model det`** — 検出器の評価。バケット別 recall（`tools/eval_det.py` と同じ突き合わせ）に
加えて **mAP50 / mAP50-95** を出す。積分の作法は Ultralytics の `compute_ap` そのまま（101 点補間、
番兵の置き方まで）で、突き合わせは 2 段構え: C++ ⇔ Python が同じ数値であること、そして Python の
`compute_ap` が **ultralytics のものと差 0** であること（`tools/parity/eval_det.py`）。
なお同じ指標でも `ultralytics val` の数字とは一致しない。**入力が違う**（本リポは素の resize、
Ultralytics は letterbox）。P/R/F1 は max-F1 を探さず**指定した conf での値**を出す（CLI と WASM が
実際に動く閾値だから）。

**`jlpr train --model corner`** — 4隅回帰器の学習。これで**学習は 3 段とも両言語**になった。
`tools/train_corner.py` と同じ手順・同じ**乱数の引き順**（画像 index → 枠ジッタ 4 つ → 拡大率）なので、
同じ seed なら同じバッチを見る。認識器・検出器と同じく **ONNX をそのまま学習**するが、2 点違う:
BatchNorm を**学習モード**で回す（run_onnx の `bn_training`、running 統計はバッファとして in-place 更新
→ `write_back` で ONNX に戻る）ことと、`--init random` で**出発点の ONNX を C++ 側が書く**こと
（PyTorch の既定初期化と同じ ±1/√fan_in）。

```sh
./jlpr.exe val --model det --data data/det_val --det models/plate_det_v8n_320.onnx --fmt cxcywh
python tools/parity/eval_det.py --data data/det_val --det models/plate_det_v8n_320.onnx

./jlpr.exe train --model corner --synth data/synth --steps 3000 --batch 32 \
                 --init random --export models/plate_corner_new.onnx
./jlpr.exe train --model corner --synth data/synth --init models/plate_corner.onnx --steps 1 \
                 --batch 8 --dump-fixture scratch/crn_fix.bin
python tools/parity/train_corner.py --fixture scratch/crn_fix.bin --onnx models/plate_corner.onnx
./jlpr.exe train --model corner --synth data/synth_val --init models/plate_corner.onnx --steps 0  # 評価だけ
```

| 確認 | 結果 |
|---|---|
| 検出器の評価 C++ ⇔ Python | 合成 60 フレームで mAP50 **0.9850** / mAP50-95 **0.8716** が両言語一致（差 5e-07 以下）、バケット表も TP/FP も完全一致 |
| `compute_ap` ⇔ ultralytics | ランダムな P/R 曲線 20 本で **差 0.00e+00** |
| 4隅学習 C++ ⇔ PyTorch | 同じ重み・同じバッチで loss **2.7e-06**、勾配の最悪 **1.6e-05**（相対） |
| C++ が書いた ONNX | onnxruntime と PyTorch CornerNet が **5.96e-08** で一致（`--check-graph`） |
| 既存 `plate_corner.onnx` の評価 | C++ で **1.93%**（Python が測った 1.93% と同じ）。評価系が同じものを測っている証拠 |

### 検出器の学習に「中身」を入れた／踏んだ罠（2026-08-19 夜）

損失が正しいだけでは学習は回らないので、`tools/train_det.py` が Ultralytics に頼んでいるものを
C++ 側にも入れた: **mosaic（4枚を 2S キャンバス）＋回転/拡大縮小/平行移動＋HSV＋左右反転＋close_mosaic**、
**SGD(momentum 0.937, nesterov, wd 5e-4) ＋ warmup（lr と momentum の両方）＋ linear/cosine 減衰**、
**EMA(0.9999, ramp)**、`--freeze N`、`--epochs`、そして**学習中の検証**（`--val/--val-every` で
製品経路＝フルグラフ + NMS を mAP で測り、`--export-best` で mAP50-95 最良の EMA 重みを書き出す）。

拡張は「箱が画素と一緒に動いているか」が命なので、`--check-aug` で**学習済み検出器に読ませて検算**する:

```sh
./jlpr.exe train --model det --data data/det_val --check-aug 4 --batch 4
#   check-aug [plain resize]: 15 labels, the shipped detector finds 100.0% (mean best IoU 0.933)
#   check-aug [augmented]   : 19 labels, the shipped detector finds 100.0% (mean best IoU 0.954)
```

**踏んだ罠（重要）: 重みでない初期化子を最適化していた。** `make_trainable` が ONNX の初期化子を
全部パラメータ扱いしていたので、neck の Resize scales（float `[1,1,2,2]`）まで対象に入っていた。
勾配は来ないので SGD/Adam は動かさないが、**decoupled weight decay は勾配を見ずに縮める**:
2.0 → 1.9999996 になり、`onnx_run` の `(int64_t)scale` が 2 ではなく **1** に落ちて neck が崩壊する。
実測 **1 step で loss 2.90 → 23.13、mAP50 0.995 → 0.005**（重みは全部 7 桁一致したまま）。
対策は 2 つ: `weight_initializers()` で Conv/Gemm/MatMul/BN の**重み位置の入力だけ**をパラメータにし、
さらに損失が到達する部分グラフに限定する（検出器なら head 6 本＝DFL 射影と decode 尾部は対象外）。
`Resize` のスケールも `llround` で読むようにした。

### 認識器はどれを使うか — 実データで測って決めた（2026-08-19）
`alpr_jp` の地域名ラベル付き **720 枚**で 2 つの既存モデルを比較（`python tools/eval_ocr.py --data <alpr_jp>`）:

| 認識器 | region top1（1クロップ） | region top1（6クロップTTA） | top3(TTA) |
|---|---|---|---|
| 自前 `plate_ocr`（`lpr_cpp` 由来・128ch dwsep-ResNet, 1.3MB） | 89.2% | 92.5% | 96.1% |
| **`plate_ocr_v2`（M5 で学習したもの・現行の既定）** | **95.6%** | — | 98.6% |
| EkMixer（PlateYOLO-JP 同梱・1.0MB） | 76.1% | 81.2% | 88.2% |

→ **自前を base に微調整する**方針で確定（M5）。EkMixer は捨てずに疑似ラベルの第二意見として使う
（両者の一致率 81.4%、両方正解 80.1% ＝ 不一致の 2 割弱に人手を集中させられる）。

## 実装（同じ機能を2系統）

```
spec/      labels.txt                地名/かな/分類番号/種別の唯一の定義（C++・Python 両方をここから生成）
           pipeline.md               入出力・前処理・マージン・閾値の仕様（両実装が従う）
pure/      onnx.hpp / onnx_run.hpp   ONNX リーダ（ファイル/メモリ/入力次元）＋統合インタプリタ  ✅
           nd.hpp                    N次元 op（transpose/softmax/matmul/broadcast、推論専用）  ✅
           infer_v8.hpp              [1,4+nc,N] head の decode+NMS（v8/v11/v12 共通）  ✅
           onnx_export_lpr.hpp       認識器の重み(manifest+weights.bin) → ONNX  ✅
           autograd/…                自作 autograd・conv・BN・Adam（姉妹リポから移植）
           infer_yolox.hpp           暫定検出器の decode+NMS  ✅（yolov8 用は M7 で追加）
           crop.hpp                  letterbox / box→128×128 crop  ✅（4隅 warp は M6）
           spec.hpp                  spec/labels.txt を実行時パース＋decode  ✅
           pipeline.hpp              detect → crop → classify → 文字列＋信頼度  ✅
           gen.cpp                   合成データ生成（C++ 版、M4）
           jlpr.cpp                  CLI: labels / export / parity-ocr / detect / rgba  ✅
tools/     labels.py rng.py jlpr.py infer.py eval_ocr.py strip_nms.py  ✅ / gen.py train_*.py  (M4 以降)
           parity/labels.py          ラベル表の一致テスト ✅（生成・loss・推論は今後）
models/    plate_det_pyj320.onnx ✅(暫定・借り物) / plate_ocr.onnx ✅ / plate_corner.onnx (M6)
wasm/      jlpr_wasm.cpp + index.html + test_node.js  ✅（カメラ/ファイル/手動枠/PNG保存）
```
- C++ ビルドは単一 TU、`cl /std:c++20 /O2` ／ `g++ -std=c++20 -O2` ／ `emcc -msimd128`。`-DUSE_EIGEN` で CPU 高速化、`-DUSE_CUDA` で GPU
- CLI は両言語で揃える: `jlpr train --model ocr --data … --epochs …` ⇔ `python tools/train_ocr.py --data … --epochs …`（引数名も同じ）
- WASM デモは 3 モデルを実行時 fetch。検出は 640 が重いので 416/512 の ONNX を併置し、ボタン押しで実行

## 既知の落とし穴（姉妹リポで実測済み。再発させない）

| 事象 | 実測 | 対策 |
|---|---|---|
| ONNX head の活性を二重適用 | YOLOX 版で sigmoid 2回 → 全セル score ≒0.5、box が画面中に散った。forward の parity(1.4e-05) は通っていた | parity は**デコードを保証しない**。ONNX の head が logits か確率かを必ず確認し、実写で box が載ることをテストで assert |
| 地域名 head の余白依存 | 同じ写真が 奄美 / 横浜 / 練馬 と入れ替わる | 4隅補正＋固定マージン（本リポの2段目） |
| 検出器のプレート色バイアス | 白 0.85 / 黄 0.40 / **黒 0.21**（同一写真・色だけ変更） | 学習データの色均衡＋色別 recall を常時報告 |
| **近接プレートが取れない（借り物の検出器 2 つとも）** | `tools/context_test.py` で同じ写真の切り出し範囲だけを変えた実測: プレートが画面幅の 5% → 0.757、8% → **0.826**、12% → 0.695、20% → 0.069、**30% 以上は検出ゼロ**。旧 YOLOX も同じ 5–12% 帯だった | 車ごと写る画角と近接の両方を学習データに入れる（M7 の合格条件に「画面幅 5%–95% で検出」を入れた）。デモは「枠を手で指定」を常備し、ページに実測帯を明記 |
| **box の質が地域名 head を支配する** | 検出器を替えただけで地域名 conf 0.57 → 0.92（同じ写真・同じ認識器） | 4隅補正で入力正規化を固定する（M6）。認識器の精度議論の前に box を疑う |
| プレート色バイアスの再現は方法依存 | 同じ写真で色だけ変える試験（`tools/recolor_test.py`、陰影を保つ置換）では **白 0.796 / 緑 0.799 / 黄 0.721 / 黒 0.835**。旧 YOLOX でも同じ方法なら 0.80/0.72/0.645/0.720 で、lpr_cpp が報告した黒 0.21 は再現しない | 色バイアスの判定は**実写の黒/黄ナンバー**で行う。合成的な色替えは陰影が残るので簡単すぎる |
| 自作 ONNX インタプリタの float 誤差 | 認識器 3.3e-05 / 検出器 3e-03（onnxruntime は同じ ONNX で 7.5e-09） | 強い検出には影響しないが、閾値ぎりぎりの box では読みが両実装で割れる。パリティテストの許容幅はそれを前提に設定 |
| ネガだけのテスト | 灰色一枚で box 0 件 → 検出器が壊れていても通る | 実写フィクスチャで位置・スコアの決定性・散らばりの無さを assert |
| 合成の指標で採否を決める | 4隅回帰を作り直して合成 val 誤差 1.93% → 1.89% と改善したのに、実写では地名確信度 0.94 → 0.75 と**悪化**（`plate_corner_v2.onnx`、採用せず） | モデルの採否は必ず**実写**で決める。合成の指標は「学習が進んでいるか」までしか言えない |
| Kaggle 側で `git pull` が中断する | 学習が書いた `models/*.onnx` が未追跡のまま残り、`untracked working tree files would be overwritten by merge` で job が即死（3回踏んだ） | 学習 job の先頭を必ず `git checkout -- models/; git clean -fdq models/; git pull` にする |
| Ultralytics の ONNX を `simplify=False` で出す | グラフに Shape/Gather が残り、自作インタプリタが `tensor '/model.22/Gather_output_0' is missing` で停止（onnxruntime は読める） | 検出器の export は必ず `simplify=True`（onnxslim が畳んでくれる）。読めるかどうかは **自作ランタイムで実写1枚**流して確認 |
| 重みの保存先を自分で組み立てる | `project/name/weights/best.pt` を組み立てていたが Ultralytics の settings が runs_dir を前置しており、71 分の学習の**直後に** export だけ FileNotFoundError で落ちた | `model.trainer.best` に聞く。長い学習には `--export-only` のような再開口を用意しておく |
| 追加クラスの初期 bias を固定値にする | 学習しないクラスは bias -10 で無害化する必要があるが、**学習するクラスに -10 のままだと**1000 step 使っても確率 1e-7・138位中133位のまま | bias は「そのクラスに勾配が来るか」で決める（`--new-class-bias`、既定は自動判定） |

## 法務・プライバシー

- プレート番号は総務省資料の解釈上「個人情報に該当しない」とされるが、周辺情報と結びつくと話が変わる。
  `alpr_jp` の方針を踏襲し、**公開する画像はプレート周辺だけにトリミングし、撮影日時・場所を紐付けない**。
- 事前学習重みは Ultralytics 由来（AGPL-3.0）。自前コードは BSD-3-Clause、同梱物は各ライセンスに従う。
