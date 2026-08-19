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

**現在の状態: ワンパス推論が CLI と WASM で動く（既存モデル）。学習系はこれから。**
進捗と残作業は [RESUME.md](RESUME.md)。

## 動かす

```sh
sh build/gcc.sh pure/jlpr.cpp -o jlpr.exe          # または sh build/cc.sh (MSVC, vcvars 不要)

./jlpr.exe detect --img assets/tokyu-bus-yokohama200ka3591.jpg \
                  --det models/plate_det_yolox.onnx --ocr models/plate_ocr.onnx --out out.png
#   plates: 2 (conf>=0.15, 6 raw detections over all classes)
#     [0] box (250,407)-(337,465) det 0.82  crops 6
#          横浜 200 か 3591
#          conf 0.57 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00

python tools/parity/labels.py                       # ラベル表の C++⇔Python 一致 (M1)
./jlpr.exe parity-ocr --ocr models/plate_ocr.onnx --ref <lpr_cpp>/pure/ref
#   exported ONNX vs original-ONNX fixture: worst 3.299e-05   argmax 9/9   MATCH
```

ブラウザ（WASM、送信なし・完全ローカル）:

```sh
sh build/emcc.sh wasm/jlpr_wasm.cpp -o wasm/jlpr.js
python -m http.server 8000     # リポジトリのルートから → http://localhost:8000/wasm/
node wasm/test_node.js         # ヘッドレスの回帰テスト（CLI と同じ画素で同じ文字列を assert）
```

実測（実写 960×640、CPU 1 スレッド）: CLI **5.6 秒** / WASM(node, SIMD) **3.82 秒** で
`横浜 200 か 3591`。内訳は検出 416px が大半、認識は 6 クロップ合算。

**まだ暫定なところ**: 検出器は `lpr_cpp` の交通カメラ由来 YOLOX-tiny（8クラス、class 7 = plate）を
そのまま使っている。yolov8n nc=1 に差し替えるのは M7。4隅補正（M6）も未実装なので、地域名 head の
信頼度はこの写真で 0.57 しかない（他の head は 1.00）——「なぜ 4隅補正を足すのか」がそのまま数字に出ている。

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
| ラベル表（地名138/かな/分類番号/種別） | `labels.py` | `lpr_labels.hpp` | 1つの定義ファイル（`spec/labels.txt`）から**両方を生成**。突き合わせテストでバイト一致 |
| 合成データ生成 | `gen.py` | `gen.cpp` | 同じ seed で**文字列・4隅・変換パラメータ・色・種別が完全一致**。画像はラスタライザ差（PIL/FreeType vs stb_truetype）が出るので、一致は幾何とラベルまでとし、画像は統計と目視で同等を確認 |
| 学習（検出 / 4隅 / 認識） | PyTorch (+Ultralytics) | 自作 autograd（`yolov8_cpp`/`lpr_cpp` 由来） | 同じ初期重み・同じバッチ・同じ LR で **loss と勾配が 1e-4 以内**。小データ・数ステップで縛る |
| 推論 | onnxruntime または自前 | 自前 ONNX ランナ | 同一画像で**同一文字列**、スコア差 1e-4 以内 |
| 評価（mAP / 全文一致 / 色別内訳） | `eval.py` | `jlpr val` | 同じデータセットで**同じ数値**（mAP は 1e-3 以内） |
| ONNX 出力 | `torch.onnx.export` | 自前エクスポータ | 出力 ONNX を相互に読み込んで **forward が 1e-5 以内** |
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

### 集める（実データ）
| 出所 | 内容 | ライセンス | 用途 |
|---|---|---|---|
| [dyama/alpr_jp](https://github.com/dyama/alpr_jp) | プレート crop 1,249枚（うち **720枚が `自家用/<地名>/` 構造＝地域名ラベル付き**, 66地名）＋ ネガ 4,420枚 | MIT（撮影者に著作権） | 認識器の実データ微調整・地域名 head の検証・検出のハードネガ |
| [Number Plate in Japan (Roboflow, 2,196枚)](https://universe.roboflow.com/moriken/number-plate-in-japan), [license-plate-japan](https://universe.roboflow.com/new-workspace-vijtn/license-plate-japan) | 日本車の全景＋プレート box | **要確認**（RESUME の未確認事項） | 検出器の学習・評価 |
| Open Images V7 `Vehicle registration plate` | 全世界のプレート box（大量） | 画像 CC BY 2.0 / 注釈 CC BY 4.0 | 検出器の量稼ぎ（日本以外も形状は近い） |
| 自前撮影 | 黒・黄プレート、近接、夜間 | 自前 | 弱点の埋め合わせと最終評価 |

### 生成する（合成）
Python の生成器を書く。プレート**画像そのもの**と**車体に貼った全景**の両方を出す。
- テンプレート: 中型 330×165 / 大型 440×220、白地緑字・黄地黒字・緑地白字・黒地黄字、図柄入りは背景に模倣柄
- 書体: [FGゼロラバウル](https://coliss.com/articles/freebies/font-fgzerorabaul.html)（プレート書体を再現した商用可フリーフォント、漢字かな入り）を軸に複数フォントを混ぜ、字幅・太さ・字間をジッタ（1書体に寄せると本物で滑る）
- 実在ルール: 地名138、分類番号の桁数とアルファベット規則、ひらがなの使用可集合、一連番号の `・` 詰め
- 物理: ボルト・封印、汚れ・錆・水滴、フレーム、影・反射・ヘッドライト、パース(yaw±35° / pitch±20°)、モーションブラー、JPEG劣化、幅 24–200px の低解像度
- 規模の目安: 認識 20–40万枚、検出コンポジット 2–5万枚（実データはその 5% 程度）
- 合成では **4隅・全head ラベルが正解付きで得られる**ので、2段目と3段目はここで作り込む

### 学習の段取り
1. 合成のみで 3 モデルを事前学習（Colab T4 で検出 1.5–3h / 認識 1–2h / 4隅 30分 が目安）
2. 認識器は既存 ONNX 重みからの転移 ＋ 実データ（alpr_jp 720枚 + 自前）で低 LR 微調整（head→backbone の段階解凍）
3. 既存モデルで実データに疑似ラベル → TTA 一致分だけ採用 → 不一致だけ人手修正（能動学習）
4. 評価は **実データ hold-out のみ**で、色別・地名別・解像度別に出す

目標値: 検出 mAP@0.5 ≥ 0.95（色別 recall も ≥ 0.90）、認識は**プレート全文一致 ≥ 95%**、region ≥ 98%（TTA なし）。

## 実装（同じ機能を2系統）

```
spec/      labels.txt                地名/かな/分類番号/種別の唯一の定義（C++・Python 両方をここから生成）
           pipeline.md               入出力・前処理・マージン・閾値の仕様（両実装が従う）
pure/      onnx.hpp / onnx_run.hpp   ONNX リーダ（ファイル/メモリ）＋統合インタプリタ  ✅
           onnx_export_lpr.hpp       認識器の重み(manifest+weights.bin) → ONNX  ✅
           autograd/…                自作 autograd・conv・BN・Adam（姉妹リポから移植）
           infer_yolox.hpp           暫定検出器の decode+NMS  ✅（yolov8 用は M7 で追加）
           crop.hpp                  letterbox / box→128×128 crop  ✅（4隅 warp は M6）
           spec.hpp                  spec/labels.txt を実行時パース＋decode  ✅
           pipeline.hpp              detect → crop → classify → 文字列＋信頼度  ✅
           gen.cpp                   合成データ生成（C++ 版、M4）
           jlpr.cpp                  CLI: labels / export / parity-ocr / detect / rgba  ✅
tools/     labels.py rng.py jlpr.py  ✅ / gen.py train_*.py eval.py  (M4 以降)
           parity/labels.py          ラベル表の一致テスト ✅（生成・loss・推論は今後）
models/    plate_det_yolox.onnx ✅(暫定) / plate_ocr.onnx ✅ / plate_corner.onnx (M6)
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
| 近接プレートが取れない | 画面幅の 5–12% が最良帯、46% だと score 0.02 で誤位置 | 車ごと写る画角と近接の両方を学習データに入れる。2パス検出は任意 |
| ネガだけのテスト | 灰色一枚で box 0 件 → 検出器が壊れていても通る | 実写フィクスチャで位置・スコアの決定性・散らばりの無さを assert |

## 法務・プライバシー

- プレート番号は総務省資料の解釈上「個人情報に該当しない」とされるが、周辺情報と結びつくと話が変わる。
  `alpr_jp` の方針を踏襲し、**公開する画像はプレート周辺だけにトリミングし、撮影日時・場所を紐付けない**。
- 事前学習重みは Ultralytics 由来（AGPL-3.0）。自前コードは BSD-3-Clause、同梱物は各ライセンスに従う。
