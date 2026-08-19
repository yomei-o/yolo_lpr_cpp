# Third-party notices

Own code in this repository is BSD-3-Clause (see `LICENSE`). Bundled and derived material keeps its
own license:

| What | Where | License |
|---|---|---|
| stb_image / stb_image_write (Sean Barrett) | `pure/third_party/stb_*.h` | MIT / public domain (dual) |
| Eigen (flattened Core subset) | `pure/third_party/eigen_flat/` | MPL-2.0 (`COPYING.MPL2`) |
| Engine headers ported from [yolov8_cpp](https://github.com/yomei-o/yolov8_cpp), [yolox_cpp](https://github.com/yomei-o/yolox_cpp), [lpr_cpp](https://github.com/yomei-o/lpr_cpp) | `pure/*.hpp` | BSD-3-Clause (same author) |
| YOLOv8 architecture and `yolov8n.pt` pretrained weights (Ultralytics) | fine-tuned detector weights | **AGPL-3.0** — models derived from them inherit it |
| **PlateYOLO-JP** detector weights (Kazuhito00/PlateYOLO-JP-Prototype, YOLO12) — shipped here as `models/plate_det_pyj320.onnx`, the same graph with its NMS tail cut off by `tools/strip_nms.py` | `models/plate_det_pyj320.onnx` | **AGPL-3.0** (upstream). Interim detector until M7 trains ours |
| EkMixer label space (region/hiragana/class-number dictionaries from the same repo's `util.py`) | `spec/ekmixer_labels.txt` | AGPL-3.0 (upstream); used to interpret that model's outputs when benchmarking |
| Recognizer weights derived from lpr_cpp's `weights.bin` | `models/plate_ocr.onnx` | see lpr_cpp (govtech pipeline origin) |
| Plate crops from [dyama/alpr_jp](https://github.com/dyama/alpr_jp) (training data, not committed here) | — | MIT (copyright stays with the photographers) |
| Open Images V7 annotations / images (training data, not committed here) | — | CC BY 4.0 / CC BY 2.0 |
| Fonts used by the synthetic generator | fetched by `tools/gen/fetch_fonts.py`, not committed | per each font's own terms |

The detector's pretrained initialisation is Ultralytics-derived on purpose (README "確定した方針"),
so the **detector weights are AGPL-3.0**. The recognizer and corner models do not depend on it.
