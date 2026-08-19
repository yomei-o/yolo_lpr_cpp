// GPU check — the same engine, the same ONNX files, on CUDA.
//
// The device seam is pure/backend.hpp: every GEMM the engine performs (conv2d forward, both conv2d
// backward products, the linear layers) goes through bk::gemm_hosted / gemm_nt_hosted /
// gemm_tn_hosted, which stage to the device and back when built with -DUSE_CUDA. Nothing else in the
// engine changes, so "does the GPU path work" is answerable by running the ordinary code and
// comparing numbers with the CPU build:
//
//   sh build/nvcc.sh pure/gpu_check.cpp -o gpu_check.exe        # CUDA (needs an NVIDIA GPU to run)
//   sh build/cc.sh   pure/gpu_check.cpp -o gpu_check_cpu.exe    # same source, CPU
//   ./gpu_check.exe --det models/plate_det_v8n_320.onnx --data data/det_smoke
//
// It prints the backend it was built with, a detector forward (the six head tensors' checksums), one
// full training step's loss and gradient norm, and the corner net's forward — so a CPU run and a GPU
// run can be diffed line by line. This machine has the CUDA toolkit but no NVIDIA GPU, so what is
// verified here is that it *compiles and links* with nvcc and that the CPU build of the same file
// produces the reference numbers; colab/gpu_check.ipynb runs it on real hardware.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "spec.hpp"
#include "pipeline.hpp"
#include "train_det.hpp"
#include "train_corner.hpp"
#include <cstdio>
#include <string>

static std::string arg_of(int argc, char** argv, const std::string& key, const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i) if (key == argv[i]) return argv[i + 1];
  return def;
}

static double checksum(const Tensor& t) {
  double s = 0;
  for (int64_t i = 0; i < t->numel(); ++i) s += (double)t->data[(size_t)i] * (double)((i % 7) + 1);
  return s;
}

int main(int argc, char** argv) {
  const std::string det_p = arg_of(argc, argv, "--det", "models/plate_det_v8n_320.onnx");
  const std::string corner_p = arg_of(argc, argv, "--corner", "models/plate_corner.onnx");
  const std::string data = arg_of(argc, argv, "--data", "data/det_smoke");
  const int batch = std::atoi(arg_of(argc, argv, "--batch", "2").c_str());
  printf("backend: %s\n", bk::backend_name());

  std::vector<det::Item> items = det::read_yolo(data);
  if (items.empty()) { printf("no images under %s/images — generate with `jlpr gen-det`\n", data.c_str()); return 1; }
  if ((int)items.size() > batch) items.resize((size_t)batch);
  std::vector<int> idx;
  for (size_t i = 0; i < items.size(); ++i) idx.push_back((int)i);

  onx::Graph g = onx::load_onnx(det_p);
  det::HeadNames hn;
  std::string why;
  if (!det::find_v8_heads(g, hn, &why)) { printf("%s: %s\n", det_p.c_str(), why.c_str()); return 1; }
  int iw = 0, ih = 0;
  jl::graph_input_hw(g, iw, ih);
  const int imgsz = iw > 0 ? iw : 320;

  std::set<std::string> needed(hn.box.begin(), hn.box.end());
  needed.insert(hn.cls.begin(), hn.cls.end());
  onx::Trainable t = onx::make_trainable(g, false, needed);
  det::Batch ba = det::make_batch(items, idx, imgsz);

  det::LossOut rep;
  std::vector<Tensor> bxs, css;
  det::LossCfg cfg;
  Tensor loss = det::forward_loss(t, hn, ba.x, ba.gts, imgsz, cfg, &rep, &bxs, &css);
  printf("detector forward (%d images at %d px):\n", (int)items.size(), imgsz);
  for (size_t l = 0; l < bxs.size(); ++l)
    printf("  level %zu  box checksum %.4f   cls checksum %.4f\n", l, checksum(bxs[l]), checksum(css[l]));
  printf("loss %.6f (box %.6f cls %.6f dfl %.6f) fg %d tss %.4f\n", rep.total, rep.box, rep.cls,
         rep.dfl, rep.fg, rep.tss);

  backward(loss);
  double gn = 0;
  for (const Tensor& p : t.params) for (float v : p->grad) gn += (double)v * v;
  printf("gradient L2 over %zu parameter tensors: %.6f\n", t.params.size(), std::sqrt(gn));
  free_graph(loss);

  onx::Graph cg = onx::load_onnx(corner_p);
  onx::Trainable ct = onx::make_trainable(cg);
  Tensor cx = make_tensor({1, 3, crn::IN_PX, crn::IN_PX}, false);
  for (int64_t i = 0; i < cx->numel(); ++i) cx->data[(size_t)i] = (float)((i * 37 % 255) / 255.0);
  std::map<std::string, Tensor> cv = onx::run_onnx(ct.g, cx, {}, &ct.init, false);
  const Tensor& corners = cv.at(ct.g.outputs[0].name);
  printf("corner forward:");
  for (int i = 0; i < 8; ++i) printf(" %.5f", corners->data[(size_t)i]);
  printf("\n");
  return 0;
}
