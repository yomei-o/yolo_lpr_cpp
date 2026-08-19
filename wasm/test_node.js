// Headless check of the WASM build: load the two ONNX models + the label spec, push the same
// pixels the CLI saw (a .rgba fixture from `jlpr rgba`), and assert the plate reads correctly.
//
// A test that only feeds a blank frame and expects zero boxes passes even when the detector is
// broken — that is exactly how a double-sigmoid bug survived in lpr_cpp. So this asserts on a real
// photo: the box must land on the plate, the score must be decisive, and the text must match.
//
//   node wasm/test_node.js [fixture.rgba] [expected text]
// default fixture: scratch/sample.rgba  (jlpr rgba --img assets/tokyu-bus-...jpg --out ...)
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const fixture = process.argv[2] || path.join(ROOT, 'scratch', 'sample.rgba');
const expect = process.argv[3] || '横浜 200 か 3591';
const EXPECT_BOX = (process.env.EXPECT_BOX || '248,414,321,463').split(',').map(Number);
// which detector to test: 1 = v8 head with xyxy boxes (PlateYOLO-JP, NMS stripped),
// 2 = v8 head with cxcywh boxes (a plain Ultralytics export of our own yolov8n), 0 = the old YOLOX
const DET_KIND = Number(process.env.DET_KIND || 1);
const DET_PATH = process.env.DET_PATH || path.join(ROOT, 'models', 'plate_det_pyj320.onnx');
// The demo loads the corner model, so the test has to as well or it exercises a path nobody uses:
// without it the reading falls back to multi-margin TTA, which on the close-up black plate reads
// 練馬 instead of 横浜. CORNER_PATH=none turns it off deliberately.
const CORNER_PATH = process.env.CORNER_PATH || path.join(ROOT, 'models', 'plate_corner.onnx');

function fail(msg) { console.log('FAIL: ' + msg); process.exit(1); }

(async () => {
  if (!fs.existsSync(fixture)) {
    fail('no fixture at ' + fixture + '\n  make one: ./jlpr.exe rgba --img assets/tokyu-bus-yokohama200ka3591.jpg --out scratch/sample.rgba');
  }
  const createJlpr = require('./jlpr.js');
  const M = await createJlpr();

  const specText = fs.readFileSync(path.join(ROOT, 'spec', 'labels.txt'), 'utf8');
  const n = M.lengthBytesUTF8(specText) + 1;
  const sp = M._malloc(n);
  M.stringToUTF8(specText, sp, n);
  const groups = M._jl_load_spec(sp);
  M._free(sp);
  if (groups <= 0) fail('spec did not load');

  const load = (file, fn) => {
    const b = fs.readFileSync(file);
    const p = M._malloc(b.length);
    M.HEAPU8.set(b, p);
    const r = fn(p, b.length);
    M._free(p);
    if (r <= 0) fail('model did not load: ' + file);
    return r;
  };
  const ocrNodes = load(path.join(ROOT, 'models', 'plate_ocr_v2.onnx'), M._jl_load_ocr);
  const detNodes = load(DET_PATH, M._jl_load_det);
  const cornerNodes = (CORNER_PATH !== 'none' && fs.existsSync(CORNER_PATH))
    ? load(CORNER_PATH, M._jl_load_corner) : 0;
  console.log('loaded: spec ' + groups + ' groups, ocr ' + ocrNodes + ' nodes, det ' + detNodes
              + ' nodes, corner ' + cornerNodes + ' nodes');

  const raw = fs.readFileSync(fixture);
  const w = raw.readInt32LE(0), h = raw.readInt32LE(4);
  const px = raw.subarray(8);
  if (px.length !== w * h * 4) fail('fixture size mismatch');
  const pf = M._malloc(px.length);
  M.HEAPU8.set(px, pf);

  const t0 = Date.now();
  const nplates = M._jl_run(pf, w, h, 0.30, 1, -1, -1, -1, -1, DET_KIND);
  const secs = (Date.now() - t0) / 1000;
  const res = JSON.parse(M.UTF8ToString(M._jl_result()));
  M._free(pf);

  console.log('detect+read: ' + nplates + ' plate(s) in ' + secs.toFixed(2) + 's on ' + w + 'x' + h);
  for (const p of res.plates) {
    console.log('  ' + p.text + '   det ' + p.det.toFixed(2) + '  region conf ' + p.conf[0].toFixed(2) +
                '  box ' + p.box.map((v) => v.toFixed(0)).join(','));
  }
  if (nplates < 1) fail('no plate detected on a real photo');

  const best = res.plates[0];
  if (best.text !== expect) fail('text is "' + best.text + '", expected "' + expect + '"');
  if (best.det < 0.5) fail('detector score ' + best.det.toFixed(2) + ' is not decisive');
  for (let i = 0; i < 4; ++i) {
    if (Math.abs(best.box[i] - EXPECT_BOX[i]) > 8) {
      fail('box ' + best.box.map((v) => v.toFixed(0)).join(',') + ' drifted from the CLI box ' + EXPECT_BOX.join(','));
    }
  }
  // a scatter of boxes at the threshold is the double-sigmoid signature; there should be few
  if (res.plates.length > 3) fail('too many plate boxes (' + res.plates.length + ') — decode looks wrong');

  console.log('PASS: WASM matches the CLI reading (' + expect + ')');
})();
