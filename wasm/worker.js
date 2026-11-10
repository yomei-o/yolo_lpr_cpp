// Inference worker. The WASM pipeline takes ~0.3-1.5 s per frame, so running it on the page's thread
// would freeze the UI and the camera preview between frames. Everything heavy lives here; the page
// only sends pixels and draws what comes back.
//
// Messages in:  {type:'init', ocr, det, corner, spec, detKind}   ArrayBuffers for the models
//               {type:'frame', pixels, w, h, conf, tta, detKind, track}
//               {type:'reset'}                                   clear the tracks
// Messages out: {type:'ready', ...} | {type:'result', json, ms, overlay, w, h} | {type:'log', text} | {type:'error'}
//               overlay = RGBA layer (transparent except the plate frames + numbers), drawn in C++ by jl_draw
let M = null;
let ready = false;

function log(text) { self.postMessage({ type: 'log', text: text }); }

function copyIn(bytes) {
  const p = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, p);
  return p;
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      self.importScripts('jlpr.js');
      M = await createJlpr();
      const specText = m.spec;
      const n = M.lengthBytesUTF8(specText) + 1;
      const sp = M._malloc(n);
      M.stringToUTF8(specText, sp, n);
      const groups = M._jl_load_spec(sp);
      M._free(sp);

      const loadModel = (buf, fn, name) => {
        if (!buf) return 0;
        const b = new Uint8Array(buf);
        const p = copyIn(b);
        const r = fn(p, b.length);
        M._free(p);
        log(name + (r > 0 ? ' ok (' + r + ' nodes, ' + (b.length / 1048576).toFixed(2) + ' MB)'
                          : ' failed to load'));
        return r;
      };
      const ocr = loadModel(m.ocr, M._jl_load_ocr, '認識モデル');
      const det = loadModel(m.det, M._jl_load_det, '検出モデル');
      const corner = loadModel(m.corner, M._jl_load_corner, '4隅補正モデル');
      ready = ocr > 0;
      self.postMessage({ type: 'ready', ocr: ocr, det: det, corner: corner, groups: groups });
      return;
    }

    if (m.type === 'reset') {
      if (M) M._jl_track_reset();
      return;
    }

    if (m.type === 'frame') {
      if (!ready) return;
      const px = new Uint8Array(m.pixels);
      const p = copyIn(px);
      const t0 = performance.now();
      let json;
      const bx = m.box || [-1, -1, -1, -1];      // a hand-drawn box skips the detector
      // tilted plates: search +-30 deg frames when upright finds nothing. No extra corner pass:
      // measured with `jlpr rot-test`, the cascade refinement lowers region accuracy.
      if (M._jl_set_rotation) M._jl_set_rotation(m.rot ? 1 : 0, 0);
      if (m.track) {
        M._jl_track_step(p, m.w, m.h, m.conf, m.tta ? 1 : 0, m.detKind);
        json = M.UTF8ToString(M._jl_tracks());
      } else {
        M._jl_run(p, m.w, m.h, m.conf, m.tta ? 1 : 0, bx[0], bx[1], bx[2], bx[3], m.detKind);
        json = M.UTF8ToString(M._jl_result());
      }
      const ms = performance.now() - t0;
      // The frame and number are drawn in C++ (jl_draw) into a transparent layer, reusing the
      // frame's buffer; the page lays it over the image / live camera preview.
      M._jl_draw(p, m.w, m.h, 1);
      const overlay = M.HEAPU8.slice(p, p + m.w * m.h * 4).buffer;
      M._free(p);
      self.postMessage({ type: 'result', json: json, ms: ms, track: !!m.track,
                         overlay: overlay, w: m.w, h: m.h }, [overlay]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', text: String(err && err.message || err) });
  }
};
