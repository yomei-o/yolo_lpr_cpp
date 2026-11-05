// Runs wasm/test_node.js over the rotation fixtures: the WASM build must give the CLI's reading with
// the page's 「傾いたナンバーも探す」 on (ROT=1) and off. Expected strings are what
// `jlpr detect --det models/plate_det_v8n_640.onnx --fmt cxcywh --corner models/plate_corner.onnx
//  [--rot-search 0,30,-30]` prints for the same image.
//
// fixtures (jlpr rgba): scratch/sample.rgba = assets/tokyu-bus-...jpg, scratch/kei.rgba = the kei
// photo, scratch/kei+30.rgba etc. = the kei photo rotated by that many degrees (PIL rotate, expand).
//   node wasm/test_rot_node.js
const { spawnSync } = require('child_process');
const path = require('path');

const KEI = '480 り 4567';
const cases = [
  // upright: rotation on must change nothing (the search only runs when upright finds nothing)
  { name: 'bus, rot off', fx: 'sample.rgba', text: '横浜 200 か 3591', env: { ROT: '0' } },
  { name: 'bus, rot on', fx: 'sample.rgba', text: '横浜 200 か 3591', env: { ROT: '1' } },
  { name: 'kei, rot on', fx: 'kei.rgba', text: '練馬 ' + KEI, env: { ROT: '1', EXPECT_BOX: '255,217,399,282' } },
  // rotated: found only with the search on
  { name: 'kei +30, rot on', fx: 'kei+30.rgba', text: '横浜 ' + KEI, env: { ROT: '1', EXPECT_BOX: 'none' } },
  { name: 'kei +45, rot on', fx: 'kei+45.rgba', text: '横浜 ' + KEI, env: { ROT: '1', EXPECT_BOX: 'none' } },
  { name: 'kei -30, rot on', fx: 'kei-30.rgba', text: '練馬 ' + KEI, env: { ROT: '1', EXPECT_BOX: 'none' } },
  { name: 'kei -45, rot on', fx: 'kei-45.rgba', text: '練馬 ' + KEI, env: { ROT: '1', EXPECT_BOX: 'none' } },
  { name: 'kei +45, rot OFF (expected to find nothing)', fx: 'kei+45.rgba', text: '横浜 ' + KEI,
    env: { ROT: '0', EXPECT_BOX: 'none' }, expectFail: true },
];

let bad = 0;
for (const c of cases) {
  const r = spawnSync(process.execPath, [path.join(__dirname, 'test_node.js'),
    path.join(__dirname, '..', 'scratch', c.fx), c.text], { env: { ...process.env, ...c.env }, encoding: 'utf8' });
  const out = (r.stdout || '') + (r.stderr || '');
  const pass = /PASS/.test(out);
  const ok = c.expectFail ? !pass : pass;
  const detail = out.split('\n').filter((l) => /plate\(s\)|FAIL|^  \S/.test(l)).join(' / ');
  console.log((ok ? 'ok   ' : 'BAD  ') + c.name + '  — ' + detail.trim());
  if (!ok) ++bad;
}
console.log(bad ? bad + ' case(s) BAD' : 'all ' + cases.length + ' cases as expected');
process.exit(bad ? 1 : 0);
