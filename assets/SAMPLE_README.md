# Sample image — attribution

`tokyu-bus-yokohama200ka3591.jpg` is the demo's built-in test photo. It is **not** our photograph,
and it is redistributed here under its own licence:

| | |
|---|---|
| Title | TokyuBus H1283 rear |
| Author | **103momo** ([Wikimedia Commons](https://commons.wikimedia.org/wiki/User:103momo)) |
| Source | https://commons.wikimedia.org/wiki/File:TokyuBus_H1283_rear.jpg |
| Licence | **CC BY-SA 4.0** — https://creativecommons.org/licenses/by-sa/4.0 |
| Date | 2013-07-06 |
| Changes | downscaled from 2592×1728 to the 960 px Commons thumbnail; nothing else |

CC BY-SA is *share-alike*: if you modify this image and redistribute it, the modified image must
carry the same licence. That obligation attaches to the image only — the code in this repository is
a separate work and keeps its own licence.

`plate.html` shows the credit on the page whenever the sample is loaded, which is what the licence
requires; do not remove it.

## Why this photo

Ground truth is **横浜 200 か 35-91**, a green commercial plate on a Tokyu bus, and the pipeline gets
it exactly right end to end:

    detect     (250,407)-(337,465)  score 0.818
    recognise  横浜 200 か 3591      agreement 0.56 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00

It also has an honest wart, deliberately left visible: a second candidate at **0.242** on the bus's
rear window that reads as junk (`鹿児島 80L ほ 85`, agreement 0.20/0.43/0.39/…). The page loads the
sample at conf 0.35 so the correct plate stands alone; drop the slider below 0.24 and the false
positive appears, with its low agreement scores marking it as noise. That is worth seeing.

## Not usable for recognition: the other fixture

`wasm/make_fixtures.py` also downloads **ERA Mini Turbo rear.JPG** as `jp1` for `test_yolox.js`. The
detector finds its plate cleanly (0.832), but the plate is a **New Zealand** one — CHRISTCHURCH,
HZF440 — so the Japanese classifier returns nonsense (`山口 200 あ 2540`, region agreement 0.28).
Use jp1 to test detection only. It is not committed here for that reason, among others.
