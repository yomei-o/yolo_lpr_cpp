# fonts/

The two faces the shipped models were trained with are committed here, so a clone can generate exactly
the datasets described in the README with no downloads at all. The face is an rng draw (draw #29 in
`spec/gen.md`), so **which fonts are present is part of the dataset definition**: the same seed with a
different set produces the same labels and different pixels. `spec/fonts.txt` is that contract, and
both generators print a `note: font set differs from spec/fonts.txt` when the directory does not match.

| file | licence | origin |
|---|---|---|
| `GenSenRounded2-B.ttc` | SIL OFL 1.1 (`SIL_Open_Font_License_1.1.txt`) | 源泉丸ゴシック, via [PlateYOLO-JP](https://github.com/Kazuhito00/PlateYOLO-JP-Prototype) which ships it with its licence |
| `DroidSansFallbackFull.ttf` | Apache-2.0 (`Apache_License_2.0.txt`) | Droid Sans Fallback, © 2006-2010 Google Corp., from the Android platform (Debian `fonts-droid-fallback`, whose copyright file states Apache-2). "Droid" is a trademark of Google Corp. |

Both are redistributable, which is the only reason they are here.

## Not committed, on purpose

`python tools/fetch_fonts.py --include-system` copies the Japanese faces already installed on the
machine — on Windows `meiryo.ttc`, `msgothic.ttc`, `YuGothB.ttc`. Those are proprietary and must never
be committed. They are worth having locally (four shapes instead of two makes the recogniser less
dependent on one skeleton), but note that adding them means your generated data is no longer identical
to what the shipped models saw. The generator records the face it used for every sample in `meta.txt`,
so a dataset can always be traced back.
