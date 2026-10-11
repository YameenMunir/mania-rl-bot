<p align="center">
  <img src="assets/logo.png" alt="mania RL Bot logo" width="200">
</p>

# osu!mania 4K RL bot

A small project that trains a neural network (PPO) to play osu!mania 4K and get better with every play.
It runs against a fast simulator of osu!mania scoring instead of the real game, so thousands of plays
take minutes.

> **Disclaimer:** This is an independent fan project. It is **not affiliated with, associated with, endorsed by, or
> connected to osu!, ppy Pty Ltd, or the osu! team** in any way. "osu!" and "osu!mania" are trademarks of
> their respective owners and are used here only to describe compatibility with the game and its file formats.
> It does not modify the game, read its memory, or inject input; see [osu! stable](#osu-stable).

## The app

`python app.py` opens a desktop app (Tkinter, no extra dependencies) that covers the whole workflow:

| Section | What you can do |
|---|---|
| **Dashboard** | Totals, last trained / best model, training status, device and osu! status, quick actions |
| **Models → Create New Model** | Name, maps (a folder, or a single `.osu` / `.osz` / `.zip` via **File…**; 4K maps are validated, everything else is listed and skipped), training length, learning rate, batch size, checkpoints, device, seed, advanced PPO settings, then a summary before training starts |
| **Models → Existing Models** | Model Library: rename, archive, delete, open a model |
| **Models → Version History** | Lineage tree of every version: test, generate a replay, continue training from it, set active, mark best, protect, favourite, archive, evaluate, compare, delete |
| **Models → Continue Training** | Resume from the active, best or latest version, any older version, or a checkpoint. Weights, optimiser state, step counter and learning-rate schedule are restored. Warns when a changed setting could hurt |
| **Training** | Live progress, losses, recent accuracy, checkpoints, ETA; pause, save checkpoint now, stop & save, cancel |
| **Test Model** | Pick model, version and a 4K map (a `.osu` file, or an `.osz` / `.zip` to choose one of its difficulties), then **Watch Bot Play** (the model plays live in a 4K playfield, then save the play and open it in osu!) or **Generate Replay** (no watching; writes an `.osr`) |
| **Replays** | Replay history with metrics; open in osu!, show in folder, copy path, generate again, delete. Entries survive deleting the model |
| **Settings** | Storage location (change it with a safe copy), storage usage, osu! stable folder, defaults, import old CLI runs, logs |

### Versions are never overwritten

Every training session creates a **new** version (`v001`, `v002`, ...) that records its parent, so
continuing from `v003` when `v005` exists creates `v006` with `parent_version: v003` and keeps
everything else. Saving is atomic: weights go to a temp folder, are verified, get their metadata,
and only then are renamed into place. Version numbers are never reused.

- **Active**: the version used by default for testing (you choose it; *Set as Active*).
- **Latest**: the most recently created version.
- **Best**: highest validation accuracy on held-out maps, unless you mark one by hand.
- **Protected** versions are kept by any cleanup and need an extra confirmation to delete.

Deleting asks for confirmation (whole models also require typing the name). A version that is
being trained from or tested is locked until that finishes.

### Where your data lives (not in Git)

Models, versions, checkpoints, training logs, evaluation results, replays and settings are stored
in your user data folder, **outside this repository**:

- Windows: `%LOCALAPPDATA%\OsuManiaAI`
- macOS: `~/Library/Application Support/OsuManiaAI`
- Linux: `~/.local/share/OsuManiaAI`

Set `OSUMANIAAI_HOME` to use another folder (the tests do this). `.gitignore` also ignores model
files, checkpoints, replays, logs and the old `runs/` and `replays/` folders as a second line of
defence. Models trained earlier with `train.py` can be copied into the library from the Dashboard
banner or *Settings → Import* (the originals are left untouched).

### osu! stable

The app finds osu! stable through `%LOCALAPPDATA%\osu!` or the `osu://` file association (or set
the folder in Settings). It hands osu! the `.osr` (to watch the replay) or an `.osz` (to import the
map first). There is no memory reading or input injection: "Watch Bot Play" runs the model live in
the app's own playfield, using the same simulator as training. Without osu! installed, everything
else still works.

Accuracy and judgement counts in the app come from the simulator, not from osu!. osu! re-judges a
replay with its own rules.

## Files

| File | What it does |
|---|---|
| `app.py` | Starts the desktop app |
| `assets/` | The logo (README, window icon and sidebar) in the sizes the app uses |
| `maniabot/storage.py` | User data folder, settings, atomic writes, storage migration |
| `maniabot/models.py` | Model library: versions, lineage, flags, checkpoints, sessions, safe deletion |
| `maniabot/training.py` | Training sessions (new / continue), checkpoints, background worker process |
| `maniabot/evaluation.py` | Inference and evaluation metrics (timing, lanes, holds, misses, extra presses) |
| `maniabot/replays.py` | Replay generation and history |
| `maniabot/beatmaps.py` | osu!mania 4K validation with reasons for rejected maps; unpacks `.osz` / `.zip` map archives into a cache |
| `maniabot/osu_integration.py` | Finds osu! stable, opens replays and mapsets |
| `maniabot/legacy.py` | Imports models trained with the old `train.py` |
| `maniabot/ui/` | The Tkinter interface |
| `osu_parser.py` | Parses `.osu` files (note times, lanes, holds, OD) |
| `mania_env.py` | Gymnasium env: 4 keys, real hit windows (MAX/300/200/100/50/miss), hold notes |
| `make_demo_maps.py` | Makes synthetic 4K maps so you can train immediately |
| `train.py` | Command-line PPO training (writes to `runs/`, which is git-ignored) |
| `evaluate.py` | Command-line scoring of a model (or a scripted oracle baseline) on full maps |
| `export_replay.py` | Command-line `.osr` export |
| `osr_writer.py` | The `.osr` file writer (header, LZMA frames, beatmap MD5) |

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest              # includes a real training worker process (marked slow)
python -m pytest -m "not slow"
```

## Quick start

```bash
pip install -r requirements.txt
python app.py                                  # the desktop app
```

Or with the original command-line scripts:

```bash
python make_demo_maps.py                       # writes maps/demo/*.osu
python evaluate.py --baseline --maps maps/demo # sanity check: oracle should score ~100%
python train.py --maps maps/demo --timesteps 300000 --out runs/demo
python evaluate.py --run runs/demo --maps maps/demo
```

Watch `runs/demo/episodes.csv` (one row per play) or `runs/demo/learning_curve.png`.

Reference run on 2 CPU cores: 700k steps (about 7 minutes) took the demo maps from about 6% to about 89%
accuracy, and it was still improving. Train longer (`--timesteps 2000000`) or use `--resume runs/demo/model.zip`
to keep going. Evaluation samples actions the way training does, because taking the single most likely
action per tick would never press a key.

## Using real maps

Download 4K osu!mania beatmaps from osu.ppy.sh (or use maps you already have) and point the app at them.
No unzipping is needed. In **Create New Model** or **Continue Training**, use **Browse…** for a folder or
**File…** for a single file. These all work:

- loose `.osu` files,
- `.osz` mapsets (an `.osz` is a zip),
- `.zip` files, either one mapset or a map pack containing many `.osz` files (unpacked one level deep).

Folders are searched recursively and may mix all of the above. The same applies to the command-line
scripts (`--maps <folder or file>`).

Archives are unpacked into a cache and your source files are never modified. Only `.osu` files and
audio are extracted (backgrounds and videos are skipped). The cache lives in
`<data folder>/cache/mapsets` (see *Where your data lives*) and is reused on later scans. If you
re-download or edit an archive it is unpacked again, and you can delete the folder at any time.

Only 4K maps with at least 50 notes are used. Everything else (5K to 14K, other game modes, corrupt
files or archives) is skipped and listed under **Show rejected maps** with the reason; nothing is
converted.

In **Test Model**, picking an `.osz` or `.zip` unpacks it and shows a list of its usable 4K
difficulties to choose from. A map pack with many mapsets lists every difficulty in it.

## Watching replays in osu! stable

A replay is tied to one exact beatmap by the MD5 of its `.osu` file, so the map has to be in your osu!
library first.

**Demo maps (quickest test):**

1. Drag every file in `maps/osz/` onto the osu! window (or double-click them) to import the 12 demo mapsets.
   Each has a click-track on the beat as its audio.
2. In the app: **Test Model → Generate Replay** on a `maps/demo` map, then **Open in osu!**
   (or from the command line: `python export_replay.py --run runs/demo --maps maps/demo --out replays`).
3. osu! plays the replay. You can also double-click an `.osr` file, or drag it onto osu!.

**Real maps:**

1. Download the map in osu! so it sits in your Songs folder.
2. Point `--maps` at your Songs folder (or at a folder of `.osz` / `.zip` downloads), for example `C:\Users\you\AppData\Local\osu!\Songs`.
   Use `--match "song name"` to pick one map, and `--tries 5` to keep the best of 5 plays.
3. `python export_replay.py --run runs/mine --maps "<Songs folder>" --match "song name" --out replays`

If osu! says the beatmap is missing, the `.osu` file you exported against is not byte-identical to the
one in your library (re-downloaded or edited maps get a new MD5). Export again from the library copy.

osu! re-judges the replay with its own rules, so the score screen can differ slightly from the
simulator's numbers, mostly on the 50/100 edges and on hold-note releases.

## Making "improving" mean something

A perfect-timing bot has nothing to learn, so use the human-like knobs:

- `--jitter-ms 8` adds Gaussian noise to every press and release time. The agent has to learn
  which behaviours stay safe under imprecision (for example, never mashing).
- `--dt-ms` sets the decision tick (default 10 ms). Smaller is more precise but slower.
- Compare against `python evaluate.py --baseline --jitter-ms 8` to see the ceiling at that noise level.

## How it works

- **Observation:** for each lane, the next 4 notes' time-to-hit (sqrt-scaled so it is most precise near
  zero), whether each is a hold note, plus key state and hold progress.
- **Action:** 4 binary values per tick, the key states. Up to down is a press, down to up is a release.
- **Reward:** accuracy weight plus a timing-precision bonus per hit, -1 per miss, small penalties for
  stray presses and broken holds.

## Ideas for next steps

1. Pretrain from replays (supervised) and fine-tune with RL.
2. Curriculum: start on low-OD, low-density maps and unlock harder ones as accuracy passes a threshold.
3. Add 5K to 7K by generalising `KEYS`.
4. Drive the real game: needs screen capture and input injection, and latency will dominate, so expect
   the simulator policy to need retraining with realistic delay and jitter.
