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

> **Personal project, all rights reserved.** This is my own personal work. You may not copy, modify,
> redistribute, or reuse any part of it, and contributions (pull requests, patches) are not accepted.
> See [Copyright and usage](#copyright-and-usage).

## Requirements

| | Minimum | Notes |
|---|---|---|
| **OS** | Windows 10 / 11 | Developed and tested on Windows 11. The Python code is cross-platform and has data-folder paths for macOS and Linux, but those are untested, and the osu! stable integration is Windows-only. |
| **Python** | 3.10 or newer, with Tkinter | Tested on 3.13. Tkinter ships with the python.org Windows installer (on Linux install `python3-tk`). |
| **CPU** | Any 64-bit CPU, 2 cores | Training runs on the CPU. Reference: 700k steps took about 7 minutes on 2 cores. More cores let it run more games in parallel, which is faster (the default is one fewer than your core count, up to 8). |
| **GPU** | Not needed | The network is small (two 128-unit layers), so the CPU build of PyTorch is enough. |
| **RAM** | 4 GB free (8 GB recommended) | Each parallel game is its own process of roughly 330 MB, plus about 450 MB for the app and 370 MB for the trainer. The default 8 parallel games used about 3.5 GB in testing. On a small machine, lower **Parallel games** under *Show advanced settings*; 1 game runs inside the trainer process and uses far less memory. |
| **Disk** | About 1 GB free, plus your maps | Dependencies are about 0.6 GB (PyTorch is most of it). Models are small (about 20 MB for several models and their checkpoints). Unpacked `.osz` / `.zip` maps are cached as `.osu` plus audio: 276 mapsets took about 1.4 GB. You can delete the cache at any time. |
| **osu!** | Optional | osu! stable is only needed to watch replays in the real game (see below). Everything else works without it. |

The memory and disk figures were measured on one machine with the default settings, so treat them as a guide.

## Getting started

### 1. Install

You need Python 3.10 or newer. On Windows, install it from [python.org](https://www.python.org/downloads/)
and tick **Add python.exe to PATH** in the installer (Tkinter is included). Then, in a terminal in this
folder:

```powershell
python -m venv .venv                 # create a virtual environment (once)
.venv\Scripts\activate               # activate it (every new terminal)
pip install -r requirements.txt      # install the dependencies (once, about 0.6 GB)
```

On macOS or Linux, activate with `source .venv/bin/activate` instead. If PowerShell refuses to run the
activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
`.venv\Scripts\python app.py` without activating.

### 2. Run the app

```powershell
python app.py
```

The app opens on the **Dashboard**. Everything below happens inside the app; you don't need the
command-line scripts unless you want them.

### 3. Your first model in five minutes

The repository ships with 12 synthetic demo maps in `maps/demo`, so you can train straight away:

1. Click **Create New Model** (on the Dashboard, or **Models → Create New Model** in the sidebar).
2. Give it a **Name**. **Beatmap folder** is already set to `maps/demo`.
3. Set **Training length** to **Short (300k)** and click **Review & Start Training**, then **Start Training**.
4. The **Training** page shows live progress. When it finishes (a few minutes on a laptop CPU), click
   **Test this version**.
5. Pick a map and click **Watch Bot Play** to see it play.

## User guide

Step-by-step instructions for the common tasks. The sections after this one go into more detail.

### Train a new model

1. **Models → Create New Model**.
2. **Model:** enter a **Name** and, optionally, a **Description**.
3. **Training maps:** choose the **Beatmap folder** with **Browse…**, or a single `.osu` / `.osz` / `.zip`
   with **File…** (see [Using real maps](#using-real-maps)). Optionally set a **Star range**, for example
   1 to 2.5 stars, to train only on easier maps. **Validation maps** (default **Auto**) is how many maps are
   held out of training to measure the model fairly.
4. **Training:** pick a **Training length**, from **Quick test (50k)** through **Standard (1M)** to
   **Long (3M)** or **Custom**. The other settings have sensible defaults; **Show advanced settings**
   holds PPO settings, network size and the number of **Parallel games** (lower it if your PC runs out of
   memory).
5. Click **Review & Start Training**, check the summary, and click **Start Training**.

### Watch and control training

The **Training** page shows progress, the time left, recent accuracy and losses.

- **Pause** / **Resume** pauses without losing anything.
- **Save checkpoint now** saves an intermediate copy you can resume from later.
- **Stop & save version** ends training early and keeps what it learned so far as a new version.
- **Cancel (discard)** ends training and throws the session away.

When it finishes, you can **Test this version**, **Generate replay**, **View version history**, or
**Train more**. You can leave the Training page while training runs; the Dashboard shows its status.

### Keep training a model

1. **Models → Continue Training** (or **Continue Training From Here** on a version, or **Train more**
   after a session).
2. Choose where to start: **Current (active) version**, **Best version**, **Latest version**,
   **A specific version:**, or **A checkpoint:**.
3. Set the **Additional training** length. You can change the maps or **Star range** here, for example
   widening the range as the bot improves.
4. Click **Review & Continue Training**, check the summary (it warns if a changed setting could hurt),
   and click **Start Training**. The result is always a **new** version; the one you started from is kept
   unchanged.

### Watch the bot play a map

1. **Test Model → Watch Bot Play**.
2. Choose the model and version.
3. Choose a map: **Choose .osu…** for a single map (an `.osz` / `.zip` lets you pick a difficulty),
   **Browse osu! Songs…** for your osu! library, or **Or browse a folder of maps** to pick from a list with
   star ratings and filters such as **Unseen by this model**.
4. Pick a **Speed** (0.5x to 2x) and click **Watch Bot Play**. The bot plays live in a 4K playfield and
   shows its [rank](#ranks) at the end.
5. Afterwards, click **Save as replay**, or **Save & watch in osu!** to open it in osu! stable.

### Make a replay for osu!

1. **Test Model → Generate Replay**, then choose a model, version and map as above.
2. Optionally set **Attempts (keep best)** to keep the best of several plays, and a **Player name**.
3. Click **Generate Replay**. When it finishes you can **Open in osu!**, **Import map into osu!** (if osu!
   doesn't have that map yet), **Open Replay Folder** or **Copy Replay Path**.

Every replay is listed under **Replays**, where you can open it again, regenerate it or delete it. See
[Watch your model play in the real game](#watch-your-model-play-in-the-real-game) for details.

### Manage models and versions

**Models → Existing Models** lists your models (use **Search** and **Show archived**). Open one to see
its tabs:

- **Version History:** the lineage tree of every version. Select a version and use **Details**,
  **Test**, **Generate Replay**, **Continue Training From Here**, **Set as Active** (the version used by
  default), **Mark as Best**, **Protect**, **Favourite**, **Archive**, **Evaluate…** or **Delete Version**.
- **Checkpoints:** intermediate saves. **Continue Training From Checkpoint**, or **Save as New Version**.
- **Training Sessions:** every session with its settings and results. **Open learning curve** shows the
  accuracy graph for one session.
- **Storage:** what is using disk space, with buttons to delete checkpoints, training logs or evaluation
  results. Versions are never deleted from here.

The buttons above the tabs **Rename**, **Archive** or **Delete Model** (deleting a whole model asks you
to type its name).

### Measure and compare versions

1. **Evaluate…** a version plays every map in a folder once at full length and records its accuracy,
   timing, misses and more. Click **Run evaluation**.
2. To compare two versions, click **Compare Versions** above the model's tabs, or select two versions
   in **Version History** and click **Compare Selected**.
3. Use **Run both on the same maps…** for a fair comparison, then read the **Table** tab, or open
   **Charts** for learning curves and bar charts. **Full screen** gives a larger, zoomable view. See
   [Comparing versions](#comparing-versions).

### Settings

**Settings** lets you see and change where your data is stored (**Change Location…** copies it safely),
**View Storage Usage**, **Clean Temporary Files**, set your osu! stable folder (**Browse…** or
**Detect**), change defaults, import models from the old `runs/` folder, and **Open Logs Folder** if
something goes wrong.

## The app

`python app.py` opens a desktop app (Tkinter, no extra dependencies) that covers the whole workflow:

| Section | What you can do |
|---|---|
| **Dashboard** | Totals, last trained / best model, training status, device and osu! status, quick actions |
| **Models → Create New Model** | Name, maps (a folder, or a single `.osu` / `.osz` / `.zip` via **File…**; 4K maps are validated, everything else is listed and skipped), an optional **star range**, training length, learning rate, batch size, checkpoints, device, seed, advanced PPO settings, then a summary before training starts |
| **Models → Existing Models** | Model Library: rename, archive, delete, open a model |
| **Models → Version History** | Lineage tree of every version: test, generate a replay, continue training from it, set active, mark best, protect, favourite, archive, evaluate, compare (as a table or as [charts](#comparing-versions)), delete |
| **Models → Continue Training** | Resume from the active, best or latest version, any older version, or a checkpoint. Weights, optimiser state, step counter and learning-rate schedule are restored. Change the maps or **star range** for the new session. Warns when a changed setting could hurt |
| **Training** | Live progress, losses, recent accuracy, checkpoints, ETA; pause, save checkpoint now, stop & save, cancel |
| **Test Model** | Pick model, version and a 4K map (a `.osu` file, an `.osz` / `.zip` to choose one of its difficulties, or browse a whole folder filtered by **star range**, with each map tagged trained on / held-out / new), then **Watch Bot Play** (the model plays live in a 4K playfield, then save the play and open it in osu!) or **Generate Replay** (no watching; writes an `.osr`) |
| **Replays** | Replay history with accuracy and rank; open in osu!, show in folder, copy path, generate again, delete. Entries survive deleting the model |
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

### Comparing versions

In the Model Library, **Compare Versions** (or select two versions and click **Compare Selected**) opens a
side-by-side comparison of any two versions, A and B, even from different models. Choose whether to
compare each version's **latest evaluation** or its **validation from training**. If the two results come
from different maps, the dialog warns you; **Run both on the same maps…** evaluates both on one set so the
numbers are directly comparable.

The **Table** tab lists every metric with the better version marked. The **Charts** tab shows:

- **Learning curve:** rolling accuracy (mean of the last 50 plays) against total training timesteps, with
  both versions overlaid. A version made with *Continue Training* includes the sessions of its parent
  versions, so its line picks up where its parent's left off. The legend gives each curve's final accuracy.
- **Bar charts** in three panels, one per unit so each has a single honest axis: **Accuracy (%)** (overall,
  notes hit, holds, each lane), **Mistakes** (misses, extra presses, broken holds) and **Timing (ms)** (mean
  offset, deviation, mean error). Each pair of bars is labelled with the better version and the margin,
  for example "B +0.9" or "A −20".
- **Hover** over a bar for its exact value, or over the learning curve to read both versions' accuracy at
  that point in training.

**Full screen** opens the charts in a maximised, resizable window that labels every bar with its value and
has a toolbar to zoom into a region (useful when both curves flatten near 100%), pan, and save; **Esc**
closes it and **F11** toggles true full screen. **Save as PNG…** exports the charts. A version imported
from the old command line, or whose training logs were deleted (*Delete training logs*), has no learning
curve; the bar charts still work.

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

### Watch your model play in the real game

If you have **osu! stable** installed, you can watch your bot's plays in osu! itself, in the normal
replay viewer, the same way you would watch anyone's replay.

1. Install osu! stable and run it once. The app finds it automatically; if it doesn't, set the osu!
   folder in **Settings**.
2. Go to **Test Model**, choose a model and version, and pick a 4K map (a `.osu`, or an `.osz` / `.zip`
   to choose a difficulty from).
3. Click **Watch Bot Play**, then **Save & watch in osu!** once the play finishes. Or click
   **Generate Replay** (no live view) and then open it in osu!. Saved plays are kept in **Replays**,
   where you can open them in osu! again at any time.
4. If the map isn't in your osu! library yet, the app first gives osu! an `.osz` of that map to import.
   It is built from the map's own files, so the beatmap is byte-identical, which a replay requires.
   Once osu! has imported it, open the replay again.

osu! stable is the only version supported; osu!lazer has not been tested. The app never touches the
running game: it only hands osu! a replay or mapset file, the same as double-clicking one. See
*Watching replays in osu! stable* below for the details and troubleshooting.

## Files

| File | What it does |
|---|---|
| `app.py` | Starts the desktop app |
| `assets/` | The logo (README, window icon and sidebar) in the sizes the app uses |
| `maniabot/storage.py` | User data folder, settings, atomic writes, storage migration |
| `maniabot/models.py` | Model library: versions, lineage, flags, checkpoints, sessions, safe deletion |
| `maniabot/training.py` | Training sessions (new / continue), checkpoints, background worker process |
| `maniabot/evaluation.py` | Inference and evaluation metrics (timing, lanes, holds, misses, extra presses) |
| `maniabot/charts.py` | Comparison charts: learning curves across a version's lineage, metric bar charts, hover tooltips |
| `maniabot/ranks.py` | osu!mania rank (SS / S / A / B / C / D) from accuracy and judgement counts |
| `maniabot/replays.py` | Replay generation and history |
| `maniabot/beatmaps.py` | osu!mania 4K validation with reasons for rejected maps; unpacks `.osz` / `.zip` map archives into a cache |
| `maniabot/osu_integration.py` | Finds osu! stable, opens replays and mapsets |
| `maniabot/legacy.py` | Imports models trained with the old `train.py` |
| `maniabot/sb3_compat.py` | Loads stable-baselines3 model files reliably across torch / Python versions |
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

## Command-line scripts

The app covers everything (see [Getting started](#getting-started)), but the original scripts still work
for scripted or headless runs. They write to `runs/`, which is git-ignored:

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

### Training on a star range

Both **Create New Model** and **Continue Training** have a **Star range** row under the beatmap folder.
Enter a minimum and/or maximum (blank means no limit), and only maps rated inside it, inclusive, are used.
The range applies to the held-out validation maps as well as the training maps, so validation scores from
different sessions stay comparable. The form shows how many maps fall in range, and what range the whole
folder covers.

A typical plan is to start on easy maps and widen the range as the bot improves:

1. Create the model with a range such as **From 1 to 2.5** stars.
2. When validation accuracy levels off, use **Continue Training** with a wider range, for example
   **1 to 4**. The form pre-fills the range the source version was trained on.

Ratings are the real osu!mania star ratings, computed with
[rosu-pp](https://github.com/MaxOhn/rosu-pp-py) (the same algorithm osu! uses; it comes with
`requirements.txt`). If the library isn't installed or can't rate a particular map, a density-based
estimate is used instead and the form says how many ratings are estimates. Ratings are cached by map
checksum next to the unpacked mapsets, so rescanning a big folder stays quick. Each version records the
range it was trained on, and the star rating of every map it used.

### Ranks

When a **Watch Bot Play** run finishes, the playfield shows the rank the bot earned (**SS**, **S**, **A**,
**B**, **C** or **D**) and the status line reads "Finished · Rank S · accuracy …". **Generate Replay** shows
the rank in its result card, and the **Replays** list has a Rank column. Ranks follow osu!mania's accuracy
cut-offs:

| Rank | Accuracy |
|---|---|
| SS | 100% (every note MAX or 300) |
| S | over 95% |
| A | over 90% |
| B | over 80% |
| C | over 70% |
| D | 70% or below |

A play you stop early gets no rank. The rank is the simulator's: osu! re-judges the replay with its own
rules, so when you open the replay in osu! stable, the results screen there is osu!'s own and can differ
slightly (mostly on 50/100 edges and hold-note releases). The replay file carries the simulator's
judgement counts, which osu! shows in the replay's score header.

### Browsing maps in Test Model

Under the map path, **Test Model** has a folder browser. Choose a folder (it accepts the same `.osu` /
`.osz` / `.zip` content as training), optionally set a **star range**, and click a map in the list to
select it. The list shows stars, notes and length, and is sortable by clicking a column header. The
**Show** menu answers "which maps can the bot play?":

| Show | Lists |
|---|---|
| **Playable maps** | Every osu!mania 4K map in the folder |
| **Unseen by this model** | Maps the selected version never trained on (held-out and brand-new): the fair test |
| **Held-out maps** | Maps kept out of training to measure that version |
| **Trained on** | Maps that version trained on |
| **Can't play (not 4K)** | Maps the bot can't play (5K to 14K, other game modes, unreadable files), with the reason |

Each row's **Status** is relative to the version chosen in step 1, and **Result** shows that version's
recorded accuracy on the map when it has one (from its validation or any evaluation you ran). The folder
and star range are remembered between visits.

In **Test Model**, picking an `.osz` or `.zip` with **Choose .osu…** unpacks it and shows a list of its usable 4K
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

## Copyright and usage

Copyright © 2026 Yameen Munir. All rights reserved.

This project is my own personal work. No licence is granted to anyone. Without my prior written
permission, you may not:

- copy, modify, adapt, or create derivative works from any part of the code, assets, or documentation;
- redistribute, publish, sublicense, or sell it, in whole or in part;
- use it, or any part of it, in another project.

Contributions are not accepted: issues, pull requests, and patches will not be reviewed or merged.
Viewing the source here does not grant any rights to it.
