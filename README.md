# osu!mania 4K RL bot

A small project that trains a neural network (PPO) to play osu!mania 4K and get better with every play.
It runs against a fast simulator of osu!mania scoring instead of the real game, so thousands of plays
take minutes.

## Files

| File | What it does |
|---|---|
| `osu_parser.py` | Parses `.osu` files (note times, lanes, holds, OD) |
| `mania_env.py` | Gymnasium env: 4 keys, real hit windows (MAX/300/200/100/50/miss), hold notes |
| `make_demo_maps.py` | Makes synthetic 4K maps so you can train immediately |
| `train.py` | PPO training, logs every play to `episodes.csv`, plots a learning curve |
| `evaluate.py` | Scores a trained model (or a scripted oracle baseline) on full maps |
| `export_replay.py` | Plays maps with a trained model and writes osu! stable `.osr` replay files |
| `osr_writer.py` | The `.osr` file writer (header, LZMA frames, beatmap MD5) |

## Quick start

```bash
pip install -r requirements.txt
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

Download 4K osu!mania beatmaps from osu.ppy.sh, unzip the `.osz` files (they are zip archives) into a
folder like `maps/real/`, then pass `--maps maps/real`. Non-4K and non-mania files are skipped automatically.

## Watching replays in osu! stable

A replay is tied to one exact beatmap by the MD5 of its `.osu` file, so the map has to be in your osu!
library first.

**Demo maps (quickest test):**

1. Drag every file in `maps/osz/` onto the osu! window (or double-click them) to import the 12 demo mapsets.
   Each has a click-track on the beat as its audio.
2. `python export_replay.py --run runs/smoke2 --maps maps/demo --out replays`
3. Double-click a `.osr` in `replays/` (or drag it onto osu!) to watch it.

Two sample replays are already in `replays/` for Demo 005 and Demo 010.

**Real maps:**

1. Download the map in osu! so it sits in your Songs folder.
2. Point `--maps` at your Songs folder, for example `C:\Users\you\AppData\Local\osu!\Songs`.
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
