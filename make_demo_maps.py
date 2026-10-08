"""Generate synthetic 4K osu!mania maps so you can train before downloading real ones.

Each map is also packaged as an .osz mapset (the .osu plus a click-track audio file), so you
can import it into osu! stable and watch replays of it. Double-click an .osz or drag it onto
the osu! window. The .osu inside is byte-identical to the one in --out, so replay MD5s match.

Usage: python make_demo_maps.py [--count 12] [--out maps/demo] [--osz-dir maps/osz] [--seed 0]
"""
from __future__ import annotations

import argparse
import io
import wave
import zipfile
from pathlib import Path

import numpy as np

LANE_X = [64, 192, 320, 448]
OFFSET_MS = 2000
RATE = 22050

HEADER = """osu file format v14

[General]
AudioFilename: demo_{idx:03d}.wav
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Normal
Mode: 3

[Metadata]
Title:Demo {idx:03d}
TitleUnicode:Demo {idx:03d}
Artist:synthetic
ArtistUnicode:synthetic
Creator:ManiaBot
Version:Demo 4K d={difficulty:.2f}
Source:
Tags:
BeatmapID:0
BeatmapSetID:-1

[Difficulty]
HP:5
CircleSize:4
OverallDifficulty:{od:.1f}
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[Events]
//Background and Video events

[TimingPoints]
{offset},{beat:.3f},4,1,0,100,1,0

[HitObjects]
"""


def make_map(rng: np.random.Generator, idx: int, difficulty: float) -> tuple[str, float, float]:
    """Returns (.osu text, beat length in ms, end time of the last note in ms)."""
    bpm = rng.uniform(110, 190)
    beat = 60000.0 / bpm
    step = beat / 2 if difficulty < 0.4 else beat / 4
    density = 0.30 + 0.5 * difficulty
    duration_ms = rng.uniform(45, 75) * 1000.0
    od = 6.0 + 3.0 * difficulty

    t = float(OFFSET_MS)
    busy = [-1e9] * 4
    last_lane = -1
    rows: list[tuple[int, int, int]] = []
    while t < duration_ms:
        if rng.random() < density:
            free = [lane for lane in range(4) if t > busy[lane] + 40]
            if free:
                n = 2 if (len(free) >= 2 and rng.random() < 0.05 + 0.15 * difficulty) else 1
                if n == 1 and len(free) > 1 and last_lane in free and rng.random() < 0.7:
                    free.remove(last_lane)
                lanes = [int(x) for x in rng.choice(free, size=n, replace=False)]
                for lane in lanes:
                    if rng.random() < 0.12:
                        end = t + beat * float(rng.choice([1.0, 1.5, 2.0]))
                        busy[lane] = end
                        rows.append((lane, round(t), round(end)))
                    else:
                        busy[lane] = t
                        rows.append((lane, round(t), round(t)))
                last_lane = lanes[-1]
        t += step

    out = [HEADER.format(idx=idx, difficulty=difficulty, od=od, beat=beat, offset=OFFSET_MS)]
    last_ms = 0
    for lane, start, end in sorted(rows, key=lambda r: (r[1], r[0])):
        last_ms = max(last_ms, end)
        if end > start:
            out.append(f"{LANE_X[lane]},192,{start},128,0,{end}:0:0:0:0:\n")
        else:
            out.append(f"{LANE_X[lane]},192,{start},1,0,0:0:0:0:\n")
    return "".join(out), beat, float(last_ms)


def click_track_wav(beat_ms: float, length_ms: float) -> bytes:
    """Mono 16-bit WAV with a short click on every beat."""
    n = int(RATE * length_ms / 1000.0)
    audio = np.zeros(n, dtype=np.float32)
    clen = int(RATE * 0.03)
    tt = np.arange(clen) / RATE
    click = (np.sin(2 * np.pi * 1000 * tt) * np.exp(-tt * 150)).astype(np.float32)
    k = 0
    while True:
        start = int((OFFSET_MS + k * beat_ms) * RATE / 1000.0)
        if start >= n:
            break
        end = min(n, start + clen)
        audio[start:end] += 0.5 * click[: end - start]
        k += 1
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--out", default="maps/demo")
    ap.add_argument("--osz-dir", default="maps/osz", help="where the importable .osz mapsets go ('' to skip)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    osz_dir = Path(args.osz_dir) if args.osz_dir else None
    if osz_dir:
        osz_dir.mkdir(parents=True, exist_ok=True)

    for i, d in enumerate(np.linspace(0.1, 0.9, args.count)):
        text, beat, last_ms = make_map(rng, i, float(d))
        osu_bytes = text.encode("utf-8")  # bytes, so Windows never rewrites newlines (keeps MD5 stable)
        (out / f"demo_{i:03d}.osu").write_bytes(osu_bytes)
        if osz_dir:
            wav = click_track_wav(beat, last_ms + 4000.0)
            with zipfile.ZipFile(osz_dir / f"demo_{i:03d}.osz", "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr(f"demo_{i:03d}.osu", osu_bytes)
                z.writestr(f"demo_{i:03d}.wav", wav)
    msg = f"wrote {args.count} maps to {out}"
    if osz_dir:
        msg += f" and importable mapsets to {osz_dir}"
    print(msg)


if __name__ == "__main__":
    main()
