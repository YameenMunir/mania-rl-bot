"""Robust loading of stable-baselines3 model zips.

SB3 passes each .pth member to torch.load as a streaming zipfile.ZipExtFile. With some
torch / Python combinations (seen with torch 2.14 on Python 3.12.0) torch's zip reader
misreads such streams for certain files ("PytorchStreamReader failed reading file
.data/serialization_id"), even though the file is intact. Buffering the member into memory
first avoids that. `load_ppo` applies the buffering only for the duration of the load.
"""
from __future__ import annotations

import io
import os
import threading
from contextlib import contextmanager

_lock = threading.Lock()


@contextmanager
def _buffered_torch_load():
    import torch

    original = torch.load

    def load(f, *args, **kwargs):
        if not isinstance(f, (str, bytes, os.PathLike, io.BytesIO)) and hasattr(f, "read"):
            f = io.BytesIO(f.read())
        return original(f, *args, **kwargs)

    with _lock:
        torch.load = load
        try:
            yield
        finally:
            torch.load = original


def load_ppo(path, **kwargs):
    from stable_baselines3 import PPO

    with _buffered_torch_load():
        return PPO.load(path, **kwargs)
