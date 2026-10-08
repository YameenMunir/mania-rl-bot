"""ManiaBot desktop app: create, train, version, test and replay osu!mania 4K bots.

Usage: python app.py

Models and all other runtime data are stored in your user data folder
(%LOCALAPPDATA%\\OsuManiaAI on Windows), never in this repository.
"""
from __future__ import annotations

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from maniabot.ui.app import main

    main()
