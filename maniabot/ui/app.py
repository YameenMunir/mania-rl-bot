"""Main window: navigation, shared services and the training-job pump."""
from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from maniabot import osu_integration
from maniabot.logs import event, setup_logging
from maniabot.models import ModelStore
from maniabot.replays import ReplayHistory
from maniabot.storage import Settings, Storage
from maniabot.training import TrainingJob, TrainingSpec, device_status
from maniabot.ui import theme
from maniabot.ui.widgets import Tasks, ask, error

log = logging.getLogger("maniabot.ui")


class AppContext:
    """Services shared by every page."""

    def __init__(self, app: "App"):
        self.app = app
        self.tasks = Tasks(app)
        self.job: TrainingJob | None = None
        self.device: dict[str, Any] | None = None
        self.osu: dict[str, Any] | None = None
        self._listeners: dict[str, list[Callable[..., None]]] = {}
        self.load()

    def load(self) -> None:
        self.settings = Settings.load()
        self.storage = Storage.from_settings(self.settings).ensure()
        setup_logging(self.storage.logs_dir)
        self.store = ModelStore(self.storage)
        self.history = ReplayHistory(self.storage)

    # simple pub/sub so pages refresh after changes made elsewhere
    def on(self, name: str, fn: Callable[..., None]) -> None:
        self._listeners.setdefault(name, []).append(fn)

    def emit(self, name: str, **kw) -> None:
        for fn in self._listeners.get(name, []):
            try:
                fn(**kw)
            except Exception:  # noqa: BLE001
                log.exception("listener for %s failed", name)

    def refresh_device(self) -> None:
        self.tasks.run(device_status, lambda d: (setattr(self, "device", d), self.emit("device")), name="device-check")

    def refresh_osu(self) -> None:
        self.tasks.run(lambda: osu_integration.status(self.settings["osu_path"]),
                       lambda s: (setattr(self, "osu", s), self.emit("osu")), name="osu-check")

    @property
    def training_running(self) -> bool:
        return self.job is not None and not self.job.finished

    def start_training(self, spec: TrainingSpec) -> TrainingJob:
        if self.training_running:
            raise RuntimeError("A training session is already running. Stop it first.")
        self.job = TrainingJob(self.store, spec).start()
        self.emit("training_started", job=self.job)
        return self.job


NAV: list[tuple[str, str, list[tuple[str, str, dict]]]] = [
    ("dashboard", "Dashboard", []),
    ("library", "Models", [
        ("create", "Create New Model", {}),
        ("library", "Existing Models", {}),
        ("history", "Version History", {}),
        ("continue", "Continue Training", {}),
    ]),
    ("training", "Training", []),
    ("test", "Test Model", [
        ("test", "Watch Bot Play", {"mode": "watch"}),
        ("test", "Generate Replay", {"mode": "replay"}),
    ]),
    ("replays", "Replays", []),
    ("settings", "Settings", []),
]


class App(tk.Tk):
    def __init__(self):
        theme.enable_dpi_awareness()
        super().__init__()
        self.title("ManiaBot — osu!mania 4K AI trainer")
        k = theme.ui_scale(self)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(int(1280 * k), sw - 40)}x{min(int(820 * k), sh - 80)}")
        self.minsize(int(1000 * k), int(640 * k))
        theme.apply(self)
        self.ctx = AppContext(self)
        event(log, "app.started", storage=self.ctx.storage.root)
        self._startup_maintenance()

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.sidebar = ttk.Frame(self, style="Sidebar.TFrame", width=int(220 * k))
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.grid_propagate(False)
        self.content = ttk.Frame(self)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)
        self._nav_buttons: dict[str, ttk.Button] = {}
        self._build_sidebar()
        self.pages: dict[str, ttk.Frame] = {}
        self.current: str | None = None

        self.ctx.refresh_device()
        self.ctx.refresh_osu()
        self.after(300, self._pump_job)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.show("dashboard")

    def _startup_maintenance(self) -> None:
        try:
            n = self.ctx.store.cleanup_temp()
            self.ctx.storage.clear_tmp()
            m = self.ctx.store.mark_interrupted_sessions()
            if n or m:
                event(log, "startup.cleanup", temp_removed=n, sessions_interrupted=m)
        except Exception:  # noqa: BLE001
            log.exception("startup maintenance failed")

    # ---------------------------------------------------------- sidebar
    def _build_sidebar(self) -> None:
        ttk.Label(self.sidebar, text="ManiaBot", style="Brand.TLabel").pack(anchor="w", padx=18, pady=(20, 0))
        ttk.Label(self.sidebar, text="osu!mania 4K AI trainer", style="SidebarMuted.TLabel").pack(anchor="w", padx=18,
                                                                                                 pady=(0, 18))
        for key, label, subs in NAV:
            b = ttk.Button(self.sidebar, text=label, style="Nav.TButton", command=lambda k=key: self.show(k))
            b.pack(fill="x")
            self._nav_buttons[key] = b
            for skey, slabel, kw in subs:
                ttk.Button(self.sidebar, text=slabel, style="SubNav.TButton",
                           command=lambda k=skey, a=kw: self.show(k, **a)).pack(fill="x")
        self.side_status = ttk.Label(self.sidebar, text="", style="SidebarMuted.TLabel", wraplength=190, justify="left")
        self.side_status.pack(side="bottom", anchor="w", padx=18, pady=16)

    def _page_class(self, name: str):
        from maniabot.ui import pages

        return {
            "dashboard": pages.DashboardPage,
            "create": pages.CreateModelPage,
            "continue": pages.ContinueTrainingPage,
            "library": pages.LibraryPage,
            "history": pages.LibraryPage,
            "training": pages.TrainingPage,
            "test": pages.TestModelPage,
            "replays": pages.ReplaysPage,
            "settings": pages.SettingsPage,
        }[name]

    def show(self, name: str, **kwargs) -> None:
        key = "library" if name == "history" else name
        if name == "history":
            kwargs.setdefault("tab", "versions")
            kwargs.setdefault("pick", True)
        page = self.pages.get(key)
        if page is None:
            page = self._page_class(name)(self.content, self.ctx)
            self.pages[key] = page
        # Hidden pages are un-gridded (not just stacked), so their buttons can't take keyboard focus.
        for other in self.pages.values():
            if other is not page:
                other.grid_remove()
        page.grid(row=0, column=0, sticky="nsew")
        page.focus_set()
        self.current = key
        nav_key = {"create": "library", "continue": "library", "history": "library"}.get(name, key)
        for k, b in self._nav_buttons.items():
            b.configure(style="NavActive.TButton" if k == nav_key else "Nav.TButton")
        try:
            page.on_show(**kwargs)
        except Exception as exc:  # noqa: BLE001
            log.exception("showing page %s", name)
            error(self, str(exc))

    # ------------------------------------------------------ training pump
    def _pump_job(self) -> None:
        job = self.ctx.job
        if job is not None and job.result is None:
            try:
                events = job.poll()
            except Exception:  # noqa: BLE001
                log.exception("polling training job")
                events = []
            for e in events:
                self.ctx.emit("training_event", event=e)
                if e["type"] == "done":
                    self._training_finished(e)
        self._update_side_status()
        self.after(300, self._pump_job)

    def _update_side_status(self) -> None:
        job = self.ctx.job
        if job is not None and not job.finished:
            p = job.last_progress
            pct = f" {p['timesteps'] / max(1, p['target']) * 100:.0f}%" if p else ""
            self.side_status.configure(text=f"● Training {job.spec.model_id}{pct}\n{job.message}", foreground=theme.ACCENT)
        else:
            self.side_status.configure(text=f"Data: {self.ctx.storage.root}", foreground=theme.FAINT)

    def _training_finished(self, e: dict[str, Any]) -> None:
        self.ctx.emit("models_changed")
        status = e.get("status")
        if status in ("completed", "stopped") and e.get("version"):
            self.bell()
        event(log, "ui.training_finished", status=status, version=e.get("version"))

    # ----------------------------------------------------------- closing
    def _on_close(self) -> None:
        if self.ctx.training_running:
            if not ask(self, "Training is running",
                       "Closing stops training. ManiaBot will stop safely and save what it has learned as a new "
                       "version (earlier versions are never touched).\n\nStop training and close?",
                       ok="Stop and close", danger=True):
                return
            self.ctx.job.stop()
            self._closing_wait(0)
            return
        self.destroy()

    def _closing_wait(self, waited: int) -> None:
        job = self.ctx.job
        job.poll()
        if job.finished or waited > 600:  # up to ~3 minutes for the final save + validation
            if not job.finished:
                job.kill()
            self.destroy()
            return
        self.side_status.configure(text="Stopping training and saving…", foreground=theme.WARN)
        self.after(300, lambda: self._closing_wait(waited + 1))


def main() -> None:
    import multiprocessing

    multiprocessing.freeze_support()
    app = App()
    app.mainloop()
