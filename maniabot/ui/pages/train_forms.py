"""Create New Model and Continue Training workflows."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from maniabot.beatmaps import ScanResult, default_val_count, star_range_label
from maniabot.models import ModelError, slugify
from maniabot.mods import mod_list_label
from maniabot.storage import dir_size, human_size
from maniabot.training import (
    DEFAULT_NET_ARCH, TrainingSpec, compatibility_warnings, default_n_envs, validate_spec,
)
from maniabot.ui.pages.base import (
    DEVICE_CHOICES, DEVICE_HELP, LENGTH_HELP, LENGTH_PRESETS, MODS_HELP, MapsFolderField, ModsField, Page, StarRangeField,
    form_row, int_var, parse_float, parse_int, summary_dialog, version_label,
)
from maniabot.ui.widgets import Card, EmptyState, KeyValueGrid, Table, error, fmt_date, fmt_int, fmt_pct


STAR_HELP = ("Only use maps whose osu! star rating is in this range (leave blank for no limit). It applies to "
             "validation maps too, so scores stay comparable between sessions. Handy for training on easy maps "
             "first, then widening the range.")


class LengthField(ttk.Frame):
    def __init__(self, parent, default: int):
        super().__init__(parent, style="Card.TFrame")
        self.preset = tk.StringVar()
        self.custom = int_var(default)
        names = list(LENGTH_PRESETS) + ["Custom"]
        match = next((k for k, v in LENGTH_PRESETS.items() if v == default), "Custom")
        self.preset.set(match)
        cb = ttk.Combobox(self, textvariable=self.preset, values=names, state="readonly", width=20)
        cb.grid(row=0, column=0)
        self.entry = ttk.Entry(self, textvariable=self.custom, width=14)
        self.entry.grid(row=0, column=1, padx=(8, 0))
        ttk.Label(self, text="timesteps", style="CardMuted.TLabel").grid(row=0, column=2, padx=(6, 0))
        cb.bind("<<ComboboxSelected>>", lambda _e: self._sync())
        self._sync()

    def _sync(self) -> None:
        v = LENGTH_PRESETS.get(self.preset.get())
        if v:
            self.custom.set(str(v))
            self.entry.state(["disabled"])
        else:
            self.entry.state(["!disabled"])

    def get(self) -> int:
        return parse_int(self.custom, "Training length")


def _device_summary(ctx, device: str) -> str:
    d = ctx.device or {}
    if device == "cuda":
        return f"CUDA GPU ({d.get('name') or 'not detected!'})"
    if device == "auto":
        return f"Auto → {'CUDA GPU (' + d['name'] + ')' if d.get('cuda') else 'CPU'}"
    return "CPU"


class AdvancedSection(ttk.Frame):
    """Settings most people never need; collapsed by default."""

    def __init__(self, parent, new_model: bool):
        super().__init__(parent, style="Card.TFrame")
        self.columnconfigure(1, weight=1)
        self.open = tk.BooleanVar(value=False)
        self.toggle = ttk.Checkbutton(self, text="Show advanced settings", variable=self.open, style="Card.TCheckbutton",
                                      command=self._render)
        self.toggle.grid(row=0, column=0, columnspan=3, sticky="w")
        self.inner = ttk.Frame(self, style="Card.TFrame")
        self.inner.columnconfigure(1, weight=1)
        self.net = tk.StringVar(value=",".join(map(str, DEFAULT_NET_ARCH)))
        self.n_steps = int_var(1024)
        self.n_epochs = int_var(6)
        self.n_envs = int_var(default_n_envs())
        self.dt = tk.StringVar(value="10")
        self.jitter = tk.StringVar(value="0")
        self.clip = tk.StringVar(value="20")
        r = 0
        if new_model:
            form_row(self.inner, r, "Network size", ttk.Entry(self.inner, textvariable=self.net, width=16),
                     "Hidden layer widths of the policy network, e.g. 128,128. Fixed once the model exists.")
            r += 1
        form_row(self.inner, r, "Parallel games", ttk.Spinbox(self.inner, from_=1, to=64, textvariable=self.n_envs, width=8),
                 "How many simulators run at once (uses that many CPU processes).")
        form_row(self.inner, r + 1, "Rollout length", ttk.Entry(self.inner, textvariable=self.n_steps, width=10),
                 "Steps each game collects before every PPO update.")
        form_row(self.inner, r + 2, "PPO passes per update", ttk.Spinbox(self.inner, from_=1, to=50, textvariable=self.n_epochs, width=8),
                 "Optimisation passes over each batch of experience (PPO 'n_epochs').")
        form_row(self.inner, r + 3, "Decision tick (ms)", ttk.Entry(self.inner, textvariable=self.dt, width=10),
                 "How often the bot decides which keys to hold. Smaller is more precise but slower.")
        form_row(self.inner, r + 4, "Timing noise (ms)", ttk.Entry(self.inner, textvariable=self.jitter, width=10),
                 "Gaussian noise added to every press/release, to train a human-like bot. 0 = perfect timing.")
        form_row(self.inner, r + 5, "Practice clip (s)", ttk.Entry(self.inner, textvariable=self.clip, width=10),
                 "Training plays random clips of this length from each map.")

    def _render(self) -> None:
        if self.open.get():
            self.inner.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        else:
            self.inner.grid_forget()

    def apply(self, spec: TrainingSpec) -> None:
        if spec.kind == "new":
            try:
                spec.net_arch = [int(x) for x in self.net.get().replace(" ", "").split(",") if x]
            except ValueError:
                raise ModelError("Network size must look like 128,128.") from None
        spec.n_envs = parse_int(self.n_envs, "Parallel games")
        spec.n_steps = parse_int(self.n_steps, "Rollout length")
        spec.n_epochs = parse_int(self.n_epochs, "PPO passes")
        spec.dt_ms = parse_float(self.dt, "Decision tick")
        spec.jitter_ms = parse_float(self.jitter, "Timing noise")
        spec.clip_seconds = parse_float(self.clip, "Practice clip")


def _running_guard(ctx, parent) -> bool:
    if ctx.training_running:
        error(parent, "A training session is already running. Stop it (or wait for it to finish) first.",
              "Training already running")
        return True
    return False


# ======================================================================== new
class CreateModelPage(Page):
    title = "Create New Model"
    subtitle = "Train a brand-new osu!mania 4K bot from scratch"

    def build(self) -> None:
        s = self.ctx.settings
        c1 = Card(self.body, "Model")
        c1.grid(row=0, column=0, sticky="ew")
        b = c1.body
        b.columnconfigure(1, weight=1)
        self.name = tk.StringVar()
        self.desc = tk.StringVar()
        form_row(b, 0, "Name", ttk.Entry(b, textvariable=self.name, width=40))
        form_row(b, 1, "Description", ttk.Entry(b, textvariable=self.desc, width=60), "Optional notes for yourself.")
        ttk.Label(b, text="Mode: osu!mania 4K (4 lanes)", style="CardMuted.TLabel").grid(row=2, column=1, sticky="w")

        c2 = Card(self.body, "Training maps", "Any folder of .osu files, .osz mapsets or .zip map packs (searched recursively). Non-4K maps are skipped "
                                               "and listed, never converted.")
        c2.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        b = c2.body
        b.columnconfigure(1, weight=1)
        self.maps = MapsFolderField(b, self.ctx, s["default_maps"], on_scan=self._scanned)
        form_row(b, 0, "Beatmap folder", self.maps)
        self.stars = StarRangeField(b, on_change=self._update_split)
        form_row(b, 1, "Star range", self.stars, STAR_HELP)
        self.val = tk.StringVar(value="Auto")
        self.val_cb = ttk.Combobox(b, textvariable=self.val, values=["Auto"], state="readonly", width=10)
        form_row(b, 2, "Validation maps", self.val_cb,
                 "Maps held out of training and used only to measure the model (and pick the Best version). "
                 "With very few maps, validation uses training maps instead.")
        self.split_lbl = ttk.Label(b, text="", style="CardMuted.TLabel")
        self.split_lbl.grid(row=3, column=1, sticky="w")
        self.val_cb.bind("<<ComboboxSelected>>", lambda _e: self._update_split())

        c3 = Card(self.body, "Training")
        c3.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        b = c3.body
        b.columnconfigure(1, weight=1)
        self.length = LengthField(b, 300_000)
        form_row(b, 0, "Training length", self.length, LENGTH_HELP, stretch=False)
        self.mods = ModsField(b)
        form_row(b, 1, "Mods", self.mods, MODS_HELP)
        self.lr = tk.StringVar(value="0.0003")
        form_row(b, 2, "Learning rate", ttk.Entry(b, textvariable=self.lr, width=12),
                 "How big each learning step is. 0.0003 is a good default.")
        self.batch = int_var(1024)
        form_row(b, 3, "Batch size", ttk.Entry(b, textvariable=self.batch, width=12),
                 "Experience samples per gradient step.")
        self.ckpt = int_var(100_000)
        form_row(b, 4, "Checkpoint every", ttk.Entry(b, textvariable=self.ckpt, width=12),
                 "Save (and validate) a checkpoint every N timesteps, so you can resume from it later.")
        self.device = tk.StringVar(value=s["default_device"])
        form_row(b, 5, "Device", ttk.Combobox(b, textvariable=self.device, values=DEVICE_CHOICES, state="readonly", width=10),
                 DEVICE_HELP)
        self.seed = int_var(0)
        form_row(b, 6, "Random seed", ttk.Entry(b, textvariable=self.seed, width=12),
                 "Same seed + same settings = reproducible run.")
        self.adv = AdvancedSection(b, new_model=True)
        self.adv.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        row = ttk.Frame(self.body)
        row.grid(row=3, column=0, sticky="e", pady=(16, 0))
        self.start_btn = ttk.Button(row, text="Review & Start Training", style="Accent.TButton", command=self.submit)
        self.start_btn.pack()
        self.scan: ScanResult | None = None

    def on_show(self, **kwargs) -> None:
        if self.maps.result is None:
            self.maps.scan()

    def _scanned(self, res: ScanResult | None) -> None:
        self.scan = res
        self.stars.show_scan(res)
        self._update_split()

    def _n_maps(self) -> int:
        return len(self.stars.in_range())

    def _n_val(self) -> int:
        n = self._n_maps()
        return default_val_count(n) if self.val.get() == "Auto" else int(self.val.get())

    def _update_split(self) -> None:
        n = self._n_maps()
        values = ["Auto"] + [str(i) for i in range(0, max(0, n - 1) + 1)]
        self.val_cb.configure(values=values)
        if self.val.get() not in values:
            self.val.set("Auto")
        if not self.scan or not n:
            self.split_lbl.configure(text="")
            return
        v = self._n_val()
        if v == 0 or v >= n:
            self.split_lbl.configure(text=f"{n} training maps; validation will use training maps (none held out)")
        else:
            self.split_lbl.configure(text=f"{n - v} training maps · {v} held-out validation maps")

    def submit(self) -> None:
        if _running_guard(self.ctx, self):
            return
        try:
            name = self.name.get().strip()
            if not name:
                raise ModelError("Give the model a name.")
            if self.ctx.store.name_taken(name):
                raise ModelError(f"A model called '{name}' already exists. Pick another name, or use Continue Training.")
            if not self.scan or not self.scan.valid:
                raise ModelError("Choose a beatmap folder with at least one osu!mania 4K map (and click Check maps).")
            self.stars.get()  # raises a readable error for a bad range
            if not self._n_maps():
                raise ModelError("No maps fall inside that star range. Widen the range or clear it.")
            spec = TrainingSpec(
                model_id="(new)", kind="new", maps_folder=self.maps.get(), timesteps=self.length.get(),
                learning_rate=parse_float(self.lr, "Learning rate"), batch_size=parse_int(self.batch, "Batch size"),
                checkpoint_every=parse_int(self.ckpt, "Checkpoint frequency"), device=self.device.get(),
                seed=parse_int(self.seed, "Random seed"), val_count=-1 if self.val.get() == "Auto" else int(self.val.get()),
                notes="Initial training",
            )
            self.adv.apply(spec)
            self.stars.apply(spec)
            self.mods.apply(spec)
            errs = validate_spec(spec)
            if errs:
                raise ModelError("\n".join(errs))
        except ModelError as exc:
            error(self, str(exc), "Check the settings")
            return
        n = self._n_maps()
        v = self._n_val()
        heldout = 0 < v < n
        rows = [
            ("Model", name),
            ("Mode", "osu!mania 4K"),
            ("Training maps", f"{n - v if heldout else n} from {spec.maps_folder}"),
            ("Star range", star_range_label(spec.min_stars, spec.max_stars)),
            ("Mods", mod_list_label(spec.mod_bits)),
            ("Validation maps", f"{v} held out" if heldout else f"{min(4, n)} training maps (not enough maps to hold any out)"),
            ("Device", _device_summary(self.ctx, spec.device)),
            ("Training length", f"{spec.timesteps:,} timesteps"),
            ("Batch size", f"{spec.batch_size:,}"),
            ("Learning rate", f"{spec.learning_rate:g}"),
            ("Checkpoints", f"every {spec.checkpoint_every:,} timesteps"),
            ("Network", "x".join(map(str, spec.net_arch)) + f"  ·  {spec.n_envs} parallel games"),
            ("Will be saved as", "v001"),
            ("Storage location", str(self.ctx.store.root / slugify(name))),
        ]
        if not summary_dialog(self, "Start training?", rows):
            return
        try:
            rec = self.ctx.store.create_model(name, self.desc.get(), {
                "algorithm": "PPO", "policy": "MlpPolicy", "net_arch": spec.net_arch, "n_upcoming": 4,
                "lookahead_ms": 1000.0}, spec.env_config)
            spec.model_id = rec.id
            self.ctx.start_training(spec)
        except Exception as exc:  # noqa: BLE001
            error(self, str(exc), "Couldn't start training")
            return
        self.ctx.emit("models_changed")
        self.name.set("")
        self.desc.set("")
        self.ctx.app.show("training")


# =================================================================== continue
class ContinueTrainingPage(Page):
    title = "Continue Training"
    subtitle = "Train an existing model further. A new version is always created; nothing is overwritten."

    def build(self) -> None:
        self.body.columnconfigure(0, weight=1)
        top = Card(self.body, "1. Choose a model")
        top.grid(row=0, column=0, sticky="ew")
        self.table = Table(top.body, [
            ("name", "Model", 200), ("active", "Active", 70), ("best", "Best", 70), ("versions", "Versions", 70),
            ("steps", "Total steps", 100), ("acc", "Val. accuracy", 100), ("last", "Last trained", 140),
            ("size", "Size", 80)], height=6)
        self.table.grid(row=0, column=0, sticky="ew")
        self.table.on_select(self._on_table_select)
        self.empty = EmptyState(top.body, "No models yet", "Create a model first; then you can keep training it here.",
                                [("Create New Model", lambda: self.ctx.app.show("create"))])

        self.detail = Card(self.body, "2. Start from")
        self.detail.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        b = self.detail.body
        b.columnconfigure(0, weight=1)
        self.info = KeyValueGrid(b, columns=3)
        self.info.grid(row=0, column=0, sticky="ew")
        for k, label in (("created", "Created"), ("last", "Last trained"), ("active", "Current version"),
                         ("best", "Best version"), ("versions", "Versions"), ("steps", "Total timesteps"),
                         ("sessions", "Training sessions"), ("val", "Validation (active)"), ("arch", "Architecture"),
                         ("dataset", "Dataset"), ("ckpt", "Latest checkpoint"), ("size", "Storage size")):
            self.info.add(k, label)
        src = ttk.Frame(b, style="Card.TFrame")
        src.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        src.columnconfigure(2, weight=1)
        self.source = tk.StringVar(value="active")
        self.version_var = tk.StringVar()
        self.ckpt_var = tk.StringVar()
        for i, (val, label) in enumerate((("active", "Current (active) version"), ("best", "Best version"),
                                          ("latest", "Latest version"), ("version", "A specific version:"),
                                          ("checkpoint", "A checkpoint:"))):
            ttk.Radiobutton(src, text=label, value=val, variable=self.source, command=self._source_changed).grid(
                row=i, column=0, sticky="w", pady=2)
        self.src_lbls = {k: ttk.Label(src, text="", style="CardMuted.TLabel") for k in ("active", "best", "latest")}
        for i, k in enumerate(("active", "best", "latest")):
            self.src_lbls[k].grid(row=i, column=1, sticky="w", padx=10)
        self.version_cb = ttk.Combobox(src, textvariable=self.version_var, state="readonly", width=44)
        self.version_cb.grid(row=3, column=1, columnspan=2, sticky="w", padx=10)
        self.ckpt_cb = ttk.Combobox(src, textvariable=self.ckpt_var, state="readonly", width=70)
        self.ckpt_cb.grid(row=4, column=1, columnspan=2, sticky="w", padx=10)
        self.version_cb.bind("<<ComboboxSelected>>", lambda _e: (self.source.set("version"), self._source_changed()))
        self.ckpt_cb.bind("<<ComboboxSelected>>", lambda _e: (self.source.set("checkpoint"), self._source_changed()))

        opts = Card(self.body, "3. Settings for this session")
        opts.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        b = opts.body
        b.columnconfigure(1, weight=1)
        self.length = LengthField(b, 300_000)
        form_row(b, 0, "Additional training", self.length, LENGTH_HELP, stretch=False)
        self.lr = tk.StringVar(value="0.0003")
        form_row(b, 1, "Learning rate", ttk.Entry(b, textvariable=self.lr, width=12),
                 "Lowering it (e.g. halving) often helps when fine-tuning an already good model.")
        self.maps = MapsFolderField(b, self.ctx, self.ctx.settings["default_maps"],
                                    on_scan=lambda res: self.stars.show_scan(res))
        form_row(b, 2, "Beatmap folder", self.maps)
        self.stars = StarRangeField(b)
        form_row(b, 3, "Star range", self.stars, STAR_HELP)
        self.mods = ModsField(b, on_change=lambda: self._update_warnings())
        form_row(b, 4, "Mods", self.mods, MODS_HELP)
        self.batch = int_var(1024)
        form_row(b, 5, "Batch size", ttk.Entry(b, textvariable=self.batch, width=12))
        self.ckpt_every = int_var(100_000)
        form_row(b, 6, "Checkpoint every", ttk.Entry(b, textvariable=self.ckpt_every, width=12))
        self.device = tk.StringVar(value=self.ctx.settings["default_device"])
        form_row(b, 7, "Device", ttk.Combobox(b, textvariable=self.device, values=DEVICE_CHOICES, state="readonly",
                                              width=10), DEVICE_HELP)
        self.notes = tk.StringVar()
        form_row(b, 8, "Notes for the new version", ttk.Entry(b, textvariable=self.notes, width=60))
        self.set_active = tk.BooleanVar(value=True)
        ttk.Checkbutton(b, text="Make the new version the active version when training finishes",
                        variable=self.set_active, style="Card.TCheckbutton").grid(row=9, column=1, sticky="w", pady=4)
        self.adv = AdvancedSection(b, new_model=False)
        self.adv.grid(row=10, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.warn_lbl = ttk.Label(b, text="", style="CardWarn.TLabel", wraplength=760, justify="left")
        self.warn_lbl.grid(row=11, column=0, columnspan=4, sticky="w", pady=(10, 0))
        for v in (self.lr, self.batch, self.adv.dt, self.adv.jitter, self.adv.n_envs, self.adv.n_steps, self.maps.var,
                  self.stars.lo, self.stars.hi):
            v.trace_add("write", lambda *_: self._update_warnings())

        row = ttk.Frame(self.body)
        row.grid(row=3, column=0, sticky="e", pady=(16, 0))
        ttk.Button(row, text="Review & Continue Training", style="Accent.TButton", command=self.submit).pack()

        self.model_id: str | None = None
        self.versions = []
        self.ckpts = []
        self.ctx.on("models_changed", lambda **_: self.refresh() if self.visible() else None)

    def on_show(self, model_id: str | None = None, version: str | None = None, checkpoint: str | None = None, **_):
        if model_id and model_id != self.model_id:
            self.model_id = None  # force a fresh load with default choices
        self.refresh(select=model_id)
        if model_id and version:
            idx = next((i for i, v in enumerate(self.versions) if v.version == version), None)
            if idx is not None:
                self.version_cb.current(idx)
                self.source.set("version")
                self._source_changed()
        if model_id and checkpoint:
            idx = next((i for i, c in enumerate(self.ckpts) if c["path"] == checkpoint), None)
            if idx is not None:
                self.ckpt_cb.current(idx)
                self.source.set("checkpoint")
                self._source_changed()

    def refresh(self, select: str | None = None) -> None:
        select = select or self.model_id
        store = self.ctx.store
        self.table.clear()
        models = [m for m in store.list_models()]
        if not models:
            self.table.grid_forget()
            self.empty.grid(row=0, column=0, sticky="ew")
            self.detail.grid_remove()
            return
        self.empty.grid_forget()
        self.table.grid(row=0, column=0, sticky="ew")
        self.detail.grid()
        for m in models:
            vs = store.list_versions(m.id)
            act = m.meta.get("active_version")
            act_v = next((v for v in vs if v.version == act), None)
            self.table.insert(m.id, [m.name + ("  (archived)" if m.archived else ""), act or "—",
                                     m.meta.get("best_version") or "—", len(vs),
                                     fmt_int(act_v.meta.get("total_timesteps")) if act_v else "—",
                                     fmt_pct(act_v.val_accuracy) if act_v else "—",
                                     fmt_date(m.meta.get("last_trained_at")), human_size(dir_size(m.path))],
                              tags=("muted",) if m.archived else ())
        target = select if select and self.table.tv.exists(select) else models[0].id
        self.table.select(target)
        # Load synchronously; the (asynchronous) select event is ignored for the already-loaded model.
        self._model_selected(target)

    def _on_table_select(self) -> None:
        mid = self.table.selected()
        if mid and mid != self.model_id:
            self._model_selected(mid)

    def _model_selected(self, mid: str) -> None:
        keep = self.source.get() if mid == self.model_id else None
        keep_version = self.version_var.get() if keep == "version" else None
        keep_ckpt = self.ckpt_var.get() if keep == "checkpoint" else None
        self.model_id = mid
        store = self.ctx.store
        try:
            m = store.get_model(mid)
        except ModelError as exc:
            error(self, str(exc))
            return
        self.versions = store.list_versions(mid)
        self.ckpts = store.list_checkpoints(mid)
        meta = m.meta
        act = next((v for v in self.versions if v.is_active), None)
        self.info.set("created", fmt_date(meta.get("created_at")))
        self.info.set("last", fmt_date(meta.get("last_trained_at")))
        self.info.set("active", meta.get("active_version") or "—")
        self.info.set("best", meta.get("best_version") or "—")
        self.info.set("versions", str(len(self.versions)))
        self.info.set("steps", fmt_int(act.meta.get("total_timesteps")) if act else "—")
        self.info.set("sessions", str(len(store.list_sessions(mid))))
        self.info.set("val", fmt_pct(act.val_accuracy) if act else "—")
        arch = meta.get("architecture") or {}
        self.info.set("arch", f"PPO MLP {'x'.join(map(str, arch.get('net_arch') or []))}")
        self.info.set("dataset", Path(act.meta.get("maps_folder") or "—").name if act else "—")
        self.info.set("ckpt", Path(self.ckpts[-1]["path"]).name if self.ckpts else "none")
        self.info.set("size", human_size(dir_size(m.path)))
        for k in ("active", "best", "latest"):
            self.src_lbls[k].configure(text=meta.get(f"{k}_version") or "none")
        self.version_cb.configure(values=[version_label(v) for v in self.versions])
        if self.versions:
            self.version_cb.current(len(self.versions) - 1)
        self.ckpt_cb.configure(values=[
            f"{Path(c['path']).name}  ·  {fmt_int(c['timesteps'])} steps  ·  from {c['parent_version'] or 'new'}"
            f"  ·  val {fmt_pct(c['val_accuracy'], 1)}{'  · BEST of session' if c['is_best'] else ''}"
            f"  ·  {fmt_date(c['modified'])}" for c in self.ckpts])
        if self.ckpts:
            self.ckpt_cb.current(len(self.ckpts) - 1)
        if keep_version in self.version_cb["values"]:
            self.version_var.set(keep_version)
        if keep_ckpt in self.ckpt_cb["values"]:
            self.ckpt_var.set(keep_ckpt)
        if keep is None:
            self.source.set("active")
        self._source_changed()

    def _source_meta(self) -> tuple[str | None, str | None, dict]:
        """(version, checkpoint_path, meta of the version the settings come from)."""
        if not self.model_id:
            return None, None, {}
        src = self.source.get()
        store = self.ctx.store
        meta = store.get_model(self.model_id).meta
        if src in ("active", "best", "latest"):
            v = meta.get(f"{src}_version") or (meta.get("latest_version") if src == "active" else None)
            if not v:
                return None, None, {}
            return v, None, store.get_version(self.model_id, v).meta
        if src == "version":
            i = self.version_cb.current()
            if i < 0:
                return None, None, {}
            v = self.versions[i]
            return v.version, None, v.meta
        i = self.ckpt_cb.current()
        if i < 0:
            return None, None, {}
        c = self.ckpts[i]
        parent = c["parent_version"]
        pmeta = {}
        if parent:
            try:
                pmeta = store.get_version(self.model_id, parent).meta
            except ModelError:
                pass
        return None, c["path"], pmeta

    def _source_changed(self) -> None:
        try:
            _v, _c, meta = self._source_meta()
        except ModelError:
            meta = {}
        hp = meta.get("hyperparameters") or {}
        env = meta.get("env_config") or {}
        if hp.get("learning_rate"):
            self.lr.set(f"{hp['learning_rate']:g}")
        if hp.get("batch_size"):
            self.batch.set(str(hp["batch_size"]))
        if hp.get("n_steps"):
            self.adv.n_steps.set(str(hp["n_steps"]))
        if hp.get("n_epochs"):
            self.adv.n_epochs.set(str(hp["n_epochs"]))
        if hp.get("n_envs"):
            self.adv.n_envs.set(str(hp["n_envs"]))
        for k, var in (("dt_ms", self.adv.dt), ("jitter_ms", self.adv.jitter), ("clip_seconds", self.adv.clip)):
            if k in env:
                var.set(f"{env[k]:g}")
        sr = meta.get("star_range") or {}
        self.stars.set(sr.get("min"), sr.get("max"))  # same range as last time unless you change it
        self.mods.set(meta.get("mods"))  # and the same mods
        folder = meta.get("maps_folder")
        if folder and Path(folder).exists() and folder != self.maps.get():
            self.maps.set(folder)
        elif self.maps.result is None:
            self.maps.scan()
        self._update_warnings()

    def _build_spec(self) -> TrainingSpec:
        version, ckpt, _meta = self._source_meta()
        if not self.model_id:
            raise ModelError("Choose a model.")
        if not version and not ckpt:
            raise ModelError("This model has no version or checkpoint to continue from yet.")
        spec = TrainingSpec(
            model_id=self.model_id, kind="continue", source_version=version, source_checkpoint=ckpt,
            maps_folder=self.maps.get(), timesteps=self.length.get(),
            learning_rate=parse_float(self.lr, "Learning rate"), batch_size=parse_int(self.batch, "Batch size"),
            checkpoint_every=parse_int(self.ckpt_every, "Checkpoint frequency"), device=self.device.get(),
            notes=self.notes.get().strip(), set_active=self.set_active.get(),
        )
        self.adv.apply(spec)
        self.stars.apply(spec)
        self.mods.apply(spec)
        return spec

    def _update_warnings(self) -> None:
        try:
            spec = self._build_spec()
            _v, _c, meta = self._source_meta()
            warns = compatibility_warnings(meta, spec)
        except (ModelError, ValueError):
            warns = []
        self.warn_lbl.configure(text="\n".join(f"⚠  {w}" for w in warns))

    def submit(self) -> None:
        if _running_guard(self.ctx, self):
            return
        try:
            spec = self._build_spec()
            errs = validate_spec(spec)
            if errs:
                raise ModelError("\n".join(errs))
            _v, _c, meta = self._source_meta()
        except ModelError as exc:
            error(self, str(exc), "Check the settings")
            return
        store = self.ctx.store
        m = store.get_model(spec.model_id)
        next_v = f"v{int(m.meta.get('next_version') or 1):03d}"
        source = spec.source_version or "?"
        if spec.source_checkpoint:
            source = f"checkpoint {Path(spec.source_checkpoint).name}" + (f" (from {spec.source_version})" if spec.source_version else "")
        n = len(self.stars.in_range())
        if not n:
            error(self, "No maps fall inside that star range (or the folder has no usable maps). "
                        "Widen the range or clear it.", "Check the settings")
            return
        rows = [
            ("Model", m.name),
            ("Mode", "osu!mania 4K"),
            ("Continue from", source),
            ("New version", f"{next_v} (parent {spec.source_version or 'checkpoint'}); existing versions are kept"),
            ("Restores", "weights, optimiser state, step counter and learning schedule"),
            ("Training maps", f"{n} map(s) in {spec.maps_folder}"),
            ("Star range", star_range_label(spec.min_stars, spec.max_stars)),
            ("Mods", mod_list_label(spec.mod_bits)),
            ("Device", _device_summary(self.ctx, spec.device)),
            ("Additional training", f"{spec.timesteps:,} timesteps"),
            ("Batch size", f"{spec.batch_size:,}"),
            ("Learning rate", f"{spec.learning_rate:g}"),
            ("Storage location", str(store.model_dir(spec.model_id))),
        ]
        if not summary_dialog(self, "Continue training?", rows, compatibility_warnings(meta, spec)):
            return
        try:
            self.ctx.start_training(spec)
        except Exception as exc:  # noqa: BLE001
            error(self, str(exc), "Couldn't start training")
            return
        self.notes.set("")
        self.ctx.app.show("training")
