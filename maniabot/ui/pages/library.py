"""Model Library: models, version history (lineage), checkpoints, sessions, storage, compare."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from maniabot import osu_integration
from maniabot.beatmaps import scan_maps
from maniabot.evaluation import POLICIES, compare, evaluate, strip_private
from maniabot.legacy import read_sb3_data
from maniabot.models import DeletionPlan, ModelError, now_iso
from maniabot.storage import dir_size, human_size
from maniabot.ui import theme
from maniabot.ui.pages.base import MapsFolderField, Page, version_label
from maniabot.ui.widgets import (
    Card, Dialog, EmptyState, KeyValueGrid, Table, ask_string, badge_row, confirm_delete, error, fmt_date,
    fmt_duration, fmt_int, fmt_pct, fmt_value, message,
)


class LibraryPage(Page):
    title = "Model Library"
    subtitle = "All your models and every saved version"

    def build(self) -> None:
        self.action("Create New Model", lambda: self.ctx.app.show("create"), accent=True)
        self.list_view = ttk.Frame(self.body)
        self.list_view.columnconfigure(0, weight=1)
        self.detail_view = ttk.Frame(self.body)
        self.detail_view.columnconfigure(0, weight=1)
        self._build_list()
        self._build_detail()
        self.model_id: str | None = None
        self.ctx.on("models_changed", lambda **_: self._refresh_current())
        self.ctx.on("training_event", lambda event: self._refresh_current() if event["type"] == "done" else None)

    # ================================================================ list
    def _build_list(self) -> None:
        bar = ttk.Frame(self.list_view)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.search = tk.StringVar()
        self.show_archived = tk.BooleanVar(value=False)
        ttk.Label(bar, text="Search").pack(side="left")
        e = ttk.Entry(bar, textvariable=self.search, width=30)
        e.pack(side="left", padx=8)
        ttk.Checkbutton(bar, text="Show archived", variable=self.show_archived, command=self.refresh_list).pack(side="left")
        self.search.trace_add("write", lambda *_: self.refresh_list())
        self.pick_hint = ttk.Label(bar, text="", style="Muted.TLabel")
        self.pick_hint.pack(side="left", padx=14)

        card = Card(self.list_view, padding=0)
        card.grid(row=1, column=0, sticky="nsew")
        self.table = Table(card.body, [
            ("name", "Model", 190), ("active", "Current", 70), ("best", "Best", 60), ("versions", "Versions", 70),
            ("steps", "Total steps", 100), ("acc", "Last evaluation", 110), ("created", "Created", 130),
            ("last", "Last trained", 130), ("size", "Size", 80)], height=14)
        self.table.grid(row=0, column=0, sticky="nsew")
        self.table.on_double(self.open_selected)
        self.table.on_select(self._update_list_buttons)
        self.list_empty = EmptyState(self.list_view, "No models yet",
                                     "Create your first osu!mania 4K model. Everything you train is stored in your "
                                     "user data folder, never in the source repository.",
                                     [("Create New Model", lambda: self.ctx.app.show("create"))])

        acts = ttk.Frame(self.list_view)
        acts.grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.list_btns = []
        for text, fn, style in (
            ("Open", self.open_selected, "Accent.TButton"),
            ("Version History", lambda: self.open_selected("versions"), "TButton"),
            ("Continue Training", self._list_continue, "TButton"),
            ("Test Model", lambda: self._list_test("watch"), "TButton"),
            ("Generate Replay", lambda: self._list_test("replay"), "TButton"),
            ("Rename", self._rename, "TButton"),
            ("Archive", self._toggle_archive, "TButton"),
            ("Delete Model", self._delete_model, "Danger.TButton"),
        ):
            b = ttk.Button(acts, text=text, command=fn, style=style)
            b.pack(side="left", padx=(0, 6))
            self.list_btns.append(b)
        self.archive_btn = self.list_btns[6]

    def refresh_list(self) -> None:
        store = self.ctx.store
        models = store.list_models(include_archived=self.show_archived.get())
        q = self.search.get().strip().lower()
        if q:
            models = [m for m in models if q in m.name.lower() or q in (m.meta.get("description") or "").lower()]
        sel = self.table.selected()
        self.table.clear()
        if not store.list_models():
            self.table.master.master.grid_remove()
            self.list_empty.grid(row=1, column=0, sticky="ew")
        else:
            self.list_empty.grid_forget()
            self.table.master.master.grid()
        for m in models:
            try:
                vs = store.list_versions(m.id)
            except ModelError:
                continue
            act = next((v for v in vs if v.is_active), vs[-1] if vs else None)
            evals = store.list_evaluations(m.id)
            last_eval = f"{fmt_pct(evals[-1].get('accuracy'))} ({evals[-1].get('version')})" if evals else "—"
            self.table.insert(m.id, [
                m.name + ("  (archived)" if m.archived else ""), m.meta.get("active_version") or "—",
                m.meta.get("best_version") or "—", len(vs),
                fmt_int(act.meta.get("total_timesteps")) if act else "—", last_eval,
                fmt_date(m.meta.get("created_at")), fmt_date(m.meta.get("last_trained_at")), human_size(dir_size(m.path)),
            ], tags=("muted",) if m.archived else ())
        if sel and self.table.tv.exists(sel):
            self.table.select(sel)
        self._update_list_buttons()

    def _update_list_buttons(self) -> None:
        mid = self.table.selected()
        for b in self.list_btns:
            b.state(["!disabled"] if mid else ["disabled"])
        if mid:
            try:
                arch = self.ctx.store.get_model(mid).archived
            except ModelError:
                arch = False
            self.archive_btn.configure(text="Unarchive" if arch else "Archive")

    def _need(self) -> str | None:
        mid = self.table.selected()
        if not mid:
            message(self, "Choose a model", "Select a model in the list first.")
        return mid

    def open_selected(self, tab: str = "versions") -> None:
        mid = self._need()
        if mid:
            self.open_model(mid, tab)

    def _list_continue(self) -> None:
        mid = self._need()
        if mid:
            self.ctx.app.show("continue", model_id=mid)

    def _list_test(self, mode: str) -> None:
        mid = self._need()
        if mid:
            self.ctx.app.show("test", model_id=mid, mode=mode)

    def _rename(self, mid: str | None = None) -> None:
        mid = mid or self._need()
        if not mid:
            return
        rec = self.ctx.store.get_model(mid)
        new = ask_string(self, "Rename model", "New display name (the storage folder keeps its name):", rec.name)
        if new and new != rec.name:
            try:
                self.ctx.store.rename_model(mid, new)
            except ModelError as exc:
                error(self, str(exc))
                return
            self.ctx.emit("models_changed")

    def _toggle_archive(self, mid: str | None = None) -> None:
        mid = mid or self._need()
        if not mid:
            return
        rec = self.ctx.store.get_model(mid)
        self.ctx.store.update_model(mid, archived=not rec.archived)
        self.ctx.emit("models_changed")

    def _delete_model(self, mid: str | None = None) -> None:
        mid = mid or self._need()
        if not mid:
            return
        store = self.ctx.store
        try:
            plan = store.plan_delete_model(mid)
        except ModelError as exc:
            error(self, str(exc))
            return
        name = store.get_model(mid).name
        ok, prot = confirm_delete(self, plan, "Delete Model", require_text=name)
        if not ok:
            return
        try:
            store.delete_model(mid, confirm_protected=prot)
        except ModelError as exc:
            error(self, str(exc))
            return
        POLICIES.clear()
        if self.model_id == mid:
            self.model_id = None
            self.show_list()
        self.ctx.emit("models_changed")
        message(self, "Model deleted", f"{name} and all of its files were deleted. Its replays are still in Replay History.",
                "success")

    # ============================================================== detail
    def _build_detail(self) -> None:
        top = ttk.Frame(self.detail_view)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Button(top, text="← All models", style="Link.TButton", command=self.show_list).grid(row=0, column=0, sticky="w")
        head = Card(self.detail_view)
        head.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        hb = head.body
        hb.columnconfigure(0, weight=1)
        self.d_name = ttk.Label(hb, text="", style="CardH2.TLabel")
        self.d_name.grid(row=0, column=0, sticky="w")
        self.d_badges = tk.Frame(hb, bg=theme.SURFACE)
        self.d_badges.grid(row=0, column=1, sticky="e")
        self.d_desc = ttk.Label(hb, text="", style="CardMuted.TLabel", wraplength=800, justify="left")
        self.d_desc.grid(row=1, column=0, columnspan=2, sticky="w")
        self.d_kv = KeyValueGrid(hb, columns=4)
        self.d_kv.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        for k, l in (("active", "Current"), ("best", "Best"), ("latest", "Latest"), ("versions", "Versions"),
                     ("created", "Created"), ("last", "Last trained"), ("arch", "Architecture"), ("size", "Size")):
            self.d_kv.add(k, l)
        acts = ttk.Frame(hb, style="Card.TFrame")
        acts.grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 0))
        for text, fn, style in (
            ("Continue Training", lambda: self.ctx.app.show("continue", model_id=self.model_id), "Accent.TButton"),
            ("Test Model", lambda: self.ctx.app.show("test", model_id=self.model_id, mode="watch"), "TButton"),
            ("Generate Replay", lambda: self.ctx.app.show("test", model_id=self.model_id, mode="replay"), "TButton"),
            ("Compare Versions", lambda: self._compare(), "TButton"),
            ("Edit description", self._edit_desc, "TButton"),
            ("Rename", lambda: self._rename(self.model_id), "TButton"),
            ("Archive", lambda: self._toggle_archive(self.model_id), "TButton"),
            ("Open folder", lambda: osu_integration.open_folder(self.ctx.store.model_dir(self.model_id)), "TButton"),
            ("Delete Model", lambda: self._delete_model(self.model_id), "Danger.TButton"),
        ):
            b = ttk.Button(acts, text=text, command=fn, style=style)
            b.pack(side="left", padx=(0, 6))
            if text == "Archive":
                self.d_archive_btn = b

        self.nb = ttk.Notebook(self.detail_view)
        self.nb.grid(row=2, column=0, sticky="nsew", pady=(14, 0))
        self.tab_versions = ttk.Frame(self.nb, style="Card.TFrame", padding=12)
        self.tab_ckpts = ttk.Frame(self.nb, style="Card.TFrame", padding=12)
        self.tab_sessions = ttk.Frame(self.nb, style="Card.TFrame", padding=12)
        self.tab_storage = ttk.Frame(self.nb, style="Card.TFrame", padding=12)
        self.nb.add(self.tab_versions, text="Version History")
        self.nb.add(self.tab_ckpts, text="Checkpoints")
        self.nb.add(self.tab_sessions, text="Training Sessions")
        self.nb.add(self.tab_storage, text="Storage")
        self._build_versions_tab()
        self._build_ckpt_tab()
        self._build_sessions_tab()
        self._build_storage_tab()

    def _build_versions_tab(self) -> None:
        t = self.tab_versions
        t.columnconfigure(0, weight=1)
        ttk.Label(t, text="Versions are listed as a lineage tree: each version sits under the version it was trained "
                          "from. Training never overwrites a version.", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        self.v_table = Table(t, [
            ("version", "Version", 120), ("tags", "Status", 190), ("created", "Created", 130),
            ("steps", "Trained steps", 95), ("total", "Total steps", 95), ("val", "Validation", 85),
            ("how", "How it ended", 90), ("size", "Size", 70), ("notes", "Notes", 200)], height=10, tree=True,
            selectmode="extended")
        self.v_table.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.v_table.on_double(self._version_details)
        self.v_table.on_select(self._update_version_buttons)
        r1 = ttk.Frame(t, style="Card.TFrame")
        r1.grid(row=2, column=0, sticky="w", pady=(10, 0))
        r2 = ttk.Frame(t, style="Card.TFrame")
        r2.grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.v_btns = {}
        for row, specs in ((r1, (
            ("Details", self._version_details, "TButton"),
            ("Test", lambda: self._v_test("watch"), "Accent.TButton"),
            ("Generate Replay", lambda: self._v_test("replay"), "TButton"),
            ("Continue Training From Here", self._v_continue, "TButton"),
            ("Set as Active", self._v_set_active, "TButton"),
            ("Mark as Best", self._v_mark_best, "TButton"),
        )), (r2, (
            ("Protect", lambda: self._v_toggle("protected"), "TButton"),
            ("Favourite", lambda: self._v_toggle("favourite"), "TButton"),
            ("Archive", lambda: self._v_toggle("archived"), "TButton"),
            ("Evaluate…", self._v_evaluate, "TButton"),
            ("Compare Selected", lambda: self._compare(self.v_table.selected_all()), "TButton"),
            ("Delete Version", self._v_delete, "Danger.TButton"),
        ))):
            for text, fn, style in specs:
                b = ttk.Button(row, text=text, command=fn, style=style)
                b.pack(side="left", padx=(0, 6))
                self.v_btns[text] = b

    def _build_ckpt_tab(self) -> None:
        t = self.tab_ckpts
        t.columnconfigure(0, weight=1)
        ttk.Label(t, text="Intermediate saves made during training. Each one can be resumed from or turned into a "
                          "version.", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        self.c_table = Table(t, [("file", "Checkpoint", 220), ("steps", "Total steps", 100), ("from", "Trained from", 90),
                                 ("val", "Validation", 90), ("tags", "", 120), ("session", "Session", 170),
                                 ("size", "Size", 70), ("when", "Saved", 130)], height=10, selectmode="extended")
        self.c_table.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        r = ttk.Frame(t, style="Card.TFrame")
        r.grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Button(r, text="Continue Training From Checkpoint", style="Accent.TButton", command=self._c_continue).pack(side="left")
        ttk.Button(r, text="Save as New Version", command=self._c_promote).pack(side="left", padx=6)
        ttk.Button(r, text="Delete Selected", style="Danger.TButton", command=self._c_delete).pack(side="left")

    def _build_sessions_tab(self) -> None:
        t = self.tab_sessions
        t.columnconfigure(0, weight=1)
        self.s_table = Table(t, [("id", "Session", 170), ("status", "Status", 90), ("from", "Started from", 90),
                                 ("result", "Created", 70), ("steps", "Steps", 90), ("maps", "Maps", 60),
                                 ("lr", "Learning rate", 90), ("device", "Device", 60), ("start", "Started", 130),
                                 ("dur", "Duration", 80), ("note", "Note", 220)], height=10)
        self.s_table.grid(row=0, column=0, sticky="nsew")
        r = ttk.Frame(t, style="Card.TFrame")
        r.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(r, text="Open learning curve", command=self._s_curve).pack(side="left")
        ttk.Button(r, text="Delete training logs", style="Danger.TButton", command=self._s_delete_logs).pack(side="left", padx=6)

    def _build_storage_tab(self) -> None:
        t = self.tab_storage
        t.columnconfigure(0, weight=1)
        self.st_table = Table(t, [("item", "Item", 280), ("count", "Count", 80), ("size", "Size", 100)], height=8)
        self.st_table.grid(row=0, column=0, sticky="nsew")
        r = ttk.Frame(t, style="Card.TFrame")
        r.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(r, text="Delete all checkpoints", style="Danger.TButton", command=self._st_del_ckpts).pack(side="left")
        ttk.Button(r, text="Delete training logs", style="Danger.TButton", command=self._s_delete_logs).pack(side="left", padx=6)
        ttk.Button(r, text="Delete evaluation results", style="Danger.TButton", command=self._st_del_evals).pack(side="left")
        ttk.Label(t, text="Versions are only deleted from the Version History tab. Cleanup here never touches versions.",
                  style="CardMuted.TLabel").grid(row=2, column=0, sticky="w", pady=(8, 0))

    # ------------------------------------------------------------- views
    def on_show(self, model_id: str | None = None, tab: str | None = None, pick: bool = False, **_) -> None:
        if model_id:
            self.open_model(model_id, tab or "versions")
        elif pick and self.model_id:
            self.open_model(self.model_id, tab or "versions")
        elif pick:
            self.show_list(hint="Choose a model to see its version history.")
        elif self.model_id and self.detail_view.winfo_ismapped():
            self.refresh_detail()
        else:
            self.show_list()

    def show_list(self, hint: str = "") -> None:
        self.detail_view.grid_forget()
        self.list_view.grid(row=0, column=0, sticky="nsew")
        self.set_title(self.title, self.subtitle)
        self.pick_hint.configure(text=hint)
        self.refresh_list()

    def open_model(self, model_id: str, tab: str = "versions") -> None:
        if not self.ctx.store.exists(model_id):
            error(self, "That model no longer exists.")
            self.show_list()
            return
        self.model_id = model_id
        self.list_view.grid_forget()
        self.detail_view.grid(row=0, column=0, sticky="nsew")
        self.refresh_detail()
        idx = {"versions": 0, "checkpoints": 1, "sessions": 2, "storage": 3}.get(tab, 0)
        self.nb.select(idx)

    def _refresh_current(self) -> None:
        if not self.winfo_exists():
            return
        if self.model_id and self.detail_view.winfo_ismapped():
            if self.ctx.store.exists(self.model_id):
                self.refresh_detail()
            else:
                self.model_id = None
                self.show_list()
        elif self.list_view.winfo_ismapped():
            self.refresh_list()

    def refresh_detail(self) -> None:
        store = self.ctx.store
        mid = self.model_id
        try:
            m = store.get_model(mid)
            versions = store.list_versions(mid)
        except ModelError as exc:
            error(self, str(exc))
            self.show_list()
            return
        self.set_title(m.name, f"osu!mania 4K model · {mid}")
        self.d_name.configure(text=m.name)
        for w in self.d_badges.winfo_children():
            w.destroy()
        badge_row(self.d_badges, ["Archived"] if m.archived else []).pack()
        self.d_desc.configure(text=m.meta.get("description") or "No description.")
        self.d_archive_btn.configure(text="Unarchive" if m.archived else "Archive")
        for k in ("active", "best", "latest"):
            self.d_kv.set(k, m.meta.get(f"{k}_version") or "—")
        if m.meta.get("best_version"):
            self.d_kv.set("best", f"{m.meta['best_version']} ({m.meta.get('best_source') or 'auto'})")
        self.d_kv.set("versions", str(len(versions)))
        self.d_kv.set("created", fmt_date(m.meta.get("created_at")))
        self.d_kv.set("last", fmt_date(m.meta.get("last_trained_at")))
        self.d_kv.set("arch", "PPO MLP " + "x".join(map(str, (m.meta.get("architecture") or {}).get("net_arch") or [])))
        self.d_kv.set("size", human_size(dir_size(m.path)))

        sel = self.v_table.selected_all()
        self.v_table.clear()
        present = {v.version for v in versions}
        for v in versions:  # sorted by number, so parents are inserted before children
            parent = v.parent if v.parent in present else ""
            notes = v.meta.get("notes") or ""
            if v.parent and v.parent not in present:
                notes = f"(parent {v.parent} deleted) {notes}"
            if v.meta.get("missing_metadata"):
                notes = "metadata missing"
            self.v_table.insert(v.version, [
                ", ".join(v.badges()), fmt_date(v.meta.get("created_at")), fmt_int(v.meta.get("trained_timesteps")),
                fmt_int(v.meta.get("total_timesteps")), fmt_pct(v.val_accuracy), v.meta.get("status") or "—",
                human_size(v.meta.get("file_size") or 0), notes], parent=parent, text=v.version,
                tags=("muted",) if v.archived else (("accent",) if v.is_active else ()))
        for s in sel:
            if self.v_table.tv.exists(s):
                self.v_table.tv.selection_add(s)
        if not sel and versions:
            act = m.meta.get("active_version")
            self.v_table.select(act if act in present else versions[-1].version)
        self._update_version_buttons()
        self._refresh_ckpts()
        self._refresh_sessions()
        self._refresh_storage(versions)

    def _update_version_buttons(self) -> None:
        sel = self.v_table.selected_all()
        one = len(sel) == 1
        for name, b in self.v_btns.items():
            ok = (len(sel) == 2) if name == "Compare Selected" else one
            b.state(["!disabled"] if ok else ["disabled"])
        if one:
            try:
                v = self.ctx.store.get_version(self.model_id, sel[0])
            except ModelError:
                return
            self.v_btns["Protect"].configure(text="Unprotect" if v.protected else "Protect")
            self.v_btns["Favourite"].configure(text="Unfavourite" if v.favourite else "Favourite")
            self.v_btns["Archive"].configure(text="Unarchive" if v.archived else "Archive")
            self.v_btns["Mark as Best"].configure(text="Use automatic Best" if (v.is_best and v.model_meta.get(
                "best_source") == "manual") else "Mark as Best")

    def _refresh_ckpts(self) -> None:
        self.c_table.clear()
        self.ckpts = self.ctx.store.list_checkpoints(self.model_id)
        for c in self.ckpts:
            tags = ", ".join(t for t, on in (("best of session", c["is_best"]), ("latest", c["is_latest"])) if on)
            self.c_table.insert(c["path"], [Path(c["path"]).name, fmt_int(c["timesteps"]), c["parent_version"] or "new",
                                            fmt_pct(c["val_accuracy"]), tags, c["session_id"], human_size(c["size"]),
                                            fmt_date(c["modified"])])

    def _refresh_sessions(self) -> None:
        self.s_table.clear()
        for s in reversed(self.ctx.store.list_sessions(self.model_id)):
            spec = s.get("spec") or {}
            self.s_table.insert(s["session_id"], [
                s["session_id"], s.get("status"), s.get("start_version") or ("new" if s.get("kind") == "new" else "—"),
                s.get("result_version") or "—", fmt_int(spec.get("timesteps")), len(s.get("training_maps") or []) or "—",
                f"{spec['learning_rate']:g}" if spec.get("learning_rate") else "—", s.get("device") or "—",
                fmt_date(s.get("started_at")), fmt_duration(s.get("duration_s")), s.get("error") or s.get("notes") or ""],
                tags=("bad",) if s.get("status") in ("failed", "interrupted") else ())

    def _refresh_storage(self, versions) -> None:
        store, mid = self.ctx.store, self.model_id
        self.st_table.clear()
        for v in versions:
            self.st_table.insert(f"v:{v.version}", [f"Version {v.version}", 1, human_size(dir_size(v.path))])
        self.st_table.insert("ckpt", ["Checkpoints", len(self.ckpts), human_size(dir_size(store.checkpoints_dir(mid)))])
        logs = list(store.sessions_dir(mid).glob("*_episodes.csv")) + list(store.sessions_dir(mid).glob("*_curve.png"))
        self.st_table.insert("logs", ["Training logs and curves", len(logs), human_size(sum(f.stat().st_size for f in logs))])
        evals = store.list_evaluations(mid)
        self.st_table.insert("eval", ["Evaluation results", len(evals), human_size(dir_size(store.evaluation_dir(mid)))])
        self.st_table.insert("total", ["Total", "", human_size(dir_size(store.model_dir(mid)))], tags=("accent",))

    # ----------------------------------------------------- model actions
    def _edit_desc(self) -> None:
        rec = self.ctx.store.get_model(self.model_id)
        new = ask_string(self, "Edit description", "Description:", rec.meta.get("description") or "")
        if new is not None:
            self.ctx.store.update_model(self.model_id, description=new)
            self.refresh_detail()

    # --------------------------------------------------- version actions
    def _sel_version(self) -> str | None:
        sel = self.v_table.selected_all()
        return sel[0] if len(sel) == 1 else None

    def _v_test(self, mode: str) -> None:
        v = self._sel_version()
        if v:
            self.ctx.app.show("test", model_id=self.model_id, version=v, mode=mode)

    def _v_continue(self) -> None:
        v = self._sel_version()
        if v:
            self.ctx.app.show("continue", model_id=self.model_id, version=v)

    def _v_set_active(self) -> None:
        v = self._sel_version()
        if v:
            self.ctx.store.set_active(self.model_id, v)
            self.ctx.emit("models_changed")
            message(self, "Active version changed", f"{v} is now the active version. No other versions were changed.",
                    "success")

    def _v_mark_best(self) -> None:
        v = self._sel_version()
        if not v:
            return
        rec = self.ctx.store.get_version(self.model_id, v)
        if rec.is_best and rec.model_meta.get("best_source") == "manual":
            self.ctx.store.set_best(self.model_id, None)
        else:
            self.ctx.store.set_best(self.model_id, v)
        self.ctx.emit("models_changed")

    def _v_toggle(self, flag: str) -> None:
        v = self._sel_version()
        if not v:
            return
        rec = self.ctx.store.get_version(self.model_id, v)
        self.ctx.store.update_version(self.model_id, v, **{flag: not rec.meta.get(flag)})
        self.refresh_detail()

    def _v_delete(self) -> None:
        v = self._sel_version()
        if not v:
            return
        try:
            plan = self.ctx.store.plan_delete_version(self.model_id, v)
        except ModelError as exc:
            error(self, str(exc))
            return
        ok, prot = confirm_delete(self, plan, "Delete Version")
        if not ok:
            return
        try:
            self.ctx.store.delete_version(self.model_id, v, confirm_protected=prot)
        except ModelError as exc:
            error(self, str(exc))
            return
        POLICIES.clear()
        self.ctx.emit("models_changed")

    def _version_details(self) -> None:
        v = self._sel_version()
        if v:
            VersionDetails(self, self.ctx, self.model_id, v).show()
            self.refresh_detail()

    def _v_evaluate(self) -> None:
        v = self._sel_version()
        if v:
            EvaluateDialog(self, self.ctx, [(self.model_id, v)]).show()
            self.refresh_detail()

    def _compare(self, selected: list[str] | None = None) -> None:
        versions = self.ctx.store.list_versions(self.model_id)
        if len(versions) + sum(len(self.ctx.store.list_versions(m.id)) for m in self.ctx.store.list_models()
                               if m.id != self.model_id) < 2:
            message(self, "Nothing to compare", "You need at least two versions to compare.")
            return
        sel = selected or []
        a = (self.model_id, sel[0]) if sel else (self.model_id, versions[-2].version if len(versions) > 1 else versions[-1].version)
        b = (self.model_id, sel[1]) if len(sel) > 1 else (self.model_id, versions[-1].version)
        CompareDialog(self, self.ctx, a, b).show()

    # ------------------------------------------------ checkpoint actions
    def _sel_ckpt(self) -> dict | None:
        sel = self.c_table.selected_all()
        if len(sel) != 1:
            message(self, "Choose one checkpoint", "Select a single checkpoint first.")
            return None
        return next(c for c in self.ckpts if c["path"] == sel[0])

    def _c_continue(self) -> None:
        c = self._sel_ckpt()
        if c:
            self.ctx.app.show("continue", model_id=self.model_id, checkpoint=c["path"])

    def _c_promote(self) -> None:
        c = self._sel_ckpt()
        if not c:
            return
        store = self.ctx.store
        parent = c["parent_version"]
        try:
            pmeta = store.get_version(self.model_id, parent).meta if parent else {}
        except ModelError:
            pmeta = {}
        try:
            data = read_sb3_data(Path(c["path"]))
            m = store.get_model(self.model_id).meta
            sess = next((s for s in store.list_sessions(self.model_id) if s["session_id"] == c["session_id"]), {})
            rec = store.version_from_file(self.model_id, Path(c["path"]), {
                "parent_version": parent,
                "session_id": c["session_id"],
                "status": "from checkpoint",
                "source": f"checkpoint {Path(c['path']).name}",
                "source_checkpoint": c["path"],
                "total_timesteps": int(data.get("num_timesteps", 0)),
                "trained_timesteps": int(data.get("num_timesteps", 0)) - int(pmeta.get("total_timesteps") or 0),
                "hyperparameters": {k: data.get(k) for k in ("learning_rate", "n_steps", "batch_size", "n_epochs", "n_envs")},
                "env_config": pmeta.get("env_config") or m.get("env_config"),
                "architecture": pmeta.get("architecture") or m.get("architecture"),
                "maps_folder": (sess.get("spec") or {}).get("maps_folder"),
                "training_maps": sess.get("training_maps") or [],
                "validation_maps": sess.get("validation_maps") or [],
                "validation": {"accuracy": c["val_accuracy"], "partial": True, "created_at": now_iso()}
                if c["val_accuracy"] is not None else None,
                "notes": f"Saved from checkpoint {Path(c['path']).name}",
            }, set_active=False)
        except ModelError as exc:
            error(self, str(exc))
            return
        self.ctx.emit("models_changed")
        message(self, "Version created", f"Saved the checkpoint as {rec.version} (parent {parent or 'none'}).", "success")

    def _c_delete(self) -> None:
        sel = self.c_table.selected_all()
        if not sel:
            return
        size = sum(Path(p).stat().st_size for p in sel if Path(p).exists())
        plan = DeletionPlan(title=f"Delete {len(sel)} checkpoint(s)?", items=[Path(p).name for p in sel[:10]] +
                            ([f"…and {len(sel) - 10} more"] if len(sel) > 10 else []), size_bytes=size,
                            warnings=["Versions are not affected."])
        ok, _ = confirm_delete(self, plan, "Delete Checkpoints")
        if ok:
            self.ctx.store.delete_checkpoints(self.model_id, sel)
            self.refresh_detail()

    def _st_del_ckpts(self) -> None:
        paths = [c["path"] for c in self.ctx.store.list_checkpoints(self.model_id)]
        if not paths:
            message(self, "No checkpoints", "This model has no checkpoints.")
            return
        live = self.ctx.job and not self.ctx.job.finished and self.ctx.job.spec.model_id == self.model_id
        if live:
            message(self, "Training is running", "Wait for this model's training session to finish first.", "warning")
            return
        plan = DeletionPlan(title=f"Delete all {len(paths)} checkpoints?", items=["All intermediate checkpoints"],
                            size_bytes=dir_size(self.ctx.store.checkpoints_dir(self.model_id)),
                            warnings=["You won't be able to resume from these checkpoints. Versions are kept."])
        ok, _ = confirm_delete(self, plan, "Delete Checkpoints")
        if ok:
            self.ctx.store.delete_checkpoints(self.model_id, paths)
            self.refresh_detail()

    def _st_del_evals(self) -> None:
        n = len(self.ctx.store.list_evaluations(self.model_id))
        if not n:
            message(self, "Nothing to delete", "There are no evaluation results.")
            return
        plan = DeletionPlan(title=f"Delete {n} evaluation result(s)?", items=["Saved evaluation reports"],
                            size_bytes=dir_size(self.ctx.store.evaluation_dir(self.model_id)),
                            warnings=["Validation numbers stored on each version are kept."])
        ok, _ = confirm_delete(self, plan, "Delete Results")
        if ok:
            self.ctx.store.delete_evaluations(self.model_id)
            self.refresh_detail()

    def _s_curve(self) -> None:
        sid = self.s_table.selected()
        if not sid:
            message(self, "Choose a session", "Select a training session first.")
            return
        png = self.ctx.store.sessions_dir(self.model_id) / f"{sid}_curve.png"
        if not png.exists():
            message(self, "No learning curve", "This session has no learning curve (it may have been deleted, or the "
                                               "session ended before any plays finished).")
            return
        osu_integration.open_file(png)

    def _s_delete_logs(self) -> None:
        sd = self.ctx.store.sessions_dir(self.model_id)
        logs = list(sd.glob("*_episodes.csv")) + list(sd.glob("*_curve.png"))
        if not logs:
            message(self, "Nothing to delete", "There are no training logs.")
            return
        plan = DeletionPlan(title="Delete training logs?", items=[f"{len(logs)} episode log / learning-curve file(s)"],
                            size_bytes=sum(f.stat().st_size for f in logs),
                            warnings=["Session summaries (used for lineage) and all versions are kept."])
        ok, _ = confirm_delete(self, plan, "Delete Logs")
        if ok:
            self.ctx.store.delete_logs(self.model_id)
            self.refresh_detail()


# ======================================================== version details
class VersionDetails(Dialog):
    def __init__(self, parent, ctx, model_id: str, version: str):
        super().__init__(parent, f"{version} details", width=760)
        self.ctx, self.model_id, self.version = ctx, model_id, version
        store = ctx.store
        v = store.get_version(model_id, version)
        m = v.meta
        ttk.Label(self.body, text=f"{v.model_meta.get('name')}  ·  {version}", style="CardH2.TLabel").grid(row=0, column=0, sticky="w")
        badge_row(self.body, v.badges()).grid(row=1, column=0, sticky="w", pady=(4, 0))
        chain = store.lineage(model_id, version)
        lineage = "  →  ".join(f"{name}{'' if ok else ' (deleted)'}" for name, ok in chain)
        kids = store.children(model_id, version)
        ttk.Label(self.body, text=f"Lineage: {lineage}" + (f"\nTrained from this: {', '.join(kids)}" if kids else ""),
                  style="CardMuted.TLabel", wraplength=700, justify="left").grid(row=2, column=0, sticky="w", pady=(8, 0))
        kv = KeyValueGrid(self.body, columns=2)
        kv.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        hp = m.get("hyperparameters") or {}
        env = m.get("env_config") or {}
        val = m.get("validation") or {}
        rows = [
            ("Parent version", m.get("parent_version") or "none (initial)"),
            ("Created", fmt_date(m.get("created_at"))),
            ("How it ended", m.get("status") or "—"),
            ("Source", m.get("source") or "—"),
            ("Trained this session", fmt_int(m.get("trained_timesteps"))),
            ("Total timesteps", fmt_int(m.get("total_timesteps"))),
            ("Training duration", fmt_duration(m.get("training_duration_s"))),
            ("Device", m.get("device") or "—"),
            ("Learning rate", f"{hp['learning_rate']:g}" if hp.get("learning_rate") else "—"),
            ("Batch size", fmt_int(hp.get("batch_size"))),
            ("Parallel games", fmt_int(hp.get("n_envs"))),
            ("Network", "x".join(map(str, (m.get("architecture") or {}).get("net_arch") or [])) or "—"),
            ("Decision tick / noise", f"{env.get('dt_ms', '—')} ms / {env.get('jitter_ms', '—')} ms"),
            ("Training maps", f"{len(m.get('training_maps') or [])} from {Path(m.get('maps_folder') or '—').name}"),
            ("Validation", f"{fmt_pct(val.get('accuracy'))} on {val.get('n_maps', len(m.get('validation_maps') or []))} "
                           f"{'held-out' if val.get('heldout') else 'training'} map(s)" if val else "not validated"),
            ("Final loss", f"{(m.get('final_loss') or {}).get('loss', 0):.4f}" if m.get("final_loss") else "—"),
            ("File size", human_size(m.get("file_size") or 0)),
            ("Source checkpoint", Path(m["source_checkpoint"]).name if m.get("source_checkpoint") else "—"),
        ]
        for i, (label, value) in enumerate(rows):
            kv.add(str(i), label, str(value))
        if val:
            mt = KeyValueGrid(self.body, columns=3)
            mt.grid(row=4, column=0, sticky="ew", pady=(12, 0))
            for key, label in (("misses", "Missed notes"), ("extra_presses", "Extra presses"),
                               ("hold_accuracy", "Holds completed"), ("mean_offset_ms", "Mean offset"),
                               ("timing_std_ms", "Timing deviation"), ("hit_rate", "Notes hit")):
                mt.add(key, label, fmt_value(key, val.get(key)))
        ttk.Label(self.body, text="Notes", style="CardMuted.TLabel").grid(row=5, column=0, sticky="w", pady=(12, 2))
        self.notes = tk.Text(self.body, height=3, bg=theme.SURFACE_2, fg=theme.TEXT, insertbackground=theme.TEXT,
                             relief="flat", wrap="word", font=theme.fonts()["body"])
        self.notes.insert("1.0", m.get("notes") or "")
        self.notes.grid(row=6, column=0, sticky="ew")
        self.buttons([("Open folder", "folder", "TButton"), ("Close", None, "TButton"),
                            ("Save notes", "save", "Accent.TButton")])

    def close(self, value) -> None:
        if value == "save":
            self.ctx.store.update_version(self.model_id, self.version, notes=self.notes.get("1.0", "end").strip())
        elif value == "folder":
            osu_integration.open_folder(self.ctx.store.version_dir(self.model_id, self.version))
            return
        super().close(value)


# ============================================================ evaluation
class EvaluateDialog(Dialog):
    """Evaluate one or more versions on the same maps (same seeds) and save the results."""

    def __init__(self, parent, ctx, targets: list[tuple[str, str]], on_done=None):
        super().__init__(parent, "Evaluate", width=620)
        self.ctx, self.targets, self.on_done = ctx, targets, on_done
        names = ", ".join(f"{ctx.store.get_model(m).name} {v}" for m, v in targets)
        ttk.Label(self.body, text=f"Evaluate {names}", style="CardH2.TLabel", wraplength=560).grid(row=0, column=0, sticky="w")
        ttk.Label(self.body, text="Plays every map in the folder once (full length) in the simulator and records "
                                  "accuracy, misses, extra presses, hold notes and timing. Not an osu! score.",
                  style="CardMuted.TLabel", wraplength=560, justify="left").grid(row=1, column=0, sticky="w", pady=(4, 10))
        self.maps = MapsFolderField(self.body, ctx, ctx.settings["default_maps"])
        self.maps.grid(row=2, column=0, sticky="ew")
        lim = ttk.Frame(self.body, style="Card.TFrame")
        lim.grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Label(lim, text="Use at most", style="Card.TLabel").pack(side="left")
        self.limit = tk.StringVar(value="12")
        ttk.Spinbox(lim, from_=1, to=500, textvariable=self.limit, width=6).pack(side="left", padx=6)
        ttk.Label(lim, text="maps", style="Card.TLabel").pack(side="left")
        self.status = ttk.Label(self.body, text="", style="CardMuted.TLabel", wraplength=560)
        self.status.grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.prog = ttk.Progressbar(self.body, maximum=1.0)
        self.buttons([("Close", None, "TButton"), ("Run evaluation", "run", "Accent.TButton")])
        self.results: list[dict] = []
        self._stop = False
        self.maps.scan()

    def close(self, value) -> None:
        if value == "run":
            self._run()
            return
        self._stop = True
        super().close(self.results or None)

    def _run(self) -> None:
        folder = self.maps.get()
        try:
            limit = max(1, int(self.limit.get()))
        except ValueError:
            limit = 12
        self._buttons["Run evaluation"].state(["disabled"])
        self.prog.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        total = len(self.targets)

        def progress(i, n, name, k):
            frac = (k + i / max(1, n)) / total
            self.ctx.tasks.post(lambda: (self.prog.configure(value=frac),
                                         self.status.configure(text=f"Playing {name} ({i + 1}/{n})…")))

        def work():
            scan = scan_maps(folder)
            maps = scan.beatmaps[:limit]
            if not maps:
                raise ModelError("No osu!mania 4K maps in that folder.")
            out = []
            for k, (mid, ver) in enumerate(self.targets):
                v = self.ctx.store.get_version(mid, ver)
                model = POLICIES.load(v.weights)
                res = strip_private(evaluate(model, maps, v.meta.get("env_config"), seed=777,
                                             progress=lambda i, n, name, k=k: progress(i, n, name, k),
                                             should_stop=lambda: self._stop))
                res.update({"kind": "evaluation", "maps_folder": folder})
                self.ctx.store.save_evaluation(mid, ver, res)
                out.append(res)
            return out

        def done(results):
            if not self.winfo_exists():
                return
            self.results = results
            self.prog.configure(value=1.0)
            txt = "\n".join(f"{self.ctx.store.get_model(m).name} {v}: accuracy {fmt_pct(r['accuracy'])}, "
                            f"{r['misses']} misses, {r['extra_presses']} extra presses on {r['n_maps']} map(s)"
                            for (m, v), r in zip(self.targets, results))
            self.status.configure(text=f"✓ Done. Results saved.\n{txt}", style="CardGood.TLabel")
            self._buttons["Run evaluation"].state(["!disabled"])
            self.ctx.emit("models_changed")
            if self.on_done:
                self.on_done(results)

        def fail(exc, tb):
            if self.winfo_exists():
                self.status.configure(text=str(exc), style="CardError.TLabel")
                self._buttons["Run evaluation"].state(["!disabled"])

        self.ctx.tasks.run(work, done, fail, name="evaluate")


# ============================================================== compare
class CompareDialog(Dialog):
    def __init__(self, parent, ctx, a: tuple[str, str], b: tuple[str, str]):
        super().__init__(parent, "Compare versions", width=820)
        self.ctx = ctx
        ttk.Label(self.body, text="Compare versions", style="CardH2.TLabel").grid(row=0, column=0, sticky="w")
        self.options = []
        for m in ctx.store.list_models():
            for v in ctx.store.list_versions(m.id):
                self.options.append((m.id, v.version, f"{m.name}  ·  {version_label(v)}"))
        labels = [o[2] for o in self.options]
        sel = ttk.Frame(self.body, style="Card.TFrame")
        sel.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.a, self.b = tk.StringVar(), tk.StringVar()
        for i, (var, key) in enumerate(((self.a, a), (self.b, b))):
            ttk.Label(sel, text="A" if i == 0 else "B", style="CardH3.TLabel").grid(row=i, column=0, padx=(0, 8), pady=3)
            cb = ttk.Combobox(sel, textvariable=var, values=labels, state="readonly", width=70)
            cb.grid(row=i, column=1, sticky="w", pady=3)
            idx = next((j for j, o in enumerate(self.options) if (o[0], o[1]) == key), 0)
            cb.current(idx)
            cb.bind("<<ComboboxSelected>>", lambda _e: self.render())
        src = ttk.Frame(self.body, style="Card.TFrame")
        src.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.source = tk.StringVar(value="latest")
        ttk.Radiobutton(src, text="Latest evaluation of each", value="latest", variable=self.source,
                        command=self.render).pack(side="left")
        ttk.Radiobutton(src, text="Validation from training", value="validation", variable=self.source,
                        command=self.render).pack(side="left", padx=12)
        ttk.Button(src, text="Run both on the same maps…", command=self.head_to_head).pack(side="left", padx=12)
        self.note = ttk.Label(self.body, text="", style="CardMuted.TLabel", wraplength=760, justify="left")
        self.note.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.table = Table(self.body, [("metric", "Metric", 230), ("a", "A", 200), ("b", "B", 200), ("better", "Better", 80)],
                           height=20)
        self.table.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        self.buttons([("Close", None, "Accent.TButton")])
        self.render()

    def _pick(self, var) -> tuple[str, str]:
        label = var.get()
        o = next(o for o in self.options if o[2] == label)
        return o[0], o[1]

    def _metrics(self, mid: str, ver: str) -> tuple[dict | None, str]:
        v = self.ctx.store.get_version(mid, ver)
        if self.source.get() == "validation":
            val = v.meta.get("validation")
            return val, f"validation on {len(v.meta.get('validation_maps') or [])} map(s)" if val else "no validation"
        evals = self.ctx.store.list_evaluations(mid, ver)
        if evals:
            e = evals[-1]
            return e, f"{e.get('kind', 'evaluation')} {fmt_date(e.get('created_at'))} on {e.get('n_maps', '?')} map(s)"
        val = v.meta.get("validation")
        return val, "validation (no separate evaluation yet)" if val else "never evaluated"

    def render(self) -> None:
        (ma, va), (mb, vb) = self._pick(self.a), self._pick(self.b)
        A, B = self.ctx.store.get_version(ma, va), self.ctx.store.get_version(mb, vb)
        ea, na = self._metrics(ma, va)
        eb, nb = self._metrics(mb, vb)
        same = ea and eb and [m.get("md5") for m in ea.get("maps", [])] == [m.get("md5") for m in eb.get("maps", [])]
        note = f"A: {na}.   B: {nb}."
        if ea and eb and not same:
            note += "\n⚠ The two results come from different maps, so they are not directly comparable. Use “Run both on the same maps”."
        self.note.configure(text=note)
        self.table.clear()
        cmp = compare(ea, eb)
        for i, r in enumerate(cmp.rows):
            better = {"a": "A", "b": "B"}.get(r["winner"], "")
            self.table.insert(f"m{i}", [r["label"], fmt_value(r["key"], r["a"]), fmt_value(r["key"], r["b"]), better],
                              tags=("good",) if better else ())
        info = [
            ("Total timesteps", A.meta.get("total_timesteps"), B.meta.get("total_timesteps")),
            ("Trained this session", A.meta.get("trained_timesteps"), B.meta.get("trained_timesteps")),
            ("Training duration", fmt_duration(A.meta.get("training_duration_s")), fmt_duration(B.meta.get("training_duration_s"))),
            ("Training maps", len(A.meta.get("training_maps") or []), len(B.meta.get("training_maps") or [])),
            ("Parent version", A.parent or "—", B.parent or "—"),
            ("Learning rate", (A.meta.get("hyperparameters") or {}).get("learning_rate"),
             (B.meta.get("hyperparameters") or {}).get("learning_rate")),
        ]
        for i, (label, x, y) in enumerate(info):
            self.table.insert(f"i{i}", [label, fmt_value("", x), fmt_value("", y), ""], tags=("muted",))

    def head_to_head(self) -> None:
        targets = [self._pick(self.a), self._pick(self.b)]
        if targets[0] == targets[1]:
            message(self, "Same version", "Choose two different versions.")
            return
        EvaluateDialog(self, self.ctx, targets).show()
        self.source.set("latest")
        self.render()
