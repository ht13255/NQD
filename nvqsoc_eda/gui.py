from __future__ import annotations

import json
import os
import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .ollama_runtime import ensure_ollama_ready, prefer_local_preloaded_model, query_ollama_models, resolve_ollama_auth
from .pipeline import run_pipeline
from .requirements import RequirementBundle, _resolve_ollama_endpoint, parse_design_requirements


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "examples" / "quantum_repeater.json"


class EdaGui:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("NV-Center QSoC EDA")
        self.root.geometry("1560x980")
        self.root.minsize(1280, 820)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.preview_image: ImageTk.PhotoImage | None = None
        self.requirements_bundle: RequirementBundle | None = None
        self.status_var = tk.StringVar(value="Ready")

        self.output_dir_var = tk.StringVar(value=str(ROOT / "outputs" / "gui_run"))
        self.seed_var = tk.StringVar(value="7")
        self.mc_var = tk.StringVar(value="96")
        self.generations_var = tk.StringVar(value="5")
        self.beam_var = tk.StringVar(value="6")
        self.mutations_var = tk.StringVar(value="5")
        self.paper_limit_var = tk.StringVar(value="5")
        self.advanced_topk_var = tk.StringVar(value="3")
        self.quality_var = tk.StringVar(value="extreme")
        self.ollama_model_var = tk.StringVar(value=os.getenv("NVQSOC_OLLAMA_MODEL", "llama3.1:8b"))
        self.ollama_endpoint_var = tk.StringVar(value=_resolve_ollama_endpoint(None))
        self.team_profile_var = tk.StringVar(value="")
        self.crawl_papers_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._load_default_spec()
        self.root.after(250, self._poll_events)
        threading.Thread(target=self._warmup_ollama, daemon=True).start()


    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Load Example", command=self._load_default_spec).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Open Spec", command=self._open_spec).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Save Spec", command=self._save_spec).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Browse Output", command=self._browse_output).pack(side="left", padx=4)
        self.run_button = ttk.Button(toolbar, text="Run EDA", command=self._run)
        self.run_button.pack(side="right", padx=4)

        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(paned, padding=8)
        right = ttk.Frame(paned, padding=8)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        ttk.Label(left, text="Spec JSON", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.spec_text = tk.Text(left, wrap="none", font=("Consolas", 10), undo=True)
        spec_scroll_y = ttk.Scrollbar(left, orient="vertical", command=self.spec_text.yview)
        spec_scroll_x = ttk.Scrollbar(left, orient="horizontal", command=self.spec_text.xview)
        self.spec_text.configure(yscrollcommand=spec_scroll_y.set, xscrollcommand=spec_scroll_x.set)
        self.spec_text.pack(fill="both", expand=True, side="left")
        spec_scroll_y.pack(fill="y", side="right")
        spec_scroll_x.pack(fill="x", side="bottom")

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)
        controls_tab = ttk.Frame(notebook, padding=10)
        requirements_tab = ttk.Frame(notebook, padding=10)
        summary_tab = ttk.Frame(notebook, padding=10)
        preview_tab = ttk.Frame(notebook, padding=10)
        log_tab = ttk.Frame(notebook, padding=10)
        notebook.add(controls_tab, text="Run")
        notebook.add(requirements_tab, text="Requirements")
        notebook.add(summary_tab, text="Summary")
        notebook.add(preview_tab, text="Preview")
        notebook.add(log_tab, text="Log")

        self._build_controls_tab(controls_tab)
        self._build_requirements_tab(requirements_tab)

        self.summary_text = tk.Text(summary_tab, wrap="word", font=("Consolas", 10), state="disabled")
        self.summary_text.pack(fill="both", expand=True)

        self.preview_label = ttk.Label(preview_tab, text="Run the EDA to see a layout preview.", anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_tab, wrap="word", font=("Consolas", 10), state="disabled")
        self.log_text.pack(fill="both", expand=True)

        status = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        status.pack(fill="x")
        self.progress = ttk.Progressbar(status, mode="indeterminate")
        self.progress.pack(side="right", padx=(8, 0))
        ttk.Label(status, textvariable=self.status_var).pack(side="left")


    def _build_controls_tab(self, parent: ttk.Frame) -> None:
        fields = [
            ("Output dir", self.output_dir_var),
            ("Seed", self.seed_var),
            ("MC trials", self.mc_var),
            ("Generations", self.generations_var),
            ("Beam width", self.beam_var),
            ("Mutations", self.mutations_var),
            ("Paper limit", self.paper_limit_var),
            ("Advanced top-k", self.advanced_topk_var),
            ("Quality mode", self.quality_var),
        ]
        for row, (label, var) in enumerate(fields):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
            entry = ttk.Entry(parent, textvariable=var, width=44)
            entry.grid(row=row, column=1, sticky="ew", pady=5)

        model_row = len(fields)
        ttk.Label(parent, text="Ollama model").grid(row=model_row, column=0, sticky="w", pady=5)
        self.ollama_model_combo = ttk.Combobox(
            parent,
            textvariable=self.ollama_model_var,
            width=42,
            state="normal",
            postcommand=self._refresh_ollama_models,
        )
        self.ollama_model_combo.grid(row=model_row, column=1, sticky="ew", pady=5)
        self.ollama_model_combo["values"] = [self.ollama_model_var.get().strip() or "llama3.1:8b"]
        self.ollama_model_combo.bind("<Button-1>", lambda _event: self._refresh_ollama_models())
        self.ollama_model_combo.bind("<FocusIn>", lambda _event: self._refresh_ollama_models())

        endpoint_row = model_row + 1
        ttk.Label(parent, text="Ollama endpoint").grid(row=endpoint_row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.ollama_endpoint_var, width=44).grid(row=endpoint_row, column=1, sticky="ew", pady=5)

        team_profile_row = endpoint_row + 1
        ttk.Label(parent, text="Team profile").grid(row=team_profile_row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.team_profile_var, width=44).grid(row=team_profile_row, column=1, sticky="ew", pady=5)

        parent.columnconfigure(1, weight=1)
        action_row = team_profile_row + 1
        ttk.Checkbutton(parent, text="Crawl papers with Scrapling", variable=self.crawl_papers_var).grid(row=action_row, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Button(parent, text="Open Output Folder", command=self._open_output_folder).grid(row=action_row + 1, column=0, sticky="w", pady=8)
        ttk.Button(parent, text="Run Pipeline", command=self._run).grid(row=action_row + 1, column=1, sticky="e", pady=8)


    def _build_requirements_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Natural-Language Requirements", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.requirements_text = tk.Text(parent, wrap="word", height=18, font=("Consolas", 10))
        self.requirements_text.pack(fill="both", expand=True, pady=(8, 8))
        self.requirements_text.insert(
            "1.0",
            "Design a fault-tolerant NV-center QSoC processor with around 256 physical qubits, accelerator-style parallel control/readout, strong logical-qubit support, realistic cryogenic routing, and dense but manufacturable layout. Keep latency under 180 ns and prioritize robustness against cross-talk.",
        )
        ttk.Label(parent, text="Team Notes / Spec Template Hints", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        self.team_notes_text = tk.Text(parent, wrap="word", height=6, font=("Consolas", 10))
        self.team_notes_text.pack(fill="x", expand=False, pady=(6, 8))
        self.team_notes_text.insert(
            "1.0",
            "Korean/English mixed terminology is common. Treat pad ring, shield fence, redundant optical trunk, cryo control, via farm, fault-tolerant logical patch, and parallel accelerator fabric as mandatory realism hints when mentioned.",
        )
        action_bar = ttk.Frame(parent)
        action_bar.pack(fill="x")
        self.parse_button = ttk.Button(action_bar, text="Parse With Ollama", command=self._parse_requirements)
        self.parse_button.pack(side="left")
        ttk.Button(action_bar, text="Browse Team Profile", command=self._browse_team_profile).pack(side="left", padx=8)
        ttk.Label(action_bar, text="The parser falls back to heuristic decomposition if Ollama is unavailable.").pack(side="left", padx=10)


    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


    def _warmup_ollama(self) -> None:
        endpoint = self.ollama_endpoint_var.get().strip() or _resolve_ollama_endpoint(None)
        model = self.ollama_model_var.get().strip() or "llama3.1:8b"
        auth = resolve_ollama_auth()
        status = ensure_ollama_ready(
            model,
            endpoint,
            timeout_s=20,
            auth_header=auth.auth_header,
            auth_mode=auth.mode,
            auto_pull=not prefer_local_preloaded_model(auth.mode, endpoint),
        )
        self.events.put(("ollama_status", status))


    def _refresh_ollama_models(self) -> None:
        endpoint = self.ollama_endpoint_var.get().strip() or _resolve_ollama_endpoint(None)
        auth = resolve_ollama_auth()
        models = query_ollama_models(endpoint, auth_header=auth.auth_header, timeout_s=6) or []
        current = self.ollama_model_var.get().strip() or "llama3.1:8b"
        merged = []
        for item in models + [current]:
            if item and item not in merged:
                merged.append(item)
        if not merged:
            merged = ["llama3.1:8b"]
        if hasattr(self, "ollama_model_combo"):
            self.ollama_model_combo["values"] = merged


    def _set_summary(self, data: dict[str, object]) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", json.dumps(data, indent=2, ensure_ascii=False))
        self.summary_text.configure(state="disabled")


    def _load_default_spec(self) -> None:
        if DEFAULT_SPEC.exists():
            self.requirements_bundle = None
            self.spec_text.delete("1.0", "end")
            self.spec_text.insert("1.0", DEFAULT_SPEC.read_text(encoding="utf-8"))
            self.status_var.set(f"Loaded {DEFAULT_SPEC}")


    def _open_spec(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if not path:
            return
        self.requirements_bundle = None
        self.spec_text.delete("1.0", "end")
        self.spec_text.insert("1.0", Path(path).read_text(encoding="utf-8"))
        self.status_var.set(f"Opened {path}")


    def _save_spec(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not path:
            return
        Path(path).write_text(self.spec_text.get("1.0", "end").strip() + "\n", encoding="utf-8")
        self.status_var.set(f"Saved {path}")


    def _browse_output(self) -> None:
        path = filedialog.askdirectory(initialdir=self.output_dir_var.get() or str(ROOT))
        if path:
            self.output_dir_var.set(path)


    def _browse_team_profile(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if path:
            self.team_profile_var.set(path)


    def _open_output_folder(self) -> None:
        output_dir = Path(self.output_dir_var.get())
        if output_dir.exists():
            os.startfile(str(output_dir))
        else:
            messagebox.showinfo("Output", "Output directory does not exist yet.")


    def _run(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("EDA", "A run is already in progress.")
            return
        try:
            spec_data = json.loads(self.spec_text.get("1.0", "end"))
            json.dumps(spec_data)
        except Exception as exc:
            messagebox.showerror("Invalid spec", str(exc))
            return

        output_dir = Path(self.output_dir_var.get())
        output_dir.mkdir(parents=True, exist_ok=True)
        spec_path = output_dir / "gui_spec.json"
        spec_path.write_text(json.dumps(spec_data, indent=2, ensure_ascii=False), encoding="utf-8")

        try:
            params = {
                "spec_path": spec_path,
                "output_dir": output_dir,
                "seed": int(self.seed_var.get()),
                "monte_carlo_trials": int(self.mc_var.get()),
                "generations": int(self.generations_var.get()),
                "beam_width": int(self.beam_var.get()),
                "mutations_per_parent": int(self.mutations_var.get()),
                "crawl_papers": self.crawl_papers_var.get(),
                "paper_limit": int(self.paper_limit_var.get()),
                "advanced_top_k": int(self.advanced_topk_var.get()),
                "quality_mode": self.quality_var.get().strip() or "extreme",
                "requirements_bundle": self.requirements_bundle,
            }
        except ValueError as exc:
            messagebox.showerror("Invalid option", str(exc))
            return
        self._append_log(f"Starting run with output `{output_dir}`")
        self.status_var.set("Running EDA...")
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.worker = threading.Thread(target=self._worker_run, args=(params,), daemon=True)
        self.worker.start()


    def _parse_requirements(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("EDA", "Wait for the current task to finish first.")
            return
        requirements = self.requirements_text.get("1.0", "end").strip()
        if not requirements:
            messagebox.showerror("Requirements", "Enter natural-language requirements first.")
            return
        model = self.ollama_model_var.get().strip() or "llama3.1:8b"
        ollama_endpoint = self.ollama_endpoint_var.get().strip() or None
        team_notes = self.team_notes_text.get("1.0", "end").strip()
        team_profile = self.team_profile_var.get().strip() or None
        self.status_var.set("Parsing requirements...")
        self.parse_button.configure(state="disabled")
        self.progress.start(12)
        self._append_log(f"Parsing requirements with model `{model}` via endpoint `{ollama_endpoint or 'default'}`")
        self.worker = threading.Thread(
            target=self._worker_parse,
            args=(requirements, model, team_profile, team_notes, ollama_endpoint),
            daemon=True,
        )
        self.worker.start()


    def _worker_run(self, params: dict[str, object]) -> None:
        try:
            summary = run_pipeline(**params)
            self.events.put(("done", summary))
        except Exception:
            self.events.put(("error", traceback.format_exc()))


    def _worker_parse(
        self,
        requirements: str,
        model: str,
        team_profile: str | None,
        team_notes: str,
        ollama_endpoint: str | None,
    ) -> None:
        try:
            bundle = parse_design_requirements(
                requirements,
                model=model,
                team_profile_path=team_profile,
                team_notes=team_notes,
                ollama_endpoint=ollama_endpoint,
            )
            self.events.put(("parse_done", bundle))
        except Exception:
            self.events.put(("error", traceback.format_exc()))


    def _show_preview(self, preview_path: str) -> None:
        path = Path(preview_path)
        if not path.exists():
            self.preview_label.configure(text="Preview image not found.", image="")
            return
        image = Image.open(path)
        max_w, max_h = 720, 620
        image.thumbnail((max_w, max_h))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")


    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "done":
                    summary = payload  # type: ignore[assignment]
                    self._set_summary(summary)
                    self._append_log("Run finished successfully.")
                    preview = summary.get("artifacts", {}).get("layout_preview_png")  # type: ignore[union-attr]
                    if isinstance(preview, str):
                        self._show_preview(preview)
                    self.status_var.set("EDA completed")
                    self.run_button.configure(state="normal")
                    self.parse_button.configure(state="normal")
                    self.progress.stop()
                    self.worker = None
                elif event == "parse_done":
                    bundle = payload  # type: ignore[assignment]
                    if isinstance(bundle, RequirementBundle):
                        self.requirements_bundle = bundle
                        self.spec_text.delete("1.0", "end")
                        self.spec_text.insert("1.0", json.dumps(bundle.normalized_spec, indent=2, ensure_ascii=False))
                        self._append_log(f"Parsed requirements via `{bundle.source}` using model `{bundle.model}`")
                        if bundle.ollama_started_local_service:
                            self._append_log("Started local Ollama service automatically.")
                        if bundle.ollama_model_pulled:
                            self._append_log("Pulled the requested Ollama model automatically.")
                        self._append_log(
                            f"Ollama runtime ready=`{bundle.ollama_runtime_ready}` model_available=`{bundle.ollama_model_available}` model_loaded=`{bundle.ollama_model_loaded}` auth_mode=`{bundle.ollama_auth_mode}` endpoint=`{bundle.ollama_endpoint}`"
                        )
                        self._append_log(f"Goals: {bundle.goals}")
                        self._append_log(f"Constraints: {bundle.constraints}")
                        self.status_var.set("Requirements parsed")
                    self.parse_button.configure(state="normal")
                    self.run_button.configure(state="normal")
                    self.progress.stop()
                    self.worker = None
                elif event == "ollama_status":
                    status = payload
                    if hasattr(status, "available"):
                        if status.available:
                            detail = "Ollama ready"
                            if status.started_local_service:
                                detail += " (local service auto-started)"
                            if status.model_pulled:
                                detail += " (model auto-pulled)"
                            if status.model_loaded:
                                detail += " (model preloaded)"
                            if getattr(status, "models", None):
                                if hasattr(self, "ollama_model_combo"):
                                    current = self.ollama_model_var.get().strip() or "llama3.1:8b"
                                    merged = []
                                    for item in list(status.models) + [current]:
                                        if item and item not in merged:
                                            merged.append(item)
                                    self.ollama_model_combo["values"] = merged
                            self._append_log(f"{detail}: endpoint `{status.endpoint}`")
                            self.status_var.set("Ollama ready")
                        else:
                            self._append_log(f"Ollama not ready at `{status.endpoint}`; parser will fall back heuristically.")
                elif event == "error":
                    self._append_log(str(payload))
                    self.status_var.set("EDA failed")
                    self.run_button.configure(state="normal")
                    self.parse_button.configure(state="normal")
                    self.progress.stop()
                    self.worker = None
                    messagebox.showerror("EDA failed", str(payload))
        except queue.Empty:
            pass
        self.root.after(250, self._poll_events)


    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    EdaGui().run()


if __name__ == "__main__":
    main()
