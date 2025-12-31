#!/usr/bin/env python3
"""
GUI worker-pool dispatcher for netcoevolve runs (Tkinter).

Everything is configured via the GUI: binary path, output folder, jobs, repeats,
list of n values, eta range, launch gap, and fixed simulation parameters.

Features:
 - Maintains up to N concurrent processes.
 - Iterates over (n, eta) combos and repeats each combo.
 - Captures each child process in its own PTY so indicatif progress bars render.
 - Shows a stacked, live-updating view (one line per active process) plus overall progress.
 - Writes outputs to selected folder with unique per-run filenames.
"""
from __future__ import annotations

import itertools
import glob
from collections import defaultdict
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional
import re

import pty
import select
import tempfile

# ANSI sanitizer (available early for threads)
ANSI_CSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
ANSI_OSC_RE = re.compile(r"\x1B\][^\x07]*\x07")

def strip_ansi(s: str) -> str:
    s = ANSI_OSC_RE.sub("", s)
    s = ANSI_CSI_RE.sub("", s)
    return s

# CSV helpers for Load/Validate
HEADER_KV_RE = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([^,\n\r#]+)")  # kept for reference, not used below

def parse_header_params_from_file(path: str) -> Dict[str, str]:
    """Parse header key=value pairs from initial '#' lines (BOM/whitespace safe).
    Tokenizes on whitespace like polarisation.py instead of regex scanning.
    """
    params: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            found_header = False
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                s = line.lstrip("\ufeff").lstrip()
                if not s:
                    continue
                if not s.startswith("#"):
                    # stop at first non-header once we've seen a header
                    if found_header:
                        break
                    else:
                        break
                found_header = True
                content = s[1:].strip()
                for tok in content.split():
                    if "=" not in tok:
                        continue
                    k, v = tok.split("=", 1)
                    params[k.strip().lower()] = v.strip()
    except Exception:
        pass
    return params

def read_last_data_time(path: str) -> Optional[float]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # Read last ~8KB
            offset = max(0, size - 8192)
            f.seek(offset)
            data = f.read().decode("utf-8", errors="ignore")
        lines = [ln for ln in data.splitlines() if ln and not ln.lstrip().startswith("#")]
        if not lines:
            return None
        last = lines[-1]
        first_field = last.split(",", 1)[0].strip()
        return float(first_field)
    except Exception:
        return None

def is_simulation_csv(path: str) -> bool:
    """A CSV is considered a simulation result if the first non-empty line
    starts with '# netcoevolve=' or '# n='"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                s = line.lstrip("\ufeff").strip()
                if not s:
                    continue
                if s.startswith("# netcoevolve=") or s.startswith("# n="):
                    return True
                return False
    except Exception:
        return False
    return False

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except Exception as e:
    print("Tkinter not available:", e, file=sys.stderr)
    sys.exit(1)


def frange(start: float, end: float, step: float) -> List[float]:
    vals: List[float] = []
    k = 0
    while True:
        v = start + k * step
        if step > 0 and v > end + 1e-12:
            break
        if step < 0 and v < end - 1e-12:
            break
        vals.append(round(v, 10))
        k += 1
    return vals


def build_tasks(config: Dict) -> List[Dict]:
    etas = frange(config["eta_start"], config["eta_end"], config["eta_step"])
    tasks: List[Dict] = []
    for eta, n in itertools.product(etas, config["ns"]):
        for _ in range(config["repeats"]):
            tasks.append({"n": n, "eta": eta})
    return tasks


def join_cmd(cmd: List[str]) -> str:
    return " ".join(cmd)


class DispatcherThread(threading.Thread):
    def __init__(self, config: Dict, tasks: List[Dict], q: "queue.Queue[tuple]") -> None:
        super().__init__(daemon=True)
        self.config = config
        self.tasks = tasks
        self.q = q
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.active: List[Dict] = []  # entries: {id, p, fd, buf, last, cmd}
        self.next_id = 1
        self.done = 0
        self.last_launch = 0.0
        self.counter = 0
        self._paused: set[int] = set()

    def stop(self) -> None:
        self.stop_event.set()

    def cleanup(self) -> None:
        # terminate and close fds
        for e in self.active:
            p = e["p"]
            try:
                p.terminate()
            except Exception:
                pass
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if all(e["p"].poll() is not None for e in self.active):
                break
            time.sleep(0.2)
        for e in self.active:
            if e["p"].poll() is None:
                try:
                    e["p"].kill()
                except Exception:
                    pass
        for e in self.active:
            try:
                os.close(e["fd"])
            except Exception:
                pass

    def pause(self) -> None:
        self.pause_event.set()
        # Suspend currently running children (best effort, POSIX only)
        for e in list(self.active):
            p: subprocess.Popen = e["p"]
            try:
                os.kill(p.pid, signal.SIGSTOP)
                self._paused.add(p.pid)
            except Exception:
                pass

    def resume(self) -> None:
        # Resume children
        for e in list(self.active):
            p: subprocess.Popen = e["p"]
            if p.pid in self._paused:
                try:
                    os.kill(p.pid, signal.SIGCONT)
                except Exception:
                    pass
        self._paused.clear()
        self.pause_event.clear()

    def _unique_output_path(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        # unique name with timestamp and counter
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.counter += 1
        return os.path.join(out_dir, f"simulation-{ts}-{self.counter:04d}.csv")

    def start_one(self, n: int, eta: float) -> None:
        # enforce launch gap
        now = time.time()
        gap = self.config["launch_gap"] - (now - self.last_launch)
        if gap > 0:
            time.sleep(gap)
        mfd, sfd = pty.openpty()
        env = os.environ.copy()
        env.setdefault("NO_COLOR", "1")
        cmd = [
            self.config["bin"],
            "--n", str(n),
        ]
        # When rho = n, we now treat the sweep variable (eta) as beta and *do not* send --rho or --eta
        # This matches the new binary interface expectation.
        if self.config.get("rho_equals_n", True):
            # Previously: we passed --rho n and --eta <val>. Now we just pass --beta <val>.
            cmd += ["--beta", str(eta)]
        else:
            # Rho fixed (not equal to n): pass the fixed rho and still treat eta as eta
            rho_val = self.config.get("rho")
            if rho_val is not None:
                cmd += ["--rho", str(rho_val)]
            cmd += ["--eta", str(eta)]
        # Fixed params
        cmd += [
            "--sd0", str(self.config["sd0"]),
            "--sd1", str(self.config["sd1"]),
            "--sc0", str(self.config["sc0"]),
            "--sc1", str(self.config["sc1"]),
            "--sample_delta", str(self.config["sample_delta"]),
            "--t_max", str(self.config["t_max"]),
        ]
        seed_val = str(self.config.get("seed", "")).strip()
        # Always pass --seed when provided, including the literal "random";
        # omitting it makes the binary use its default (42).
        if seed_val:
            cmd += ["--seed", seed_val]
        if self.config.get("dump_adj"):
            cmd.append("--dump_adj")
        # Optional probability parameters if provided
        for key in ("p1", "p00", "p01", "p11"):
            v = self.config.get(key)
            if v is not None and str(v) != "":
                cmd += [f"--{key}", str(v)]
        # Launch in working directory so binary writes into work_dir/output/
        cwd = self.config.get("work_dir") or os.getcwd()
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=sfd,
            stderr=sfd,
            close_fds=True,
            env=env,
            cwd=cwd,
        )
        os.close(sfd)
        jid = self.next_id
        self.next_id += 1
        entry = {"id": jid, "p": p, "fd": mfd, "buf": "", "last": "", "cmd": cmd}
        self.active.append(entry)
        self.last_launch = time.time()
        self.q.put(("start", jid, cmd))

    def run(self) -> None:
        # Validate binary exists
        if not (os.path.isfile(self.config["bin"]) and os.access(self.config["bin"], os.X_OK)):
            self.q.put(("error", f"Binary not found or not executable: {self.config['bin']}"))
            return
        total = len(self.tasks)
        self.q.put(("meta", {"total": total}))
        it = iter(self.tasks)
        try:
            while not self.stop_event.is_set():
                # if paused, don't launch new tasks; still poll I/O so UI can update
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                # launch up to capacity
                while len(self.active) < self.config["jobs"] and not self.stop_event.is_set():
                    if self.pause_event.is_set():
                        break
                    try:
                        t = next(it)
                    except StopIteration:
                        break
                    self.start_one(t["n"], t["eta"])
                if not self.active:
                    # nothing active and no more tasks
                    break
                # multiplex reads
                fds = [e["fd"] for e in self.active]
                try:
                    rlist, _, _ = select.select(fds, [], [], 0.2)
                except Exception:
                    rlist = []
                for fd in rlist:
                    try:
                        chunk = os.read(fd, 4096)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        continue
                    text = chunk.decode("utf-8", errors="replace")
                    for e in self.active:
                        if e["fd"] == fd:
                            e["buf"] += text
                            buf = e["buf"]
                            if len(buf) > 4000:
                                buf = buf[-2000:]
                            li = max(buf.rfind("\r"), buf.rfind("\n"))
                            if li != -1 and li + 1 < len(buf):
                                e["last"] = buf[li + 1 :].strip()
                                # Sanitize ANSI before sending to UI
                                self.q.put(("update", e["id"], strip_ansi(e["last"])) )
                            e["buf"] = buf
                            break
                # prune finished
                i = 0
                changed = False
                while i < len(self.active):
                    e = self.active[i]
                    ret = e["p"].poll()
                    if ret is None:
                        i += 1
                        continue
                    try:
                        os.close(e["fd"])
                    except Exception:
                        pass
                    self.active.pop(i)
                    self.done += 1
                    changed = True
                    self.q.put(("finish", e["id"], ret))
                if changed:
                    self.q.put(("progress", {"done": self.done, "active": len(self.active)}))
        finally:
            # attempt to clean up lingering fds; processes should be finished
            for e in self.active:
                try:
                    os.close(e["fd"])
                except Exception:
                    pass
            self.q.put(("done", None))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("netcoevolve batch - Dashboard")
        self.geometry("1100x700")

        # Top: overall progress
        self.meta_total = 0
        self.meta_done = 0
        self.meta_active = 0

        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=10)
        self.progress = ttk.Progressbar(top, orient=tk.HORIZONTAL, mode="determinate")
        self.progress.pack(fill=tk.X)
        self.status_label = ttk.Label(top, text="")
        self.status_label.pack(anchor=tk.W, pady=(6, 0))
        # Schedule summary in main window
        self.schedule_main_label = ttk.Label(top, text="Schedule: (not set)")
        self.schedule_main_label.pack(anchor=tk.W, pady=(4, 0))

        # Parameters frame (top controls)
        params = ttk.LabelFrame(self, text="Parameters")
        params.pack(fill=tk.X, padx=10, pady=10)

        # Row 1: binary and output dir
        self.bin_var = tk.StringVar(value="./target/release/netcoevolve")
        self.out_var = tk.StringVar(value="output")
        ttk.Label(params, text="Binary:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        self.bin_entry = ttk.Entry(params, textvariable=self.bin_var, width=70)
        self.bin_entry.grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Button(params, text="Browse", command=self.browse_bin).grid(row=0, column=2, padx=4, pady=4)
        ttk.Label(params, text="Output folder:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        self.out_entry = ttk.Entry(params, textvariable=self.out_var, width=70)
        self.out_entry.grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Button(params, text="Browse", command=self.browse_out).grid(row=1, column=2, padx=4, pady=4)

        # Row 2: jobs, repeats, launch gap
        self.jobs_var = tk.IntVar(value=6)
        self.repeats_var = tk.IntVar(value=6)
        self.gap_var = tk.DoubleVar(value=1.1)
        ttk.Label(params, text="Jobs:").grid(row=2, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox(params, from_=1, to=64, textvariable=self.jobs_var, width=6).grid(row=2, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(params, text="Repeats:").grid(row=2, column=1, sticky=tk.W, padx=(120,4), pady=4)
        ttk.Spinbox(params, from_=1, to=100, textvariable=self.repeats_var, width=6).grid(row=2, column=1, sticky=tk.W, padx=(200,4), pady=4)
        ttk.Label(params, text="Launch gap (s):").grid(row=2, column=1, sticky=tk.W, padx=(320,4), pady=4)
        ttk.Entry(params, textvariable=self.gap_var, width=8).grid(row=2, column=1, sticky=tk.W, padx=(430,4), pady=4)

        # Row 3: Ns
        self.ns_var = tk.StringVar(value="500 1000 1500 2000 2500")
        self.eta_start_var = tk.DoubleVar(value=5.00)
        self.eta_end_var = tk.DoubleVar(value=9.00)
        self.eta_step_var = tk.DoubleVar(value=0.05)
        ttk.Label(params, text="Ns (space/comma sep):").grid(row=3, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(params, textvariable=self.ns_var, width=40).grid(row=3, column=1, sticky=tk.W, padx=4, pady=4)
        # Row 4: Eta range (on its own row to avoid overlap)
        ttk.Label(params, text="Eta start/end/step:").grid(row=4, column=0, sticky=tk.W, padx=4, pady=4)
        frame_eta = ttk.Frame(params)
        frame_eta.grid(row=4, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(frame_eta, textvariable=self.eta_start_var, width=8).pack(side=tk.LEFT)
        ttk.Entry(frame_eta, textvariable=self.eta_end_var, width=8).pack(side=tk.LEFT, padx=(6,0))
        ttk.Entry(frame_eta, textvariable=self.eta_step_var, width=8).pack(side=tk.LEFT, padx=(6,0))

        # Row 5: Fixed params
        self.sd0_var = tk.DoubleVar(value=0.0)
        self.sd1_var = tk.DoubleVar(value=1.0)
        self.sc0_var = tk.DoubleVar(value=1.0)
        self.sc1_var = tk.DoubleVar(value=0.0)
        self.sample_delta_var = tk.DoubleVar(value=0.002)
        self.t_max_var = tk.DoubleVar(value=2.0)
        self.seed_var = tk.StringVar(value="random")
        fixed = ttk.Frame(params)
        fixed.grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=4, pady=4)
        for label, var in [
            ("sd0", self.sd0_var), ("sd1", self.sd1_var), ("sc0", self.sc0_var), ("sc1", self.sc1_var),
            ("sample_delta", self.sample_delta_var), ("t_max", self.t_max_var), ("seed", self.seed_var)
        ]:
            ttk.Label(fixed, text=f"{label}:").pack(side=tk.LEFT, padx=(0,2))
            ttk.Entry(fixed, textvariable=var, width=8 if label != "seed" else 10).pack(side=tk.LEFT, padx=(0,10))

        # Row 6: Rho and probability parameters
        self.rho_equals_n_var = tk.BooleanVar(value=True)
        self.rho_value_var = tk.DoubleVar(value=0.0)
        self.dump_adj_var = tk.BooleanVar(value=False)
        rho_frame = ttk.Frame(params)
        rho_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=4, pady=4)
        ttk.Checkbutton(rho_frame, text="rho = n", variable=self.rho_equals_n_var).pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(rho_frame, text="rho (when not n):").pack(side=tk.LEFT)
        ttk.Entry(rho_frame, textvariable=self.rho_value_var, width=8).pack(side=tk.LEFT, padx=(4,10))
        ttk.Checkbutton(rho_frame, text="Dump Adj", variable=self.dump_adj_var).pack(side=tk.LEFT, padx=(0,10))
        # p-parameters (optional)
        self.p1_var = tk.StringVar(value="")
        self.p00_var = tk.StringVar(value="")
        self.p01_var = tk.StringVar(value="")
        self.p11_var = tk.StringVar(value="")
        for label, var in [("p1", self.p1_var), ("p00", self.p00_var), ("p01", self.p01_var), ("p11", self.p11_var)]:
            ttk.Label(rho_frame, text=f"{label}:").pack(side=tk.LEFT, padx=(0,2))
            ttk.Entry(rho_frame, textvariable=var, width=6).pack(side=tk.LEFT, padx=(0,10))

        # Middle: scrollable list of active jobs
        mid = ttk.Frame(self)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas = tk.Canvas(mid, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(mid, orient="vertical", command=self.canvas.yview)
        self.scrollframe = ttk.Frame(self.canvas)
        self.scrollframe.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollframe, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Dictionary job id -> (frame, label_title, label_line)
        self.rows = {}

        # Bottom: controls
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=10, pady=10)
        self.start_btn = ttk.Button(bottom, text="Start", command=self.on_start)
        self.start_btn.pack(side=tk.RIGHT, padx=(0,8))
        self.pause_btn = ttk.Button(bottom, text="Pause", command=self.on_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.RIGHT, padx=(0,8))
        self.resume_btn = ttk.Button(bottom, text="Resume", command=self.on_resume, state=tk.DISABLED)
        self.resume_btn.pack(side=tk.RIGHT, padx=(0,8))
        self.stop_btn = ttk.Button(bottom, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT)
        self.load_btn = ttk.Button(bottom, text="Load", command=self.on_load)
        self.load_btn.pack(side=tk.LEFT)
        self.schedule_btn = ttk.Button(bottom, text="Schedule", command=self.on_schedule)
        self.schedule_btn.pack(side=tk.LEFT, padx=(8,0))

        # Worker thread and queue
        self.q = queue.Queue()
        self.worker = None
        # Scheduling state
        self.schedule_viewed = False
        self.latest_counts = {}  # (n, rounded_eta) -> done count
        self.latest_expected_total: Optional[int] = None
        self.latest_done_ok: Optional[int] = None
        self.done_offset: int = 0
        # Live schedule window + model
        self.schedule_win: Optional[tk.Toplevel] = None
        self.schedule_tree: Optional[ttk.Treeview] = None
        self.schedule_summary: Optional[ttk.Label] = None
        self.schedule_model = None  # dict with ns, etas, repeats, counts_before, active_counts, done_session, rows
        self.job_combo: Dict[int, tuple] = {}

        # Signals: tie SIGINT to stopping as well (best-effort)
        try:
            signal.signal(signal.SIGINT, lambda s, f: self.on_stop())
            signal.signal(signal.SIGTERM, lambda s, f: self.on_stop())
        except Exception:
            pass

        self.after(50, self.poll_queue)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Require viewing schedule before starting
        self.start_btn.configure(state=tk.DISABLED)
        # Invalidate schedule when core inputs change
        self.ns_var.trace_add("write", lambda *args: self.invalidate_schedule())
        self.eta_start_var.trace_add("write", lambda *args: self.invalidate_schedule())
        self.eta_end_var.trace_add("write", lambda *args: self.invalidate_schedule())
        self.eta_step_var.trace_add("write", lambda *args: self.invalidate_schedule())
        self.repeats_var.trace_add("write", lambda *args: self.invalidate_schedule())

    def browse_bin(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Executable", "*"), ("All files", "*")])
        if path:
            self.bin_var.set(path)

    def browse_out(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.out_var.set(path)

    def on_start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        # Enforce that the user pressed Schedule before starting
        if not self.schedule_viewed:
            messagebox.showwarning("Schedule required", "Please press 'Schedule' to review the plan before starting.")
            return
        # Validate inputs
        bin_path_in = self.bin_var.get().strip()
        # Resolve binary path relative to repo root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        bin_path = bin_path_in
        if not os.path.isabs(bin_path):
            bin_path = os.path.abspath(os.path.join(repo_root, bin_path))
        if not (os.path.isfile(bin_path) and os.access(bin_path, os.X_OK)):
            messagebox.showerror("Error", f"Binary not found or not executable: {bin_path_in}\nResolved: {bin_path}")
            return
        out_dir = self.out_var.get().strip()
        # Create session working dir with 'output' symlink -> selected folder
        try:
            if not out_dir:
                out_dir = os.path.join(repo_root, "output")
            out_dir = os.path.abspath(os.path.expanduser(out_dir))
            os.makedirs(out_dir, exist_ok=True)
            session_dir = tempfile.mkdtemp(prefix="netcoevolve-")
            link_path = os.path.join(session_dir, "output")
            try:
                os.symlink(out_dir, link_path)
            except FileExistsError:
                pass
            work_dir = session_dir
        except Exception as e:
            messagebox.showerror("Error", f"Cannot prepare output folder: {out_dir}\n{e}")
            return
        # Parse Ns
        try:
            ns_tokens = [t for t in self.ns_var.get().replace(",", " ").split() if t]
            ns_list = [int(t) for t in ns_tokens]
        except Exception:
            messagebox.showerror("Error", "Invalid Ns list. Provide integers separated by space or comma.")
            return
        # Build config
        config = {
            "bin": bin_path,
            "work_dir": work_dir,
            "jobs": max(1, int(self.jobs_var.get())),
            "repeats": max(1, int(self.repeats_var.get())),
            "ns": ns_list,
            "eta_start": float(self.eta_start_var.get()),
            "eta_end": float(self.eta_end_var.get()),
            "eta_step": float(self.eta_step_var.get()),
            "launch_gap": float(self.gap_var.get()),
            "sd0": float(self.sd0_var.get()),
            "sd1": float(self.sd1_var.get()),
            "sc0": float(self.sc0_var.get()),
            "sc1": float(self.sc1_var.get()),
            "sample_delta": float(self.sample_delta_var.get()),
            "t_max": float(self.t_max_var.get()),
            "seed": self.seed_var.get(),
            "rho_equals_n": bool(self.rho_equals_n_var.get()),
            "dump_adj": bool(self.dump_adj_var.get()),
        }
        if not config["rho_equals_n"]:
            try:
                config["rho"] = float(self.rho_value_var.get())
            except Exception:
                messagebox.showerror("Error", "Invalid rho value.")
                return
        # Optional p-parameters
        for key, var in [("p1", self.p1_var), ("p00", self.p00_var), ("p01", self.p01_var), ("p11", self.p11_var)]:
            val = var.get().strip()
            if val != "":
                try:
                    config[key] = float(val)
                except Exception:
                    messagebox.showerror("Error", f"Invalid value for {key}.")
                    return
        tasks = build_tasks(config)
        # If we have counts from a Load, reduce tasks to remaining only
        if self.latest_counts:
            remaining_tasks: List[Dict] = []
            etas = frange(config["eta_start"], config["eta_end"], config["eta_step"])
            for n in config["ns"]:
                for eta in etas:
                    key = (n, round(eta, 10))
                    done_count = int(self.latest_counts.get(key, 0))
                    rem = max(0, config["repeats"] - done_count)
                    for _ in range(rem):
                        remaining_tasks.append({"n": n, "eta": eta})
            tasks = remaining_tasks
        # Reset UI status
        self.rows.clear()
        for child in self.scrollframe.winfo_children():
            child.destroy()
        # If we loaded earlier, preserve done count and expected total; otherwise show based on tasks
        if self.latest_expected_total is not None and self.latest_done_ok is not None:
            self.done_offset = int(self.latest_done_ok)
            self.set_meta(done=self.done_offset, total=int(self.latest_expected_total), active=0)
        else:
            self.done_offset = 0
            self.set_meta(done=0, total=len(tasks), active=0)
        # Start worker
        self.q = queue.Queue()
        self.worker = DispatcherThread(config, tasks, self.q)
        self.worker.start()
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        # Enable pause, disable resume initially
        try:
            self.pause_btn.configure(state=tk.NORMAL)
            self.resume_btn.configure(state=tk.DISABLED)
        except Exception:
            pass
        # remember session dir for cleanup
        self._session_dir = work_dir

    def on_stop(self) -> None:
        if self.worker and self.worker.is_alive():
            self.worker.stop()
            self.worker.cleanup()
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        try:
            self.pause_btn.configure(state=tk.DISABLED)
            self.resume_btn.configure(state=tk.DISABLED)
        except Exception:
            pass
        # Cleanup session work dir
        try:
            if hasattr(self, "_session_dir") and self._session_dir and os.path.isdir(self._session_dir):
                shutil.rmtree(self._session_dir, ignore_errors=True)
        except Exception:
            pass

    def on_close(self) -> None:
        # Called when window is closed via titlebar
        try:
            self.on_stop()
        finally:
            try:
                self.destroy()
            except Exception:
                pass

    def on_load(self) -> None:
        # Validate existing CSVs in output folder against current parameters
        out_dir = self.out_var.get().strip()
        if not out_dir:
            messagebox.showerror("Load", "Please choose an output folder first.")
            return
        out_dir = os.path.abspath(os.path.expanduser(out_dir))
        if not os.path.isdir(out_dir):
            messagebox.showerror("Load", f"Folder does not exist: {out_dir}")
            return
        # Expected combos
        try:
            ns_tokens = [t for t in self.ns_var.get().replace(",", " ").split() if t]
            ns_list = [int(t) for t in ns_tokens]
        except Exception:
            messagebox.showerror("Load", "Invalid Ns list.")
            return
        try:
            eta_vals = frange(float(self.eta_start_var.get()), float(self.eta_end_var.get()), float(self.eta_step_var.get()))
        except Exception:
            messagebox.showerror("Load", "Invalid eta range.")
            return
        repeats = max(1, int(self.repeats_var.get()))

        # Base parameter expectations
        base_checks = {
            "sd0": str(float(self.sd0_var.get())),
            "sd1": str(float(self.sd1_var.get())),
            "sc0": str(float(self.sc0_var.get())),
            "sc1": str(float(self.sc1_var.get())),
            "sample_delta": str(float(self.sample_delta_var.get())),
            "t_max": str(float(self.t_max_var.get())),
        }
        # Optional p-params
        opt_p = {}
        for key, var in [("p1", self.p1_var), ("p00", self.p00_var), ("p01", self.p01_var), ("p11", self.p11_var)]:
            val = var.get().strip()
            if val != "":
                opt_p[key] = str(float(val))
        # Rho check
        rho_equals_n = bool(self.rho_equals_n_var.get())
        rho_fixed = None if rho_equals_n else str(float(self.rho_value_var.get()))
        t_max = float(self.t_max_var.get())
        sweep_label = "beta" if rho_equals_n else "eta"
        # When rho_equals_n: expect beta in header; otherwise expect eta and possibly rho
        files = sorted(glob.glob(os.path.join(out_dir, "*.csv")))
        if not files:
            messagebox.showinfo("Load", f"No CSV files found in: {out_dir}")
            return
        ok = 0
        mismatched = 0
        incomplete = 0
        skipped_non_sim = 0
        counts = defaultdict(int)  # (n, eta) -> count
        mismatched_examples = []
        first_mismatch_reason: Optional[str] = None
        incomplete_examples = []
        for fp in files:
            # Skip non-simulation CSVs
            if not is_simulation_csv(fp):
                skipped_non_sim += 1
                continue
            params = parse_header_params_from_file(fp)
            # Normalize params
            def getp(k: str) -> Optional[str]:
                return params.get(k.lower())
            try:
                n_val = int(float(getp("n") or ""))
            except Exception:
                mismatched += 1; mismatched_examples.append(os.path.basename(fp));
                if first_mismatch_reason is None:
                    first_mismatch_reason = f"{os.path.basename(fp)}: missing or invalid n in header"
                continue
            # Sweep value: beta (when rho=n) or eta otherwise
            sweep_raw = getp(sweep_label) or ""
            try:
                sweep_val = float(sweep_raw)
            except Exception:
                mismatched += 1; mismatched_examples.append(os.path.basename(fp));
                if first_mismatch_reason is None:
                    first_mismatch_reason = f"{os.path.basename(fp)}: missing or invalid {sweep_label} in header"
                continue
            # Check sweep/n are expected
            if n_val not in ns_list or all(abs(sweep_val - e) > 1e-6 for e in eta_vals):
                mismatched += 1; mismatched_examples.append(os.path.basename(fp));
                if first_mismatch_reason is None:
                    first_mismatch_reason = f"{os.path.basename(fp)}: unexpected combo (n={n_val}, {sweep_label}={sweep_val}) not in GUI selection"
                continue
            # Base checks
            bad = False
            bad_reason = None
            for k, v in base_checks.items():
                pv = getp(k)
                if pv is None:
                    bad = True; bad_reason = f"missing parameter {k}"; break
                try:
                    if abs(float(pv) - float(v)) > 1e-9:
                        bad = True; bad_reason = f"mismatch {k}: file={pv}, expected={v}"; break
                except Exception:
                    bad = True; bad_reason = f"invalid numeric value for {k}: {pv}"; break
            if bad:
                mismatched += 1; mismatched_examples.append(os.path.basename(fp));
                if first_mismatch_reason is None and bad_reason:
                    first_mismatch_reason = f"{os.path.basename(fp)}: {bad_reason}"
                continue
            # p-params if provided
            bad = False
            bad_reason = None
            for k, v in opt_p.items():
                pv = getp(k)
                if pv is None:
                    bad = True; bad_reason = f"missing p-parameter {k}"; break
                try:
                    if abs(float(pv) - float(v)) > 1e-9:
                        bad = True; bad_reason = f"mismatch {k}: file={pv}, expected={v}"; break
                except Exception:
                    bad = True; bad_reason = f"invalid numeric value for {k}: {pv}"; break
            if bad:
                mismatched += 1; mismatched_examples.append(os.path.basename(fp));
                if first_mismatch_reason is None and bad_reason:
                    first_mismatch_reason = f"{os.path.basename(fp)}: {bad_reason}"
                continue
            # rho validation: only enforced when rho != n in GUI
            if not rho_equals_n:
                rho_p = getp("rho") or "nan"
                try:
                    if abs(float(rho_p) - float(rho_fixed or 0.0)) > 1e-9:
                        mismatched += 1; mismatched_examples.append(os.path.basename(fp));
                        if first_mismatch_reason is None:
                            first_mismatch_reason = f"{os.path.basename(fp)}: rho != fixed (rho={rho_p}, expected={rho_fixed})"
                        continue
                except Exception:
                    mismatched += 1; mismatched_examples.append(os.path.basename(fp));
                    if first_mismatch_reason is None:
                        first_mismatch_reason = f"{os.path.basename(fp)}: invalid rho value: {rho_p}"
                    continue
            # Completion: last time >= t_max - small eps
            last_t = read_last_data_time(fp)
            if last_t is None or last_t + 1e-9 < t_max:
                incomplete += 1
                incomplete_examples.append(os.path.basename(fp))
                continue
            ok += 1
            counts[(n_val, round(sweep_val, 10))] += 1

        # Compare against expected repeats
        missing = []
        for n in ns_list:
            for e in eta_vals:
                c = counts[(n, round(e, 10))]
                if c < repeats:
                    missing.append((n, e, repeats - c))

        summary = [
            f"Scanned files: {len(files)}",
            f"OK: {ok}",
            f"Mismatched params: {mismatched}",
            f"Incomplete (t < t_max): {incomplete}",
            f"Expected total (n x eta x repeats): {len(ns_list) * len(eta_vals) * repeats}",
            f"Missing combos: {len(missing)}",
        ]
        if skipped_non_sim:
            summary.insert(1, f"Skipped non-simulation files: {skipped_non_sim}")
        if missing:
            preview = ", ".join([f"(n={n}, eta={e:.3f}, missing={m})" for n, e, m in missing[:10]])
            summary.append(f"First missing: {preview}{' ...' if len(missing) > 10 else ''}")
        if mismatched_examples:
            summary.append(f"Examples mismatched: {', '.join(mismatched_examples[:5])}{' ...' if len(mismatched_examples)>5 else ''}")
        if first_mismatch_reason:
            summary.append(f"First mismatch reason: {first_mismatch_reason}")
        if incomplete_examples:
            summary.append(f"Examples incomplete: {', '.join(incomplete_examples[:5])}{' ...' if len(incomplete_examples)>5 else ''}")
        # Update progress bar to reflect done/total
        expected_total = len(ns_list) * len(eta_vals) * repeats
        self.set_meta(done=ok, total=expected_total, active=0)
        # Save counts for scheduling remaining tasks
        self.latest_counts = dict(counts)
        self.latest_expected_total = expected_total
        self.latest_done_ok = ok
        messagebox.showinfo("Load Summary", "\n".join(summary))
        # Refresh schedule window if open (update done_before and recompute rows)
        try:
            if self.schedule_win and self.schedule_model and self.schedule_win.winfo_exists():
                # Replace counts_before with latest_counts (capped by repeats per row when rendering)
                self.schedule_model["counts_before"] = dict(self.latest_counts)
                self._schedule_refresh_all_rows()
        except Exception:
            pass

    def on_schedule(self) -> None:
        # Show a table of all (n, eta) combos with done and remaining
        try:
            ns_tokens = [t for t in self.ns_var.get().replace(",", " ").split() if t]
            ns_list = [int(t) for t in ns_tokens]
        except Exception:
            messagebox.showerror("Schedule", "Invalid Ns list.")
            return
        try:
            eta_vals = frange(float(self.eta_start_var.get()), float(self.eta_end_var.get()), float(self.eta_step_var.get()))
        except Exception:
            messagebox.showerror("Schedule", "Invalid eta range.")
            return
        repeats = max(1, int(self.repeats_var.get()))
        counts_before = dict(self.latest_counts or {})
        # Recreate schedule window if already open
        if self.schedule_win and self.schedule_win.winfo_exists():
            try:
                self.schedule_win.destroy()
            except Exception:
                pass
        win = tk.Toplevel(self)
        win.title("Schedule")
        win.geometry("760x520")
        cols = ("n", "eta", "planned", "done_before", "running", "done_now", "to_dispatch")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c)
        # Set narrower column widths and disable stretch
        tree.column("n", width=70, stretch=False, anchor=tk.E)
        tree.column("eta", width=80, stretch=False, anchor=tk.E)
        tree.column("planned", width=90, stretch=False, anchor=tk.E)
        tree.column("done_before", width=110, stretch=False, anchor=tk.E)
        tree.column("running", width=90, stretch=False, anchor=tk.E)
        tree.column("done_now", width=90, stretch=False, anchor=tk.E)
        tree.column("to_dispatch", width=110, stretch=False, anchor=tk.E)
        tree.pack(fill=tk.BOTH, expand=True)
        # Prepare model
        model = {
            "ns": ns_list,
            "etas": eta_vals,
            "repeats": repeats,
            "counts_before": counts_before,
            "active_counts": defaultdict(int),
            "done_session": defaultdict(int),
            "rows": {},
        }
        total_planned = 0
        total_done_before = 0
        total_running = 0
        total_done_now = 0
        total_to_dispatch = 0
        for n in ns_list:
            for e in eta_vals:
                key = (n, round(e, 10))
                planned = repeats
                done_before = min(planned, int(counts_before.get(key, 0)))
                running = 0
                done_now = 0
                to_dispatch = max(0, planned - done_before - running - done_now)
                item_id = tree.insert("", tk.END, values=(n, f"{e:.3f}", planned, done_before, running, done_now, to_dispatch))
                model["rows"][key] = item_id
                total_planned += planned
                total_done_before += done_before
                total_running += running
                total_done_now += done_now
                total_to_dispatch += to_dispatch
        summ = ttk.Label(win, text=f"Planned: {total_planned}    Done before: {total_done_before}    Running: {total_running}    Done now: {total_done_now}    To dispatch: {total_to_dispatch}")
        summ.pack(anchor=tk.W, padx=8, pady=8)
        # Save handles
        self.schedule_win = win
        self.schedule_tree = tree
        self.schedule_summary = summ
        self.schedule_model = model
        # Also reflect schedule summary in main window
        try:
            self.schedule_main_label.configure(text=f"Schedule — Planned: {total_planned} | Done before: {total_done_before} | Running: {total_running} | Done now: {total_done_now} | To dispatch: {total_to_dispatch}")
        except Exception:
            pass
        # Update top progress to reflect this schedule (pre-done only)
        self.latest_expected_total = total_planned
        self.latest_done_ok = total_done_before
        self.set_meta(done=total_done_before, total=total_planned, active=self.meta_active)
        # Mark schedule as viewed and enable Start
        self.schedule_viewed = True
        self.start_btn.configure(state=tk.NORMAL if not (self.worker and self.worker.is_alive()) else tk.DISABLED)

    def invalidate_schedule(self) -> None:
        self.schedule_viewed = False
        self.start_btn.configure(state=tk.DISABLED)

    def _schedule_refresh_all_rows(self) -> None:
        if not (self.schedule_model and self.schedule_tree and self.schedule_win and self.schedule_win.winfo_exists()):
            return
        ns_list = self.schedule_model["ns"]
        eta_vals = self.schedule_model["etas"]
        repeats = self.schedule_model["repeats"]
        counts_before = self.schedule_model["counts_before"]
        active_counts = self.schedule_model["active_counts"]
        done_session = self.schedule_model["done_session"]
        rows = self.schedule_model["rows"]
        total_planned = total_done_before = total_running = total_done_now = total_to_dispatch = 0
        for n in ns_list:
            for e in eta_vals:
                key = (n, round(e, 10))
                planned = repeats
                done_before = min(planned, int(counts_before.get(key, 0)))
                running = int(active_counts.get(key, 0))
                done_now = int(done_session.get(key, 0))
                to_dispatch = max(0, planned - done_before - running - done_now)
                item_id = rows.get(key)
                if item_id:
                    try:
                        self.schedule_tree.item(item_id, values=(n, f"{e:.3f}", planned, done_before, running, done_now, to_dispatch))
                    except Exception:
                        pass
                total_planned += planned
                total_done_before += done_before
                total_running += running
                total_done_now += done_now
                total_to_dispatch += to_dispatch
        try:
            if self.schedule_summary:
                self.schedule_summary.configure(text=f"Planned: {total_planned}    Done before: {total_done_before}    Running: {total_running}    Done now: {total_done_now}    To dispatch: {total_to_dispatch}")
            # Mirror into main window label as well
            self.schedule_main_label.configure(text=f"Schedule — Planned: {total_planned} | Done before: {total_done_before} | Running: {total_running} | Done now: {total_done_now} | To dispatch: {total_to_dispatch}")
        except Exception:
            pass

    @staticmethod
    def _parse_cmd_combo(cmd: List[str]) -> Optional[tuple]:
        # Extract (n, sweep_value) where sweep_value is eta or beta depending on usage
        try:
            n_val = None
            sweep_val = None
            i = 0
            while i < len(cmd):
                if cmd[i] == "--n" and i + 1 < len(cmd):
                    n_val = int(cmd[i+1]); i += 2; continue
                if cmd[i] in ("--eta", "--beta") and i + 1 < len(cmd):
                    sweep_val = float(cmd[i+1]); i += 2; continue
                i += 1
            if n_val is None or sweep_val is None:
                return None
            return (n_val, round(sweep_val, 10))
        except Exception:
            return None

    def on_pause(self) -> None:
        if self.worker and self.worker.is_alive():
            try:
                self.worker.pause()
                self.pause_btn.configure(state=tk.DISABLED)
                self.resume_btn.configure(state=tk.NORMAL)
            except Exception:
                pass

    def on_resume(self) -> None:
        if self.worker and self.worker.is_alive():
            try:
                self.worker.resume()
                self.pause_btn.configure(state=tk.NORMAL)
                self.resume_btn.configure(state=tk.DISABLED)
            except Exception:
                pass

    def add_row(self, jid: int, cmd: List[str]) -> None:
        f = ttk.Frame(self.scrollframe, padding=(2, 2))
        f.pack(fill=tk.X)
        title = ttk.Label(f, text=join_cmd(cmd), font=("TkDefaultFont", 9, "bold"), wraplength=1000, justify=tk.LEFT)
        title.pack(anchor=tk.W)
        line = ttk.Label(f, text="", font=("TkFixedFont", 9))
        line.pack(anchor=tk.W)
        self.rows[jid] = (f, title, line)
        # Track job combo for schedule live status
        combo = self._parse_cmd_combo(cmd)
        if combo is not None:
            self.job_combo[jid] = combo
            if self.schedule_model:
                try:
                    self.schedule_model["active_counts"][combo] += 1
                    self._schedule_refresh_all_rows()
                except Exception:
                    pass

    def update_row(self, jid: int, text: str) -> None:
        row = self.rows.get(jid)
        if not row:
            return
        _, _, line = row
        # Replace inner newlines/carriage returns and strip ANSI in case
        clean = strip_ansi(text).replace("\r", " ").replace("\n", " ")
        line.configure(text=clean)

    def remove_row(self, jid: int) -> None:
        row = self.rows.pop(jid, None)
        if not row:
            return
        f, _, _ = row
        f.destroy()
        # Update schedule running/done_now when a job finishes
        combo = self.job_combo.pop(jid, None)
        if combo is not None and self.schedule_model:
            try:
                if self.schedule_model["active_counts"].get(combo, 0) > 0:
                    self.schedule_model["active_counts"][combo] -= 1
                self.schedule_model["done_session"][combo] += 1
                self._schedule_refresh_all_rows()
            except Exception:
                pass

    def set_meta(self, done: Optional[int] = None, total: Optional[int] = None, active: Optional[int] = None) -> None:
        if done is not None:
            self.meta_done = done
        if total is not None:
            self.meta_total = total
        if active is not None:
            self.meta_active = active
        total = max(1, self.meta_total)
        self.progress.configure(maximum=total, value=self.meta_done)
        to_dispatch = max(0, self.meta_total - self.meta_done - self.meta_active)
        self.status_label.configure(text=f"Completed: {self.meta_done} / {self.meta_total}    To dispatch: {to_dispatch}    Active: {self.meta_active}")

    def poll_queue(self) -> None:
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "meta":
                    data = msg[1]
                    if self.done_offset > 0 and self.latest_expected_total is not None:
                        # Keep loaded totals; just reset active
                        self.set_meta(done=self.done_offset, total=self.latest_expected_total, active=0)
                    else:
                        self.set_meta(done=0, total=data.get("total", 0), active=0)
                elif kind == "start":
                    jid, cmd = msg[1], msg[2]
                    self.add_row(jid, cmd)
                    self.set_meta(active=self.meta_active + 1)
                    # schedule refresh handled in add_row
                elif kind == "update":
                    jid, text = msg[1], msg[2]
                    self.update_row(jid, text)
                elif kind == "finish":
                    jid, ret = msg[1], msg[2]
                    self.remove_row(jid)
                    self.set_meta(done=self.meta_done + 1, active=max(0, self.meta_active - 1))
                    # schedule refresh handled in remove_row
                elif kind == "progress":
                    data = msg[1]
                    done_now = data.get("done", 0)
                    if self.done_offset > 0:
                        done_now += self.done_offset
                    self.set_meta(done=done_now, active=data.get("active", self.meta_active))
                elif kind == "error":
                    err = msg[1]
                    messagebox.showerror("Error", err)
                    self.start_btn.configure(state=tk.NORMAL)
                    self.stop_btn.configure(state=tk.DISABLED)
                    self.pause_btn.configure(state=tk.DISABLED)
                    self.resume_btn.configure(state=tk.DISABLED)
                elif kind == "done":
                    # Worker finished queuing
                    self.start_btn.configure(state=tk.NORMAL)
                    self.stop_btn.configure(state=tk.DISABLED)
                    self.pause_btn.configure(state=tk.DISABLED)
                    self.resume_btn.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        # schedule next poll
        self.after(50, self.poll_queue)


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
