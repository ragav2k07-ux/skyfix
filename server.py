#!/usr/bin/env python3
"""SkyFix local server.

Serves the website (public/index.html) and runs train_rul.py when you choose
NASA C-MAPSS files on the "Give input" page.  Everything stays on your computer.

    python server.py            # then open http://localhost:5000
    python server.py --port 8000
"""
import argparse, os, re, shutil, subprocess, sys, tempfile, threading, time, uuid
from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(ROOT, "public")
SAMPLE = os.path.join(ROOT, "sample_data")
TRAIN_SCRIPT = os.path.join(ROOT, "train_rul.py")
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
TIMEOUT_S = 30 * 60

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
JOBS = {}            # job id -> dict(status, log, csv_path, error, started)
LOCK = threading.Lock()


@app.after_request
def cors(resp):
    # lets the page work if you open it from VS Code Live Server (another port)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------- validation
def check_nasa_file(path, kind):
    """Cheap sanity checks so mistakes are reported in plain words, before training."""
    try:
        with open(path, "r", errors="replace") as f:
            lines = [ln for ln in f if ln.strip()]
    except OSError as e:
        raise ValueError(f"Could not read the {kind} file: {e}")
    if not lines:
        raise ValueError(f"The {kind} file is empty.")
    if kind == "RUL":
        try:
            vals = [float(ln.split()[0]) for ln in lines]
        except ValueError:
            raise ValueError("The RUL file should hold one number per line (cycles left for each test engine).")
        return len(vals)
    cols = len(lines[0].split())
    if cols < 26:
        raise ValueError(f"The {kind} file has {cols} columns but NASA C-MAPSS files have 26 "
                         "(unit, cycle, 3 settings, 21 sensors). Did you choose the right file?")
    try:
        units = {int(float(ln.split()[0])) for ln in lines}
    except ValueError:
        raise ValueError(f"The first column of the {kind} file should be the engine number.")
    return len(units)


# ------------------------------------------------------------------- the job
def run_job(job_id, data_dir, subset, cleanup):
    job = JOBS[job_id]
    out_csv = os.path.join(data_dir, "predictions.csv")
    cmd = [sys.executable, "-u", TRAIN_SCRIPT, "--data", data_dir, "--subset", subset, "--out", out_csv]
    job["log"].append(f"$ python train_rul.py --subset {subset}")
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        timer = threading.Timer(TIMEOUT_S, p.kill)
        timer.start()
        for line in p.stdout:
            job["log"].append(line.rstrip())
        p.wait()
        timer.cancel()
        if p.returncode == 0 and os.path.exists(out_csv):
            job["csv_path"] = out_csv
            job["status"] = "done"
        else:
            tail = [l for l in job["log"] if l.strip()][-1:] or ["unknown error"]
            job["error"] = f"train_rul.py stopped with an error: {tail[0]}"
            job["status"] = "error"
    except Exception as e:                       # noqa: BLE001
        job["error"] = f"Could not run train_rul.py: {e}"
        job["status"] = "error"
    finally:
        job["finished"] = time.time()
        if cleanup and job["status"] != "done":
            shutil.rmtree(data_dir, ignore_errors=True)


def start_job(data_dir, subset, cleanup):
    job_id = uuid.uuid4().hex[:10]
    with LOCK:
        JOBS[job_id] = {"status": "running", "log": [], "csv_path": None, "error": None,
                        "started": time.time(), "subset": subset}
    threading.Thread(target=run_job, args=(job_id, data_dir, subset, cleanup), daemon=True).start()
    return job_id


# -------------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(PUBLIC, "index.html")


@app.get("/api/health")
def health():
    have_sample = all(os.path.exists(os.path.join(SAMPLE, f"{n}_FD001.txt")) for n in ("train", "test", "RUL"))
    return jsonify(ok=True, script=os.path.exists(TRAIN_SCRIPT), sample=have_sample)


@app.post("/api/train")
def train():
    subset = request.form.get("subset", "FD001")
    if subset not in SUBSETS:
        return jsonify(error="Subset must be FD001, FD002, FD003 or FD004."), 400
    files = {k: request.files.get(k) for k in ("train", "test", "rul")}
    missing = [k for k, v in files.items() if v is None or not v.filename]
    if missing:
        return jsonify(error="Please choose all three files. Missing: " + ", ".join(missing) + "."), 400

    data_dir = tempfile.mkdtemp(prefix="skyfix_")
    names = {"train": f"train_{subset}.txt", "test": f"test_{subset}.txt", "rul": f"RUL_{subset}.txt"}
    for k, f in files.items():
        f.save(os.path.join(data_dir, names[k]))
    try:
        n_train = check_nasa_file(os.path.join(data_dir, names["train"]), "train")
        n_test = check_nasa_file(os.path.join(data_dir, names["test"]), "test")
        n_rul = check_nasa_file(os.path.join(data_dir, names["rul"]), "RUL")
        if n_rul != n_test:
            raise ValueError(f"The test file has {n_test} engines but the RUL file has {n_rul} lines. "
                             "They must match. Are they from the same subset (FD001-FD004)?")
        if n_train < 5:
            raise ValueError(f"The train file has only {n_train} engines. Need at least 5 to train.")
    except ValueError as e:
        shutil.rmtree(data_dir, ignore_errors=True)
        return jsonify(error=str(e)), 400
    return jsonify(job=start_job(data_dir, subset, cleanup=True), train_engines=n_train, test_engines=n_test)


@app.post("/api/train-sample")
def train_sample():
    """Train on the small simulated files in sample_data/ (for testing the whole chain)."""
    if not os.path.exists(os.path.join(SAMPLE, "train_FD001.txt")):
        return jsonify(error="sample_data/ is empty. Run: python make_sample_data.py"), 400
    data_dir = tempfile.mkdtemp(prefix="skyfix_")
    for n in ("train", "test", "RUL"):
        shutil.copy(os.path.join(SAMPLE, f"{n}_FD001.txt"), data_dir)
    return jsonify(job=start_job(data_dir, "FD001", cleanup=True))


@app.get("/api/job/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Unknown job. The server may have been restarted."), 404
    return jsonify(status=job["status"], log=job["log"][-200:], error=job["error"],
                   seconds=round((job.get("finished") or time.time()) - job["started"], 1))


@app.get("/api/job/<job_id>/predictions.csv")
def job_csv(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return jsonify(error="Predictions are not ready."), 404
    resp = send_from_directory(os.path.dirname(job["csv_path"]), "predictions.csv", mimetype="text/csv")
    resp.headers["Content-Disposition"] = f'attachment; filename="predictions_{job["subset"]}.csv"'
    return resp


@app.get("/<path:name>")
def static_files(name):
    if re.search(r"\.\.|^/", name):
        return jsonify(error="Not found"), 404
    return send_from_directory(PUBLIC, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"\n  SkyFix is running.  Open  http://localhost:{a.port}  in your browser.  (Ctrl+C to stop)\n")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)
