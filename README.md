# SkyFix – predictive maintenance for aircraft engines

```
skyfix/
├── server.py            ← run this: serves the website and runs the training
├── train_rul.py         ← your training script (unchanged)
├── make_sample_data.py  ← makes small SIMULATED NASA-style files for testing
├── requirements.txt
├── public/index.html    ← the website
├── sample_data/         ← simulated train/test/RUL files (FD001 format)
└── .vscode/             ← VS Code run/debug/task settings
```

## Run it in VS Code (5 steps)

1. **Install** Python 3.10+ (https://www.python.org) and VS Code. In VS Code install the **Python** extension.
2. **File → Open Folder…** and choose the `skyfix` folder.
3. Open a terminal (**Terminal → New Terminal**) and run:
   ```
   python -m venv .venv
   ```
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`

   then:
   ```
   pip install -r requirements.txt
   ```
   (If VS Code asks "use this new environment?", click **Yes**.)
4. Run the app: `python server.py` (or press **F5** and pick *SkyFix: run website + training server*).
5. Open **http://localhost:5000** → **Give input** → **Option 3**.

## Test it

- **Quick test:** on Option 3 press **Train on the bundled sample files** (~30–60 s). The Fleet, Engine, Planner and Cost pages switch to the trained predictions.
- **Your own files:** choose `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` from the NASA C-MAPSS download, then **Train and show results**. The subset is picked from the file name. FD001/FD003 take about a minute, FD002/FD004 a few minutes.
- **Mistakes are caught:** wrong file in a slot, wrong column count, RUL file that doesn't match the test file.
- **Already have a predictions.csv?** Use **Option 2 → Choose File**. No server needed (just double-click `public/index.html`).

## Command line only (no website)

```
python make_sample_data.py
python train_rul.py --data sample_data --subset FD001 --out predictions.csv
python train_rul.py --data CMAPSSData  --subset FD004          # real NASA files
```

## Notes

- Option 3 needs `python server.py` running. Opening `index.html` by double-click works for everything else; the page will show "Training server not running" until you start it.
- If port 5000 is busy (common on Mac): `python server.py --port 8000`, then open http://localhost:8000.
- Sample files are simulated. Quote accuracy only from real NASA data.

## Deploy the website to Netlify

`netlify.toml` publishes the `public/` folder as a static site. There is no build step and no `npm` involved.

- **Works on Netlify:** every page, plus **Option 2** (load a `predictions.csv` you already have). This all runs in the browser.
- **Doesn't run on Netlify:** `server.py`. It's a long-running Flask server that runs scikit-learn training for minutes at a time. Netlify only hosts static files and short serverless functions, so it can't host this.
- **Option 3 on the deployed site:** start `python server.py` on your own computer. The deployed page then connects to `http://127.0.0.1:5000`. If your browser blocks that connection, open `http://localhost:5000` directly instead. To offer training to every visitor, host `server.py` on a Python host that allows long-running processes (Render, Railway, Fly.io, a VM) and point the `API` variable in `public/index.html` at that URL.
