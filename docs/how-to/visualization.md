---
myst:
    html_meta:
        "description": "Build and serve the AgentKernelArena dashboard to compare baseline and treatment experiment runs."
        "keywords": "AgentKernelArena, visualization, A/B testing, dashboard, compare runs, agents, ROCm, GPU kernel"
---

# Visualize and compare runs in AgentKernelArena

AgentKernelArena ships a static dashboard under `visualization/` for comparing
experiment run reports. Reports identify the agent, target GPU, and run
timestamp; model/provider provenance is not currently a dashboard field. This
topic covers how to build the dashboard data and serve it.

## What the dashboard reads

The dashboard scans for report directories that contain:

- `overall_summary.csv`
- `task_type_breakdown.json`
- `overall_report.txt`

By default, it scans only visualization-specific report bundles:

```text
visualization/reports/<report_name>/
```

Workspace-run reports, which are usually located at
`workspace_<gpu>_<agent>/run_<timestamp>/reports/`, can also be scanned, but this
is opt-in.

## Build the dashboard data and serve it

After a normal AgentKernelArena run, reports land in
`workspace_<gpu>_<agent>/run_<timestamp>/reports/`. Pass
`--include-workspace-runs` so the build script picks them up:

```bash
python3 visualization/backend/scripts/build_dashboard_data.py --include-workspace-runs
python3 visualization/backend/server.py --host 127.0.0.1 --port 8080
```

Then open:

```text
http://127.0.0.1:8080
```

Without this flag, the script only scans `visualization/reports/`, which is
empty by default, and the dashboard shows no data.

### Serve on port `80`

Port `80` usually requires elevated privileges:

```bash
sudo python3 visualization/backend/server.py --host 0.0.0.0 --port 80
```

Or use the helper script, which rebuilds the data first and then serves on port
`80`:

```bash
bash visualization/setup.sh
```

## Rebuild after new runs

Whenever a run produces new reports, rebuild the dashboard payload with the
same flag used at first build:

```bash
python3 visualization/backend/scripts/build_dashboard_data.py --include-workspace-runs
# or
bash visualization/setup.sh --include-workspace-runs
```

The dashboard picks up newly discovered report directories on the next refresh.

If you placed report bundles manually in `visualization/reports/<report_name>/`
instead, omit the flag:

```bash
python3 visualization/backend/scripts/build_dashboard_data.py
```

## Dashboard implementation notes

The following details apply to the dashboard implementation:

- The HTTP service serves the UI from `visualization/frontend/`.
- Source-file links are exposed through an `/artifacts/...` route that only
  allows `.csv`, `.json`, and `.txt` files.
- If no reports are found yet, the dashboard still builds and shows an empty
  state.
- `frontend/dashboard/data.js` and `frontend/dashboard/data.json` are generated
  files; don't edit them by hand.
