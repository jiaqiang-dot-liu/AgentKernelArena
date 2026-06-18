# Visualize and compare runs

AgentKernelArena ships a static dashboard under `visualization/` for comparing
run reports across agents and models. This page covers building the dashboard
data and serving it.

## What the dashboard reads

The dashboard scans for report directories that contain:

- `overall_summary.csv`
- `task_type_breakdown.json`
- `overall_report.txt`

By default it scans only visualization-specific report bundles:

```text
visualization/reports/<report_name>/
```

Workspace-run reports, which are usually located at
`workspace_<gpu>_<agent>/run_<timestamp>/reports/`, can also be scanned, but this
is opt-in.

## Build the dashboard data and serve it

From the `visualization/` directory:

```bash
python backend/scripts/build_dashboard_data.py
python backend/server.py --host 127.0.0.1 --port 8080
```

Then open:

```text
http://127.0.0.1:8080
```

### Serve on port 80

Port `80` usually requires elevated privileges:

```bash
sudo python backend/server.py --host 0.0.0.0 --port 80
```

Or use the helper script, which rebuilds the data first and then serves on port
`80`:

```bash
bash setup.sh
```

## Rebuild after new runs

Whenever a run produces new reports, rebuild the dashboard payload:

```bash
python backend/scripts/build_dashboard_data.py
```

To also include workspace runs (not just `visualization/reports/`):

```bash
python backend/scripts/build_dashboard_data.py --include-workspace-runs
# or
bash setup.sh --include-workspace-runs
```

The dashboard picks up newly discovered report directories on the next refresh.

## Notes

- The HTTP service serves the UI from `visualization/frontend/`.
- Source-file links are exposed through an `/artifacts/...` route that only
  allows `.csv`, `.json`, and `.txt` files.
- If no reports are found yet, the dashboard still builds and shows an empty
  state.
- `frontend/dashboard/data.js` and `frontend/dashboard/data.json` are generated
  files; do not edit them by hand.
