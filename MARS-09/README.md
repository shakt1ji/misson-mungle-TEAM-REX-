# MARS-09 — Terrain Intelligence & Autonomous Navigation (Member 1)

Modular, open-source Mars rover terrain analysis and uncertainty-aware navigation modules built for the MARS-09 hackathon project.

## Ownership Scope (Member 1)
- `terrain/` — DEM loading & CRS/unit validation, numerical slope computation, traversability masking, and weighted terrain cost maps.
- `navigation/` — Distance heuristics, **UARA\*** (Uncertainty-Aware Risk-Adaptive A\*), supercover line-of-sight path smoothing, and discrete rover simulation.
- `tests/` — Deterministic Pytest suite.
- `docs/INTERFACES.md` — Formal array, coordinate, and function signature contract for Member 2 (`ml/`, `perception/`) and Member 3 (`dashboard/`, `science/`).
- `main.py` — Minimal end-to-end integration demo.

## Installation (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running Tests

```powershell
pytest -v
```

## Running the Demo

Run with the built-in synthetic Mars terrain fixture:

```powershell
python main.py
```

Run headlessly and save the diagnostic plot:

```powershell
python main.py --no-show --save-plot outputs/demo_mission.png
```

Run with a real projected GeoTIFF DEM (horizontal CRS in meters):

```powershell
python main.py --dem data/raw/mars_site_projected.tif
```

## Integration Notes
See `docs/INTERFACES.md` for all shared coordinate conventions `(row, column)`, array shapes, masks, and function signatures.

---

## Interactive Rover Simulation & Mission Dashboard (Member 3)

### Added Modules
- `simulation/` — Real-time rover physics (`physics.py`, `g = 3.721 m/s²`, Bekker soil slip/sinkage/rolling resistance), interactive rock obstacle manager (`dynamic_obstacles.py`), DEM + AI4MARS scenario builder & terrain editor (`environment_editor.py`), mission state (`simulation_state.py`), live UARA* replanning controller (`simulation_controller.py`), and JSON-lines RPC bridge (`bridge_server.py`).
- `dashboard/` — Streamlit multi-view mission operations center (`app.py`), interactive drag-and-drop canvas component (`components/interactive_rover_canvas.py`), simulation transport controls (`components/simulation_controls.py`), live physics & replan monitor (`components/mission_monitor.py`), Plotly route/3D DEM visualizations (`visualizations/route_map.py`), and analysis pages (`pages/04_route_planning.py`, `pages/08_interactive_simulator.py`).

### Running the Streamlit Dashboard (from `MARS-09/`)
```powershell
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

### Running All Simulation & Core Tests (from `MARS-09/`)
```powershell
pytest -v tests simulation/tests
```

