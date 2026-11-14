"""MARS-09 Mission Operations Center — Streamlit Entrypoint (`streamlit run dashboard/app.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure MARS-09 project root is importable regardless of working directory
_MARS09_ROOT = Path(__file__).resolve().parents[1]
if str(_MARS09_ROOT) not in sys.path:
    sys.path.insert(0, str(_MARS09_ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="MARS-09 Autonomous Rover Simulator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.pages import __doc__ as _pages_doc  # noqa: E402, F401
import importlib  # noqa: E402


sim_page = importlib.import_module("dashboard.pages.08_interactive_simulator")
route_page = importlib.import_module("dashboard.pages.04_route_planning")


def main() -> None:
    with st.sidebar:
        st.markdown("## MARS-09 Operations")
        view = st.radio(
            "Mission View",
            options=["Interactive Rover Simulator", "Terrain & UARA* Route Analysis"],
            label_visibility="collapsed",
        )
        st.divider()

    if view == "Interactive Rover Simulator":
        sim_page.render_interactive_simulator_page()
    else:
        route_page.render_route_planning_page()


if __name__ == "__main__":
    main()
