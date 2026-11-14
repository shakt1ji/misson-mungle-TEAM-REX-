"""Persistent stdin/stdout JSON-lines RPC bridge for the Node/Express MARS-09 web server."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure MARS-09 root is on sys.path regardless of invocation working directory
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.dynamic_obstacles import InvalidObstaclePlacementError  # noqa: E402
from simulation.environment_editor import SCENARIO_DESCRIPTIONS  # noqa: E402
from simulation.simulation_controller import RoverSimulationController  # noqa: E402


def handle_command(controller: RoverSimulationController, req: dict[str, Any]) -> dict[str, Any]:
    cmd = str(req.get("cmd", "")).strip()
    args = req.get("args") or {}
    include_grids = bool(req.get("include_grids", True))

    if cmd == "get_state":
        return {"ok": True, "state": controller.to_payload(include_grids=include_grids)}

    if cmd == "scenarios":
        return {"ok": True, "scenarios": SCENARIO_DESCRIPTIONS}

    if cmd == "load_scenario":
        scenario = str(args.get("scenario_name", "dynamic_obstacle"))
        controller.load_scenario(scenario)
        return {"ok": True, "state": controller.to_payload(include_grids=True)}

    if cmd == "start":
        controller.start()
        return {"ok": True, "state": controller.to_payload(include_grids=include_grids)}

    if cmd == "pause":
        controller.pause()
        return {"ok": True, "state": controller.to_payload(include_grids=include_grids)}

    if cmd == "resume":
        controller.resume()
        return {"ok": True, "state": controller.to_payload(include_grids=include_grids)}

    if cmd == "reset":
        controller.reset()
        return {"ok": True, "state": controller.to_payload(include_grids=True)}

    if cmd == "step":
        dt_sec = float(args.get("dt_sec", 0.25))
        controller.step(dt_sec=dt_sec)
        return {"ok": True, "state": controller.to_payload(include_grids=include_grids)}

    if cmd == "replan":
        controller.replan_from_current_position(trigger="manual_replan")
        return {"ok": True, "state": controller.to_payload(include_grids=True)}

    if cmd == "set_controls":
        controller.set_controls(
            speed_multiplier=args.get("speed_multiplier"),
            auto_replan=args.get("auto_replan"),
            show_cost_overlay=args.get("show_cost_overlay"),
            show_uncertainty_overlay=args.get("show_uncertainty_overlay"),
            show_previous_route=args.get("show_previous_route"),
        )
        return {"ok": True, "state": controller.to_payload(include_grids=include_grids)}

    if cmd == "add_obstacle":
        obs = controller.add_obstacle(
            row=int(args["row"]),
            col=int(args["col"]),
            radius_cells=int(args.get("radius_cells", 0)),
            height_m=float(args.get("height_m", 1.25)),
            label=str(args.get("label", "Boulder")),
        )
        return {
            "ok": True,
            "obstacle_id": obs.obstacle_id,
            "state": controller.to_payload(include_grids=True),
        }

    if cmd == "move_obstacle":
        obs = controller.move_obstacle(
            obstacle_id=str(args["obstacle_id"]),
            new_row=int(args["row"]),
            new_col=int(args["col"]),
        )
        return {
            "ok": True,
            "obstacle_id": obs.obstacle_id,
            "state": controller.to_payload(include_grids=True),
        }

    if cmd == "remove_obstacle":
        obs_id = args.get("obstacle_id")
        if obs_id:
            controller.remove_obstacle(str(obs_id))
        else:
            controller.remove_obstacle_at(int(args["row"]), int(args["col"]))
        return {"ok": True, "state": controller.to_payload(include_grids=True)}

    if cmd == "paint_terrain":
        raw_cells = args.get("cells") or []
        cells = [(int(c[0]), int(c[1])) for c in raw_cells]
        terrain_type = str(args["terrain_type"])
        updated = controller.paint_terrain(cells=cells, terrain_type=terrain_type)
        return {
            "ok": True,
            "updated_cells": updated,
            "state": controller.to_payload(include_grids=True),
        }

    if cmd == "set_start_goal":
        s = args.get("start_cell")
        g = args.get("goal_cell")
        controller.set_start_goal(
            start_cell=(int(s[0]), int(s[1])) if s is not None else None,
            goal_cell=(int(g[0]), int(g[1])) if g is not None else None,
        )
        return {"ok": True, "state": controller.to_payload(include_grids=True)}

    return {"ok": False, "error": f"Unknown command '{cmd}'."}


def main() -> None:
    controller = RoverSimulationController(scenario_name="dynamic_obstacle", rows=30, cols=30)
    # Signal readiness
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        req_id = None
        try:
            req = json.loads(text)
            req_id = req.get("id")
            resp = handle_command(controller, req)
        except (InvalidObstaclePlacementError, KeyError, ValueError) as exc:
            resp = {"ok": False, "error": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            resp = {"ok": False, "error": f"Simulation error: {exc}"}

        if req_id is not None:
            resp["id"] = req_id
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
