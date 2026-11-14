import { useCallback, useEffect, useRef, useState } from "react";
import { simClient } from "./api/simClient";
import { ControlHeader } from "./components/ControlHeader";
import { MarsRover3DView } from "./components/MarsRover3DView";
import { Tactical2DCanvas } from "./components/Tactical2DCanvas";
import { TelemetrySidebar } from "./components/TelemetrySidebar";
import type {
  CameraPreset,
  InteractionMode,
  ObstacleItem,
  SimulationGrids,
  SimulationStatePayload,
  TerrainBrushType,
  ViewLayout,
} from "./types/simulation";

export default function App() {
  const [simState, setSimState] = useState<SimulationStatePayload | null>(null);
  const [scenarios, setScenarios] = useState<Record<string, string>>({});
  const [viewLayout, setViewLayout] = useState<ViewLayout>("split");
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("chase");
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("drag");
  const [terrainBrush, setTerrainBrush] = useState<TerrainBrushType>("sand");
  const [rockRadiusCells, setRockRadiusCells] = useState<number>(1);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const cachedGridsRef = useRef<SimulationGrids | undefined>(undefined);
  const stepInFlightRef = useRef<boolean>(false);

  const applyServerState = useCallback((next: SimulationStatePayload) => {
    if (next.grids) {
      cachedGridsRef.current = next.grids;
      setSimState(next);
    } else {
      setSimState({ ...next, grids: cachedGridsRef.current });
    }
    setErrorMsg(null);
  }, []);

  const runAction = useCallback(
    async (fn: () => Promise<{ state: SimulationStatePayload }>) => {
      try {
        const res = await fn();
        applyServerState(res.state);
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : String(err));
      }
    },
    [applyServerState]
  );

  // Initial load from live Python MARS-09 bridge
  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const [stRes, scRes] = await Promise.all([
          simClient.getState(true),
          simClient.getScenarios(),
        ]);
        if (!active) return;
        setScenarios(scRes.scenarios);
        applyServerState(stRes.state);
      } catch (err) {
        if (active) setErrorMsg(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      active = false;
    };
  }, [applyServerState]);

  // Continuous physics step loop while rover is running
  useEffect(() => {
    if (!simState?.is_running) return;
    const intervalId = window.setInterval(() => {
      if (stepInFlightRef.current) return;
      stepInFlightRef.current = true;
      void simClient
        .step(0.18)
        .then((res) => {
          applyServerState(res.state);
        })
        .catch((err) => {
          setErrorMsg(err instanceof Error ? err.message : String(err));
        })
        .finally(() => {
          stepInFlightRef.current = false;
        });
    }, 150);

    return () => window.clearInterval(intervalId);
  }, [simState?.is_running, applyServerState]);

  if (!simState) {
    return (
      <div className="h-screen w-screen flex flex-col items-center justify-center bg-[#0c0a09] text-[#e7d8c9] gap-2">
        <div className="text-sm font-semibold tracking-wider">
          INITIALIZING MARS-09 PYTHON TERRAIN &amp; NAVIGATION ENGINE...
        </div>
        {errorMsg && <div className="text-xs text-[#fca5a5]">{errorMsg}</div>}
      </div>
    );
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0c0a09] text-[#f3ece4] overflow-hidden">
      <ControlHeader
        simState={simState}
        scenarios={scenarios}
        viewLayout={viewLayout}
        cameraPreset={cameraPreset}
        errorMsg={errorMsg}
        onChangeScenario={(name) => void runAction(() => simClient.loadScenario(name))}
        onChangeLayout={setViewLayout}
        onChangeCamera={setCameraPreset}
        onStart={() => void runAction(() => simClient.start())}
        onPause={() => void runAction(() => simClient.pause())}
        onResume={() => void runAction(() => simClient.resume())}
        onStep={() => void runAction(() => simClient.step(0.5))}
        onReplan={() => void runAction(() => simClient.replan())}
        onReset={() => void runAction(() => simClient.reset())}
        onChangeSpeed={(speed) =>
          void runAction(() => simClient.setControls({ speed_multiplier: speed }))
        }
      />

      <main className="flex-1 flex min-h-0 overflow-hidden">
        <div className="flex-1 grid gap-2 p-2 min-w-0 min-h-0" style={{
          gridTemplateColumns:
            viewLayout === "split" ? "minmax(0, 1.25fr) minmax(0, 1fr)" : "minmax(0, 1fr)",
        }}>
          {(viewLayout === "split" || viewLayout === "3d") && (
            <MarsRover3DView
              simState={simState}
              cameraPreset={cameraPreset}
              interactionMode={interactionMode}
              terrainBrush={terrainBrush}
              rockRadiusCells={rockRadiusCells}
              onMoveObstacle={(id, r, c) =>
                void runAction(() => simClient.moveObstacle(id, r, c))
              }
              onAddObstacle={(r, c, rad) =>
                void runAction(() => simClient.addObstacle(r, c, rad))
              }
              onRemoveObstacle={(obs: ObstacleItem) =>
                void runAction(() =>
                  simClient.removeObstacle({ obstacle_id: obs.obstacle_id })
                )
              }
              onPaintTerrain={(cells, tType) =>
                void runAction(() => simClient.paintTerrain(cells, tType))
              }
              onSetStart={(r, c) =>
                void runAction(() => simClient.setEndpoints({ start_cell: [r, c] }))
              }
              onSetGoal={(r, c) =>
                void runAction(() => simClient.setEndpoints({ goal_cell: [r, c] }))
              }
            />
          )}

          {(viewLayout === "split" || viewLayout === "2d") && (
            <Tactical2DCanvas
              simState={simState}
              interactionMode={interactionMode}
              terrainBrush={terrainBrush}
              rockRadiusCells={rockRadiusCells}
              onMoveObstacle={(id, r, c) =>
                void runAction(() => simClient.moveObstacle(id, r, c))
              }
              onAddObstacle={(r, c, rad) =>
                void runAction(() => simClient.addObstacle(r, c, rad))
              }
              onRemoveObstacle={(obs: ObstacleItem) =>
                void runAction(() =>
                  simClient.removeObstacle({ obstacle_id: obs.obstacle_id })
                )
              }
              onPaintTerrain={(cells, tType) =>
                void runAction(() => simClient.paintTerrain(cells, tType))
              }
              onSetStart={(r, c) =>
                void runAction(() => simClient.setEndpoints({ start_cell: [r, c] }))
              }
              onSetGoal={(r, c) =>
                void runAction(() => simClient.setEndpoints({ goal_cell: [r, c] }))
              }
            />
          )}
        </div>

        <TelemetrySidebar
          simState={simState}
          interactionMode={interactionMode}
          terrainBrush={terrainBrush}
          rockRadiusCells={rockRadiusCells}
          onChangeMode={setInteractionMode}
          onChangeBrush={setTerrainBrush}
          onChangeRockRadius={setRockRadiusCells}
          onToggleControl={(patch) =>
            void runAction(() => simClient.setControls(patch))
          }
        />
      </main>
    </div>
  );
}
