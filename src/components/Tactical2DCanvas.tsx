import React, { useCallback, useEffect, useRef, useState } from "react";
import type {
  InteractionMode,
  ObstacleItem,
  SimulationStatePayload,
  TerrainBrushType,
} from "../types/simulation";

interface Tactical2DCanvasProps {
  simState: SimulationStatePayload;
  interactionMode: InteractionMode;
  terrainBrush: TerrainBrushType;
  rockRadiusCells: number;
  onMoveObstacle: (obstacleId: string, row: number, col: number) => void;
  onAddObstacle: (row: number, col: number, radiusCells: number) => void;
  onRemoveObstacle: (obs: ObstacleItem) => void;
  onPaintTerrain: (cells: [number, number][], terrainType: TerrainBrushType) => void;
  onSetStart: (row: number, col: number) => void;
  onSetGoal: (row: number, col: number) => void;
}

const TERRAIN_COLORS: Record<number, [number, number, number]> = {
  0: [164, 114, 74],
  1: [110, 108, 112],
  2: [212, 166, 98],
  3: [152, 66, 48],
  255: [42, 42, 46],
};

const TERRAIN_NAMES: Record<number, string> = {
  0: "Soil",
  1: "Bedrock",
  2: "Loose Sand",
  3: "Big Rock",
  255: "Unknown",
};

export const Tactical2DCanvas: React.FC<Tactical2DCanvasProps> = ({
  simState,
  interactionMode,
  terrainBrush,
  rockRadiusCells,
  onMoveObstacle,
  onAddObstacle,
  onRemoveObstacle,
  onPaintTerrain,
  onSetStart,
  onSetGoal,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverCell, setHoverCell] = useState<[number, number] | null>(null);
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [dragCell, setDragCell] = useState<[number, number] | null>(null);
  const isPaintingRef = useRef(false);
  const lastPaintedRef = useRef<string>("");

  const findObstacleAt = useCallback(
    (row: number, col: number): ObstacleItem | null => {
      for (const obs of simState.obstacles) {
        if (obs.row === row && obs.col === col) return obs;
      }
      for (const obs of simState.obstacles) {
        for (const [fr, fc] of obs.footprint) {
          if (fr === row && fc === col) return obs;
        }
      }
      return null;
    },
    [simState.obstacles]
  );

  const clientToGrid = useCallback(
    (clientX: number, clientY: number): [number, number] | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const [rows, cols] = simState.grid_shape;
      const col = Math.max(0, Math.min(cols - 1, Math.floor(((clientX - rect.left) / rect.width) * cols)));
      const row = Math.max(0, Math.min(rows - 1, Math.floor(((clientY - rect.top) / rect.height) * rows)));
      return [row, col];
    },
    [simState.grid_shape]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !simState.grids) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const [rows, cols] = simState.grid_shape;
    const cw = canvas.width / cols;
    const ch = canvas.height / rows;
    const { elevation, terrain_class_ids, total_cost, uncertainty, blocked_mask } = simState.grids;

    let zMin = Infinity;
    let zMax = -Infinity;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const z = elevation[r][c];
        if (z < zMin) zMin = z;
        if (z > zMax) zMax = z;
      }
    }
    const zSpan = Math.max(zMax - zMin, 1e-3);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cid = terrain_class_ids[r][c];
        const base = TERRAIN_COLORS[cid] || TERRAIN_COLORS[255];
        const normZ = (elevation[r][c] - zMin) / zSpan;
        const shade = 0.72 + 0.42 * normZ;
        let rr = Math.min(255, Math.round(base[0] * shade));
        let gg = Math.min(255, Math.round(base[1] * shade));
        let bb = Math.min(255, Math.round(base[2] * shade));

        if (simState.show_cost_overlay) {
          const cv = total_cost[r][c];
          if (cv < 0) {
            rr = 148;
            gg = 32;
            bb = 32;
          } else {
            const t = Math.min(1, cv / 12.0);
            rr = Math.round(rr * 0.42 + 245 * t * 0.58);
            gg = Math.round(gg * 0.42 + 165 * (1 - t) * 0.58);
            bb = Math.round(bb * 0.42 + 35 * (1 - t) * 0.58);
          }
        }

        if (simState.show_uncertainty_overlay) {
          const uv = uncertainty[r][c];
          rr = Math.round(rr * (1 - 0.55 * uv) + 192 * 0.55 * uv);
          gg = Math.round(gg * (1 - 0.55 * uv) + 72 * 0.55 * uv);
          bb = Math.round(bb * (1 - 0.55 * uv) + 235 * 0.55 * uv);
        }

        ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
        ctx.fillRect(c * cw, r * ch, cw + 0.6, ch + 0.6);

        if (blocked_mask[r][c] === 1) {
          ctx.fillStyle = "rgba(180, 42, 32, 0.24)";
          ctx.fillRect(c * cw, r * ch, cw, ch);
        }

        ctx.strokeStyle = "rgba(255, 225, 195, 0.06)";
        ctx.lineWidth = 0.6;
        ctx.strokeRect(c * cw, r * ch, cw, ch);
      }
    }

    if (simState.show_previous_route && simState.routes.previous_route.length >= 2) {
      ctx.save();
      ctx.strokeStyle = "rgba(180, 196, 214, 0.72)";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      simState.routes.previous_route.forEach(([r, c], i) => {
        const x = (c + 0.5) * cw;
        const y = (r + 0.5) * ch;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }

    const pts = simState.routes.travelled_points;
    if (pts.length >= 2) {
      for (const offset of [-0.18, 0.18]) {
        ctx.save();
        ctx.strokeStyle = "rgba(37, 99, 235, 0.9)";
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        for (let i = 0; i < pts.length; i++) {
          const [r, c] = pts[i];
          const prev = pts[Math.max(0, i - 1)];
          const next = pts[Math.min(pts.length - 1, i + 1)];
          const ang = Math.atan2(next[0] - prev[0], next[1] - prev[1]);
          const rx = (c + 0.5 - Math.sin(ang) * offset) * cw;
          const ry = (r + 0.5 + Math.cos(ang) * offset) * ch;
          if (i === 0) ctx.moveTo(rx, ry);
          else ctx.lineTo(rx, ry);
        }
        ctx.stroke();
        ctx.restore();
      }
    }

    const active = simState.routes.active_route;
    if (active.length >= 2) {
      ctx.save();
      ctx.strokeStyle = "rgba(0, 229, 255, 0.34)";
      ctx.lineWidth = 6.5;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      active.forEach(([r, c], i) => {
        const x = (c + 0.5) * cw;
        const y = (r + 0.5) * ch;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.strokeStyle = "#00e5ff";
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      active.forEach(([r, c], i) => {
        const x = (c + 0.5) * cw;
        const y = (r + 0.5) * ch;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }

    // Start & Goal markers
    const [sr, sc] = simState.start_cell;
    ctx.fillStyle = "#22c55e";
    ctx.beginPath();
    ctx.arc((sc + 0.5) * cw, (sr + 0.5) * ch, cw * 0.32, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#08120c";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    const [gr, gc] = simState.goal_cell;
    const gx = (gc + 0.5) * cw;
    const gy = (gr + 0.5) * ch;
    ctx.fillStyle = "#f59e0b";
    ctx.beginPath();
    ctx.moveTo(gx, gy - ch * 0.46);
    ctx.lineTo(gx + cw * 0.46, gy);
    ctx.lineTo(gx, gy + ch * 0.46);
    ctx.lineTo(gx - cw * 0.46, gy);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = "#1c1204";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Rock obstacles
    for (const obs of simState.obstacles) {
      const isDragged = draggingId === obs.obstacle_id;
      const drawR = isDragged && dragCell ? dragCell[0] : obs.row;
      const drawC = isDragged && dragCell ? dragCell[1] : obs.col;

      ctx.fillStyle = isDragged ? "rgba(255, 196, 60, 0.32)" : "rgba(220, 68, 46, 0.22)";
      if (isDragged && dragCell) {
        const rad = obs.radius_cells || 0;
        for (let dr = -rad; dr <= rad; dr++) {
          for (let dc = -rad; dc <= rad; dc++) {
            if (Math.hypot(dr, dc) <= rad + 0.15) {
              ctx.fillRect((drawC + dc) * cw, (drawR + dr) * ch, cw, ch);
            }
          }
        }
      } else {
        for (const [fr, fc] of obs.footprint) {
          ctx.fillRect(fc * cw, fr * ch, cw, ch);
        }
      }

      const cx = (drawC + 0.5) * cw;
      const cy = (drawR + 0.5) * ch;
      const radPx = (0.44 + (obs.radius_cells || 0) * 0.62) * Math.min(cw, ch);

      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.beginPath();
      ctx.ellipse(cx + 2, cy + 3, radPx * 1.05, radPx * 0.82, 0.2, 0, Math.PI * 2);
      ctx.fill();

      const grad = ctx.createRadialGradient(cx - radPx * 0.3, cy - radPx * 0.3, 1, cx, cy, radPx);
      grad.addColorStop(0, isDragged ? "#f9a852" : "#b86344");
      grad.addColorStop(1, isDragged ? "#993d1c" : "#51261a");
      ctx.fillStyle = grad;
      ctx.beginPath();
      for (let k = 0; k < 7; k++) {
        const a = (k / 7) * Math.PI * 2 + 0.3;
        const jitter = 0.82 + (0.22 * ((k * 37) % 5)) / 4;
        const px = cx + Math.cos(a) * radPx * jitter;
        const py = cy + Math.sin(a) * radPx * jitter;
        if (k === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = isDragged ? "#fde047" : "#f87171";
      ctx.lineWidth = isDragged ? 2 : 1.1;
      ctx.stroke();
    }

    if (hoverCell) {
      const [hr, hc] = hoverCell;
      ctx.strokeStyle = "rgba(255,255,255,0.65)";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(hc * cw + 0.5, hr * ch + 0.5, cw - 1, ch - 1);
    }

    // 6-wheel rover glyph
    const rx = (simState.rover.col_f + 0.5) * cw;
    const ry = (simState.rover.row_f + 0.5) * ch;
    ctx.save();
    ctx.translate(rx, ry);
    ctx.rotate(simState.rover.heading_rad || 0);
    ctx.fillStyle = "#18181b";
    for (const wx of [-0.36 * cw, 0, 0.36 * cw]) {
      for (const wy of [-0.36 * ch, 0.36 * ch]) {
        ctx.fillRect(wx - 2.4, wy - 1.8, 4.8, 3.6);
      }
    }
    ctx.fillStyle = "#e5e7eb";
    ctx.strokeStyle = "#0ea5e9";
    ctx.lineWidth = 1.4;
    ctx.fillRect(-0.34 * cw, -0.24 * ch, 0.68 * cw, 0.48 * ch);
    ctx.strokeRect(-0.34 * cw, -0.24 * ch, 0.68 * cw, 0.48 * ch);
    ctx.fillStyle = "#38bdf8";
    ctx.fillRect(0.14 * cw, -0.1 * ch, 0.2 * cw, 0.2 * ch);
    ctx.restore();
  }, [simState, hoverCell, draggingId, dragCell]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const cell = clientToGrid(e.clientX, e.clientY);
    if (!cell) return;
    const [row, col] = cell;

    if (interactionMode === "drag") {
      const obs = findObstacleAt(row, col);
      if (obs && obs.is_draggable) {
        setDraggingId(obs.obstacle_id);
        setDragCell([row, col]);
      }
    } else if (interactionMode === "add") {
      onAddObstacle(row, col, rockRadiusCells);
    } else if (interactionMode === "remove") {
      const obs = findObstacleAt(row, col);
      if (obs) onRemoveObstacle(obs);
    } else if (interactionMode === "paint") {
      isPaintingRef.current = true;
      lastPaintedRef.current = `${row},${col}`;
      onPaintTerrain([[row, col]], terrainBrush);
    } else if (interactionMode === "start") {
      onSetStart(row, col);
    } else if (interactionMode === "goal") {
      onSetGoal(row, col);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const cell = clientToGrid(e.clientX, e.clientY);
    if (!cell) return;
    setHoverCell(cell);
    if (draggingId) {
      setDragCell(cell);
    } else if (isPaintingRef.current && interactionMode === "paint") {
      const key = `${cell[0]},${cell[1]}`;
      if (key !== lastPaintedRef.current) {
        lastPaintedRef.current = key;
        onPaintTerrain([cell], terrainBrush);
      }
    }
  };

  const finishDrag = () => {
    if (draggingId && dragCell) {
      onMoveObstacle(draggingId, dragCell[0], dragCell[1]);
    }
    setDraggingId(null);
    setDragCell(null);
    isPaintingRef.current = false;
  };

  const cursorClass =
    draggingId
      ? "cursor-grabbing"
      : interactionMode === "drag"
      ? "cursor-grab"
      : interactionMode === "remove"
      ? "cursor-not-allowed"
      : "cursor-crosshair";

  let hoverLabel = "Hover or drag rocks on 2D tactical grid";
  if (hoverCell && simState.grids) {
    const [hr, hc] = hoverCell;
    const elev = simState.grids.elevation[hr]?.[hc] ?? 0;
    const slope = simState.grids.slope_deg[hr]?.[hc] ?? 0;
    const cost = simState.grids.total_cost[hr]?.[hc] ?? 0;
    const tc = TERRAIN_NAMES[simState.grids.terrain_class_ids[hr]?.[hc] ?? 255] ?? "Unknown";
    hoverLabel = `Cell (${hr}, ${hc}) · ${tc} · Elev ${elev.toFixed(1)}m · Slope ${slope.toFixed(1)}° · Cost ${cost < 0 ? "INF" : cost.toFixed(2)}`;
  }

  return (
    <div className="flex flex-col h-full bg-[#120f0d] border border-[#2b221c] rounded-md overflow-hidden">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-[#2b221c] bg-[#171210] text-xs">
        <span className="font-semibold tracking-wide text-[#e7d8c9]">2D TACTICAL TOP-DOWN GRID</span>
        <span className="font-mono tabular-nums text-[11px] text-[#b4a494]">{hoverLabel}</span>
      </div>
      <div className="flex-1 flex items-center justify-center p-2 min-h-0">
        <canvas
          ref={canvasRef}
          width={600}
          height={600}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={finishDrag}
          onMouseLeave={() => {
            setHoverCell(null);
            finishDrag();
          }}
          className={`max-w-full max-h-full aspect-square rounded border border-[#322720] ${cursorClass}`}
        />
      </div>
    </div>
  );
};
