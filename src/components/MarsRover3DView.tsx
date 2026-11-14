import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import type {
  CameraPreset,
  InteractionMode,
  ObstacleItem,
  SimulationStatePayload,
  TerrainBrushType,
} from "../types/simulation";

interface MarsRover3DViewProps {
  simState: SimulationStatePayload;
  cameraPreset: CameraPreset;
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

function createRegolithTexture(): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#b87a52";
  ctx.fillRect(0, 0, size, size);

  let seed = 1337;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };

  for (let i = 0; i < 4500; i++) {
    const x = rand() * size;
    const y = rand() * size;
    const r = 0.6 + rand() * 2.2;
    const shade = rand();
    ctx.fillStyle =
      shade > 0.65
        ? "rgba(222, 172, 130, 0.22)"
        : shade > 0.3
        ? "rgba(116, 68, 42, 0.25)"
        : "rgba(62, 34, 22, 0.2)";
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(14, 14);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function createRoverGroup(): {
  root: THREE.Group;
  tiltGroup: THREE.Group;
  wheels: THREE.Mesh[];
  mastHead: THREE.Group;
} {
  const root = new THREE.Group();
  const tiltGroup = new THREE.Group();
  root.add(tiltGroup);

  const chassisMat = new THREE.MeshStandardMaterial({
    color: 0xe6dfd3,
    roughness: 0.42,
    metalness: 0.25,
  });
  const foilMat = new THREE.MeshStandardMaterial({
    color: 0xd99b26,
    roughness: 0.3,
    metalness: 0.85,
  });
  const darkMetalMat = new THREE.MeshStandardMaterial({
    color: 0x262930,
    roughness: 0.52,
    metalness: 0.72,
  });
  const wheelMat = new THREE.MeshStandardMaterial({
    color: 0x1c1e24,
    roughness: 0.72,
    metalness: 0.48,
  });
  const lensMat = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    emissive: 0x0ea5e9,
    emissiveIntensity: 0.85,
    roughness: 0.15,
    metalness: 0.9,
  });

  // 1. Warm electronics box (WEB) chassis body
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.56, 1.18), chassisMat);
  body.position.set(0, 0.72, 0);
  body.castShadow = true;
  tiltGroup.add(body);

  // 2. Gold Kapton thermal belly pan
  const belly = new THREE.Mesh(new THREE.BoxGeometry(1.52, 0.14, 1.05), foilMat);
  belly.position.set(0, 0.42, 0);
  tiltGroup.add(belly);

  // 3. Top deck equipment plate & high-gain antenna dish
  const deck = new THREE.Mesh(new THREE.BoxGeometry(1.62, 0.08, 1.12), darkMetalMat);
  deck.position.set(-0.04, 1.03, 0);
  tiltGroup.add(deck);

  const dish = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.08, 0.12, 16), chassisMat);
  dish.position.set(-0.35, 1.24, -0.25);
  dish.rotation.z = -0.35;
  tiltGroup.add(dish);

  // 4. Rear MMRTG nuclear power generator + cooling fins
  const rtg = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.22, 0.68, 12), darkMetalMat);
  rtg.rotation.z = Math.PI / 2 - 0.22;
  rtg.position.set(-1.02, 0.82, 0);
  rtg.castShadow = true;
  tiltGroup.add(rtg);

  // 5. Rocker-bogie suspension struts (left & right)
  for (const side of [-1, 1]) {
    const zOff = side * 0.72;
    const rockerBar = new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.06, 0.07), darkMetalMat);
    rockerBar.position.set(0, 0.46, zOff);
    tiltGroup.add(rockerBar);

    for (const wx of [-0.82, 0.0, 0.82]) {
      const strut = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 0.36, 8), darkMetalMat);
      strut.position.set(wx, 0.34, zOff);
      tiltGroup.add(strut);
    }
  }

  // 6. Six cleated aluminum-titanium wheels
  const wheels: THREE.Mesh[] = [];
  const wheelGeo = new THREE.CylinderGeometry(0.26, 0.26, 0.22, 16);
  wheelGeo.rotateX(Math.PI / 2);
  const rimGeo = new THREE.CylinderGeometry(0.14, 0.14, 0.24, 12);
  rimGeo.rotateX(Math.PI / 2);

  for (const side of [-1, 1]) {
    const zPos = side * 0.8;
    for (const wx of [-0.84, 0.0, 0.84]) {
      const wheel = new THREE.Mesh(wheelGeo, wheelMat);
      wheel.position.set(wx, 0.26, zPos);
      wheel.castShadow = true;
      const rim = new THREE.Mesh(rimGeo, foilMat);
      wheel.add(rim);
      tiltGroup.add(wheel);
      wheels.push(wheel);
    }
  }

  // 7. Remote Sensing Mast (RSM) + Mastcam-Z / SuperCam head (+X forward)
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.065, 0.82, 10), chassisMat);
  mast.position.set(0.54, 1.42, 0.26);
  mast.castShadow = true;
  tiltGroup.add(mast);

  const mastHead = new THREE.Group();
  mastHead.position.set(0.56, 1.84, 0.26);
  const headBox = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.2, 0.42), chassisMat);
  mastHead.add(headBox);
  for (const lz of [-0.1, 0.1]) {
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.08, 12), lensMat);
    lens.rotation.z = Math.PI / 2;
    lens.position.set(0.15, 0.02, lz);
    mastHead.add(lens);
  }
  tiltGroup.add(mastHead);

  // 8. Forward Robotic Arm + Turret
  const armUpper = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.07, 0.07), darkMetalMat);
  armUpper.position.set(1.02, 0.64, -0.18);
  armUpper.rotation.z = -0.35;
  tiltGroup.add(armUpper);

  const turret = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.14, 10), foilMat);
  turret.position.set(1.26, 0.52, -0.18);
  tiltGroup.add(turret);

  return { root, tiltGroup, wheels, mastHead };
}

function createBoulderMesh(radiusM: number, heightM: number, idHash: number): THREE.Mesh {
  const geo = new THREE.DodecahedronGeometry(radiusM, 1);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const vx = pos.getX(i);
    const vy = pos.getY(i);
    const vz = pos.getZ(i);
    const noise =
      0.82 +
      0.28 * Math.sin(vx * 3.7 + idHash) * Math.cos(vz * 4.1 - idHash * 0.5) +
      0.1 * Math.cos(vy * 5.0 + idHash);
    pos.setXYZ(i, vx * noise, vy * (heightM / Math.max(radiusM, 0.4)) * 0.72 * noise, vz * noise);
  }
  geo.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color: 0x7c4330,
    roughness: 0.92,
    metalness: 0.06,
    flatShading: true,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

export const MarsRover3DView: React.FC<MarsRover3DViewProps> = ({
  simState,
  cameraPreset,
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
  const containerRef = useRef<HTMLDivElement | null>(null);

  const latestPropsRef = useRef({
    simState,
    cameraPreset,
    interactionMode,
    terrainBrush,
    rockRadiusCells,
    onMoveObstacle,
    onAddObstacle,
    onRemoveObstacle,
    onPaintTerrain,
    onSetStart,
    onSetGoal,
  });
  latestPropsRef.current = {
    simState,
    cameraPreset,
    interactionMode,
    terrainBrush,
    rockRadiusCells,
    onMoveObstacle,
    onAddObstacle,
    onRemoveObstacle,
    onPaintTerrain,
    onSetStart,
    onSetGoal,
  };

  const sceneCtxRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    terrainMesh: THREE.Mesh | null;
    terrainGeo: THREE.PlaneGeometry | null;
    elevOrigin: number;
    rover: ReturnType<typeof createRoverGroup>;
    bouldersGroup: THREE.Group;
    boulderMeshes: Map<string, { mesh: THREE.Mesh; ring: THREE.Mesh; obs: ObstacleItem }>;
    routesGroup: THREE.Group;
    startBeacon: THREE.Mesh;
    goalBeacon: THREE.Group;
    scatterMesh: THREE.InstancedMesh | null;
    dragGhost: THREE.Mesh;
    applyTerrainGeometry: (st: SimulationStatePayload) => void;
    syncDynamicVisuals: (st: SimulationStatePayload) => void;
    sampleSurfaceY: (st: SimulationStatePayload, rowF: number, colF: number) => number;
  } | null>(null);

  // Initialize Three.js WebGL scene once
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    const gl = renderer.getContext();
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    const rawRenderer = String(
      (dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : null) ||
        gl.getParameter(gl.RENDERER) ||
        ""
    );
    const isSoftwareGl = /SwiftShader|llvmpipe|Softpipe|Software/i.test(rawRenderer);

    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const marsSkyColor = new THREE.Color(0xc98c66);
    scene.background = marsSkyColor;
    scene.fog = new THREE.FogExp2(0xc98c66, 0.0115);

    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = envTex;
    scene.environmentIntensity = 0.55;

    const camera = new THREE.PerspectiveCamera(
      48,
      Math.max(1, container.clientWidth) / Math.max(1, container.clientHeight),
      0.2,
      260
    );
    camera.position.set(0, 22, 34);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxPolarAngle = Math.PI * 0.47;
    controls.minDistance = 4;
    controls.maxDistance = 95;

    // Warm Martian Sunlight + Sky Hemisphere
    const hemi = new THREE.HemisphereLight(0xf3cbb0, 0x6b3a24, 0.9);
    scene.add(hemi);

    const sun = new THREE.DirectionalLight(0xffe7cf, 2.3);
    sun.position.set(28, 36, 18);
    if (!isSoftwareGl) {
      sun.castShadow = true;
      sun.shadow.mapSize.set(1024, 1024);
      sun.shadow.camera.near = 5;
      sun.shadow.camera.far = 110;
      const d = 38;
      sun.shadow.camera.left = -d;
      sun.shadow.camera.right = d;
      sun.shadow.camera.top = d;
      sun.shadow.camera.bottom = -d;
      sun.shadow.bias = -0.0005;
    }
    scene.add(sun);

    const regolithTex = createRegolithTexture();
    const terrainMat = new THREE.MeshStandardMaterial({
      map: regolithTex,
      vertexColors: true,
      roughness: 0.92,
      metalness: 0.04,
    });

    const rover = createRoverGroup();
    scene.add(rover.root);

    const bouldersGroup = new THREE.Group();
    scene.add(bouldersGroup);

    const routesGroup = new THREE.Group();
    scene.add(routesGroup);

    const startBeacon = new THREE.Mesh(
      new THREE.CylinderGeometry(0.45, 0.45, 0.2, 16),
      new THREE.MeshStandardMaterial({
        color: 0x22c55e,
        emissive: 0x16a34a,
        emissiveIntensity: 0.55,
      })
    );
    scene.add(startBeacon);

    const goalBeacon = new THREE.Group();
    const goalRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.7, 0.08, 10, 24),
      new THREE.MeshStandardMaterial({
        color: 0xf59e0b,
        emissive: 0xf59e0b,
        emissiveIntensity: 0.9,
      })
    );
    goalRing.rotation.x = Math.PI / 2;
    const goalPillar = new THREE.Mesh(
      new THREE.CylinderGeometry(0.08, 0.25, 2.2, 12),
      new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.55 })
    );
    goalPillar.position.y = 1.1;
    goalBeacon.add(goalRing, goalPillar);
    scene.add(goalBeacon);

    const dragGhost = new THREE.Mesh(
      new THREE.RingGeometry(0.8, 1.15, 24),
      new THREE.MeshBasicMaterial({ color: 0xfde047, side: THREE.DoubleSide })
    );
    dragGhost.rotation.x = -Math.PI / 2;
    dragGhost.visible = false;
    scene.add(dragGhost);

    const gridToWorldXZ = (st: SimulationStatePayload, rowF: number, colF: number) => {
      const [rows, cols] = st.grid_shape;
      const [dyM, dxM] = st.resolution_m;
      return {
        x: (colF - (cols - 1) / 2) * dxM,
        z: (rowF - (rows - 1) / 2) * dyM,
      };
    };

    const sampleSurfaceY = (st: SimulationStatePayload, rowF: number, colF: number): number => {
      const grids = st.grids;
      if (!grids || !sceneCtxRef.current) return 0;
      const [rows, cols] = st.grid_shape;
      const r = Math.max(0, Math.min(rows - 1, rowF));
      const c = Math.max(0, Math.min(cols - 1, colF));
      const r0 = Math.floor(r);
      const c0 = Math.floor(c);
      const r1 = Math.min(r0 + 1, rows - 1);
      const c1 = Math.min(c0 + 1, cols - 1);
      const fr = r - r0;
      const fc = c - c0;
      const e = grids.elevation;
      const zVal =
        e[r0][c0] * (1 - fr) * (1 - fc) +
        e[r0][c1] * (1 - fr) * fc +
        e[r1][c0] * fr * (1 - fc) +
        e[r1][c1] * fr * fc;
      return zVal - sceneCtxRef.current.elevOrigin;
    };

    const applyTerrainGeometry = (st: SimulationStatePayload) => {
      const ctx = sceneCtxRef.current;
      if (!ctx || !st.grids) return;
      const [rows, cols] = st.grid_shape;
      const [dyM, dxM] = st.resolution_m;
      const elev = st.grids.elevation;
      const tclass = st.grids.terrain_class_ids;
      const cost = st.grids.total_cost;
      const unc = st.grids.uncertainty;

      let zMin = Infinity;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          if (elev[r][c] < zMin) zMin = elev[r][c];
        }
      }
      ctx.elevOrigin = Number.isFinite(zMin) ? zMin : 0;

      if (!ctx.terrainMesh) {
        const geo = new THREE.PlaneGeometry((cols - 1) * dxM, (rows - 1) * dyM, cols - 1, rows - 1);
        geo.rotateX(-Math.PI / 2);
        const mesh = new THREE.Mesh(geo, terrainMat);
        mesh.receiveShadow = !isSoftwareGl;
        scene.add(mesh);
        ctx.terrainGeo = geo;
        ctx.terrainMesh = mesh;

        // Add atmospheric scatter rocks across non-path terrain
        const pebbleCount = isSoftwareGl ? 140 : 340;
        const pebbleGeo = new THREE.DodecahedronGeometry(0.18, 0);
        const pebbleMat = new THREE.MeshStandardMaterial({
          color: 0x6e412d,
          roughness: 0.94,
          flatShading: true,
        });
        const inst = new THREE.InstancedMesh(pebbleGeo, pebbleMat, pebbleCount);
        const dummy = new THREE.Object3D();
        let seed = 42;
        const rand = () => {
          seed = (seed * 16807) % 2147483647;
          return (seed - 1) / 2147483646;
        };
        for (let i = 0; i < pebbleCount; i++) {
          const rf = rand() * (rows - 1);
          const cf = rand() * (cols - 1);
          const { x, z } = gridToWorldXZ(st, rf, cf);
          const y = sampleSurfaceY(st, rf, cf);
          dummy.position.set(x, y + 0.05, z);
          dummy.rotation.set(rand() * 3, rand() * 3, rand() * 3);
          const s = 0.5 + rand() * 1.4;
          dummy.scale.set(s, s * 0.65, s);
          dummy.updateMatrix();
          inst.setMatrixAt(i, dummy.matrix);
        }
        inst.instanceMatrix.needsUpdate = true;
        scene.add(inst);
        ctx.scatterMesh = inst;
      }

      const geo = ctx.terrainGeo!;
      const pos = geo.attributes.position;
      const colors = new Float32Array(pos.count * 3);

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const vIdx = r * cols + c;
          pos.setY(vIdx, elev[r][c] - ctx.elevOrigin);

          const cid = tclass[r][c];
          let cr = 0.72,
            cg = 0.48,
            cb = 0.34;
          if (cid === 1) {
            cr = 0.52;
            cg = 0.49;
            cb = 0.48;
          } else if (cid === 2) {
            cr = 0.88;
            cg = 0.68;
            cb = 0.42;
          } else if (cid === 3) {
            cr = 0.62;
            cg = 0.3;
            cb = 0.22;
          } else if (cid === 255) {
            cr = 0.25;
            cg = 0.25;
            cb = 0.28;
          }

          if (st.show_cost_overlay) {
            const cv = cost[r][c];
            if (cv < 0) {
              cr = 0.72;
              cg = 0.14;
              cb = 0.14;
            } else {
              const t = Math.min(1, cv / 12);
              cr = cr * 0.4 + 0.95 * t * 0.6;
              cg = cg * 0.4 + 0.68 * (1 - t) * 0.6;
              cb = cb * 0.4 + 0.15 * (1 - t) * 0.6;
            }
          }

          if (st.show_uncertainty_overlay) {
            const uv = unc[r][c];
            cr = cr * (1 - 0.5 * uv) + 0.78 * 0.5 * uv;
            cg = cg * (1 - 0.5 * uv) + 0.26 * 0.5 * uv;
            cb = cb * (1 - 0.5 * uv) + 0.92 * 0.5 * uv;
          }

          colors[vIdx * 3] = cr;
          colors[vIdx * 3 + 1] = cg;
          colors[vIdx * 3 + 2] = cb;
        }
      }
      pos.needsUpdate = true;
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geo.computeVertexNormals();
    };

    const buildDualTrackRibbon = (
      pts: [number, number][],
      st: SimulationStatePayload,
      colorHex: number,
      widthHalfM: number,
      yOffset: number,
      opacity: number
    ) => {
      if (pts.length < 2) return;
      for (const side of [-widthHalfM, widthHalfM]) {
        const curvePts: THREE.Vector3[] = [];
        for (let i = 0; i < pts.length; i++) {
          const [r, c] = pts[i];
          const prev = pts[Math.max(0, i - 1)];
          const next = pts[Math.min(pts.length - 1, i + 1)];
          const pA = gridToWorldXZ(st, prev[0], prev[1]);
          const pB = gridToWorldXZ(st, next[0], next[1]);
          const dx = pB.x - pA.x;
          const dz = pB.z - pA.z;
          const len = Math.hypot(dx, dz) || 1;
          const nx = -dz / len;
          const nz = dx / len;

          const base = gridToWorldXZ(st, r, c);
          const wx = base.x + nx * side;
          const wz = base.z + nz * side;
          const wy = sampleSurfaceY(st, r, c) + yOffset;
          curvePts.push(new THREE.Vector3(wx, wy, wz));
        }
        const curve = new THREE.CatmullRomCurve3(curvePts, false, "catmullrom", 0.2);
        const tubeGeo = new THREE.TubeGeometry(curve, Math.max(16, pts.length * 4), 0.085, 6, false);
        const tubeMat = new THREE.MeshBasicMaterial({
          color: colorHex,
          transparent: opacity < 1,
          opacity,
        });
        routesGroup.add(new THREE.Mesh(tubeGeo, tubeMat));
      }
    };

    const syncDynamicVisuals = (st: SimulationStatePayload) => {
      const ctx = sceneCtxRef.current;
      if (!ctx) return;

      // Sync Rover Pose + Physical Slope Pitch/Roll + Wheel Rotation
      const roverXZ = gridToWorldXZ(st, st.rover.row_f, st.rover.col_f);
      const roverY = sampleSurfaceY(st, st.rover.row_f, st.rover.col_f);
      ctx.rover.root.position.set(roverXZ.x, roverY, roverXZ.z);
      ctx.rover.root.rotation.y = -(st.rover.heading_rad || 0);
      ctx.rover.tiltGroup.rotation.z = THREE.MathUtils.degToRad(st.physics.pitch_deg || 0);
      ctx.rover.tiltGroup.rotation.x = THREE.MathUtils.degToRad(-(st.physics.roll_deg || 0));

      const spin = st.physics.wheel_spin_rad || 0;
      for (const w of ctx.rover.wheels) {
        w.rotation.z = -spin;
      }

      // Sync Start & Goal Beacons
      const sXZ = gridToWorldXZ(st, st.start_cell[0], st.start_cell[1]);
      ctx.startBeacon.position.set(
        sXZ.x,
        sampleSurfaceY(st, st.start_cell[0], st.start_cell[1]) + 0.1,
        sXZ.z
      );
      const gXZ = gridToWorldXZ(st, st.goal_cell[0], st.goal_cell[1]);
      ctx.goalBeacon.position.set(
        gXZ.x,
        sampleSurfaceY(st, st.goal_cell[0], st.goal_cell[1]) + 0.08,
        gXZ.z
      );

      // Rebuild Boulders cleanly
      for (const [, item] of ctx.boulderMeshes) {
        ctx.bouldersGroup.remove(item.mesh);
        ctx.bouldersGroup.remove(item.ring);
        item.mesh.geometry.dispose();
        (item.mesh.material as THREE.Material).dispose();
        item.ring.geometry.dispose();
        (item.ring.material as THREE.Material).dispose();
      }
      ctx.boulderMeshes.clear();

      for (const obs of st.obstacles) {
        const radM = obs.radius_cells > 0 ? 1.75 : 0.92;
        const hash = obs.obstacle_id.split("").reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
        const mesh = createBoulderMesh(radM, obs.height_m, hash);
        const { x, z } = gridToWorldXZ(st, obs.row, obs.col);
        const y = sampleSurfaceY(st, obs.row, obs.col);
        mesh.position.set(x, y + obs.height_m * 0.35, z);
        mesh.userData.obstacleId = obs.obstacle_id;

        const ring = new THREE.Mesh(
          new THREE.RingGeometry(radM * 1.05, radM * 1.24, 24),
          new THREE.MeshBasicMaterial({
            color: 0xf87171,
            transparent: true,
            opacity: 0.45,
            side: THREE.DoubleSide,
          })
        );
        ring.rotation.x = -Math.PI / 2;
        ring.position.set(x, y + 0.12, z);

        ctx.bouldersGroup.add(mesh, ring);
        ctx.boulderMeshes.set(obs.obstacle_id, { mesh, ring, obs });
      }

      // Rebuild Routes (dispose old geometries)
      while (ctx.routesGroup.children.length > 0) {
        const child = ctx.routesGroup.children.pop() as THREE.Mesh | THREE.Line;
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }

      if (st.routes.travelled_points.length >= 2) {
        buildDualTrackRibbon(st.routes.travelled_points, st, 0x2563eb, 0.56, 0.12, 0.95);
      }
      if (st.routes.active_route.length >= 2) {
        buildDualTrackRibbon(st.routes.active_route, st, 0x00e5ff, 0.62, 0.18, 0.92);
      }
      if (st.show_previous_route && st.routes.previous_route.length >= 2) {
        const linePts = st.routes.previous_route.map(([r, c]) => {
          const xz = gridToWorldXZ(st, r, c);
          return new THREE.Vector3(xz.x, sampleSurfaceY(st, r, c) + 0.25, xz.z);
        });
        const lineGeo = new THREE.BufferGeometry().setFromPoints(linePts);
        const lineMat = new THREE.LineBasicMaterial({
          color: 0x94a3b8,
          transparent: true,
          opacity: 0.6,
        });
        ctx.routesGroup.add(new THREE.Line(lineGeo, lineMat));
      }
    };

    sceneCtxRef.current = {
      scene,
      camera,
      controls,
      terrainMesh: null,
      terrainGeo: null,
      elevOrigin: 0,
      rover,
      bouldersGroup,
      boulderMeshes: new Map(),
      routesGroup,
      startBeacon,
      goalBeacon,
      scatterMesh: null,
      dragGhost,
      applyTerrainGeometry,
      syncDynamicVisuals,
      sampleSurfaceY,
    };

    // Raycast interaction for 3D dragging & terrain editing
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let draggedObsId: string | null = null;
    let dragTargetCell: [number, number] | null = null;

    const hitToGrid = (point: THREE.Vector3, st: SimulationStatePayload): [number, number] => {
      const [rows, cols] = st.grid_shape;
      const [dyM, dxM] = st.resolution_m;
      const colF = point.x / dxM + (cols - 1) / 2;
      const rowF = point.z / dyM + (rows - 1) / 2;
      return [
        Math.max(0, Math.min(rows - 1, Math.round(rowF))),
        Math.max(0, Math.min(cols - 1, Math.round(colF))),
      ];
    };

    const setPointer = (ev: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    };

    const onPointerDown = (ev: PointerEvent) => {
      const ctx = sceneCtxRef.current;
      const props = latestPropsRef.current;
      if (!ctx || !ctx.terrainMesh) return;
      setPointer(ev);
      raycaster.setFromCamera(pointer, camera);

      if (props.interactionMode === "drag") {
        const meshes = Array.from(ctx.boulderMeshes.values()).map((b) => b.mesh);
        const hits = raycaster.intersectObjects(meshes, false);
        if (hits.length > 0) {
          draggedObsId = String(hits[0].object.userData.obstacleId || "");
          controls.enabled = false;
          dragGhost.visible = true;
          const entry = ctx.boulderMeshes.get(draggedObsId);
          if (entry) {
            dragGhost.position.copy(entry.ring.position);
            dragTargetCell = [entry.obs.row, entry.obs.col];
          }
        }
        return;
      }

      if (props.interactionMode === "remove") {
        const meshes = Array.from(ctx.boulderMeshes.values()).map((b) => b.mesh);
        const hits = raycaster.intersectObjects(meshes, false);
        if (hits.length > 0) {
          const obsId = String(hits[0].object.userData.obstacleId || "");
          const entry = ctx.boulderMeshes.get(obsId);
          if (entry) props.onRemoveObstacle(entry.obs);
        }
        return;
      }

      const terrainHits = raycaster.intersectObject(ctx.terrainMesh, false);
      if (terrainHits.length > 0) {
        const [row, col] = hitToGrid(terrainHits[0].point, props.simState);
        if (props.interactionMode === "add") {
          props.onAddObstacle(row, col, props.rockRadiusCells);
        } else if (props.interactionMode === "paint") {
          props.onPaintTerrain([[row, col]], props.terrainBrush);
        } else if (props.interactionMode === "start") {
          props.onSetStart(row, col);
        } else if (props.interactionMode === "goal") {
          props.onSetGoal(row, col);
        }
      }
    };

    const onPointerMove = (ev: PointerEvent) => {
      const ctx = sceneCtxRef.current;
      const props = latestPropsRef.current;
      if (!draggedObsId || !ctx || !ctx.terrainMesh) return;
      setPointer(ev);
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObject(ctx.terrainMesh, false);
      if (hits.length > 0) {
        const cell = hitToGrid(hits[0].point, props.simState);
        dragTargetCell = cell;
        const xz = gridToWorldXZ(props.simState, cell[0], cell[1]);
        const y = sampleSurfaceY(props.simState, cell[0], cell[1]);
        dragGhost.position.set(xz.x, y + 0.22, xz.z);
      }
    };

    const onPointerUp = () => {
      const props = latestPropsRef.current;
      if (draggedObsId && dragTargetCell) {
        props.onMoveObstacle(draggedObsId, dragTargetCell[0], dragTargetCell[1]);
      }
      draggedObsId = null;
      dragTargetCell = null;
      dragGhost.visible = false;
      controls.enabled = true;
    };

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);

    const resizeObserver = new ResizeObserver(() => {
      if (!container) return;
      const w = Math.max(1, container.clientWidth);
      const h = Math.max(1, container.clientHeight);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    resizeObserver.observe(container);

    let rafId = 0;
    let lastDraw = 0;
    const desiredCamPos = new THREE.Vector3();
    const desiredTarget = new THREE.Vector3();

    const animate = (now: number) => {
      rafId = requestAnimationFrame(animate);
      if (isSoftwareGl && now - lastDraw < 40) return;
      lastDraw = now;

      const ctx = sceneCtxRef.current;
      const { simState: st, cameraPreset: preset } = latestPropsRef.current;
      if (ctx) {
        // Gentle mastcam scan & goal beacon pulse
        ctx.rover.mastHead.rotation.y = Math.sin(now * 0.0011) * 0.22;
        ctx.goalBeacon.rotation.y = now * 0.0012;

        if (preset === "chase" && !draggedObsId) {
          const rx = ctx.rover.root.position.x;
          const ry = ctx.rover.root.position.y;
          const rz = ctx.rover.root.position.z;
          const h = st.rover.heading_rad || 0;
          const fwdX = Math.cos(h);
          const fwdZ = Math.sin(h);
          desiredCamPos.set(rx - fwdX * 9.5, ry + 5.2, rz - fwdZ * 9.5);
          desiredTarget.set(rx + fwdX * 7.5, ry + 1.0, rz + fwdZ * 7.5);
          camera.position.lerp(desiredCamPos, 0.08);
          controls.target.lerp(desiredTarget, 0.1);
        }
      }

      controls.update();
      renderer.render(scene, camera);
    };
    rafId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafId);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      controls.dispose();
      regolithTex.dispose();
      envTex.dispose();
      pmrem.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
      sceneCtxRef.current = null;
    };
  }, []);

  // Re-apply terrain mesh whenever grids or overlay toggles update
  useEffect(() => {
    const ctx = sceneCtxRef.current;
    if (!ctx || !simState.grids) return;
    ctx.applyTerrainGeometry(simState);
    ctx.syncDynamicVisuals(simState);
  }, [
    simState.grids,
    simState.show_cost_overlay,
    simState.show_uncertainty_overlay,
  ]);

  // Fast update for rover pose, obstacles, and routes
  useEffect(() => {
    const ctx = sceneCtxRef.current;
    if (!ctx) return;
    ctx.syncDynamicVisuals(simState);
  }, [simState]);

  // Snap camera cleanly when user switches Orbit / Top-Down preset
  useEffect(() => {
    const ctx = sceneCtxRef.current;
    if (!ctx) return;
    if (cameraPreset === "orbit") {
      ctx.camera.position.set(-22, 24, 34);
      ctx.controls.target.set(0, 1.5, 0);
    } else if (cameraPreset === "top") {
      ctx.camera.position.set(0, 58, 0.1);
      ctx.controls.target.set(0, 0, 0);
    }
  }, [cameraPreset]);

  return (
    <div className="flex flex-col h-full bg-[#120f0d] border border-[#2b221c] rounded-md overflow-hidden">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-[#2b221c] bg-[#171210] text-xs">
        <span className="font-semibold tracking-wide text-[#e7d8c9]">
          3D MARTIAN SURFACE · ROCKER-BOGIE ROVER VIEW
        </span>
        <span className="font-mono text-[11px] text-[#b4a494]">
          {interactionMode === "drag"
            ? "Drag 3D boulders or orbit scene"
            : `Click 3D terrain (${interactionMode.toUpperCase()})`}
        </span>
      </div>
      <div ref={containerRef} className="flex-1 w-full min-h-0 relative" />
    </div>
  );
};
