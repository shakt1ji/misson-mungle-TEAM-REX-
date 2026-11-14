import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import path from "node:path";
import readline from "node:readline";
import express from "express";
import { createServer as createViteServer } from "vite";

interface PendingRequest {
  id: number;
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
}

class PythonSimulationBridge {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private readyPromise: Promise<void> | null = null;
  private reqCounter = 0;
  private pending = new Map<number, PendingRequest>();
  private queue: Promise<unknown> = Promise.resolve();

  private startProcess(): Promise<void> {
    if (this.readyPromise) return this.readyPromise;

    this.readyPromise = new Promise<void>((resolve, reject) => {
      const scriptPath = path.resolve(process.cwd(), "MARS-09", "simulation", "bridge_server.py");
      const child = spawn("python3", ["-u", scriptPath], {
        cwd: path.resolve(process.cwd(), "MARS-09"),
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
        stdio: ["pipe", "pipe", "pipe"],
      });
      this.proc = child;

      let settledReady = false;
      const rl = readline.createInterface({ input: child.stdout });

      rl.on("line", (line) => {
        const trimmed = line.trim();
        if (!trimmed) return;
        try {
          const msg = JSON.parse(trimmed) as Record<string, unknown>;
          if (!settledReady && msg.ready) {
            settledReady = true;
            resolve();
            return;
          }
          const reqId = typeof msg.id === "number" ? msg.id : null;
          if (reqId !== null && this.pending.has(reqId)) {
            const entry = this.pending.get(reqId)!;
            this.pending.delete(reqId);
            entry.resolve(msg);
          }
        } catch (err) {
          console.error("[bridge stdout parse error]", err, line);
        }
      });

      child.stderr.on("data", (chunk) => {
        console.error("[mars09-python]", chunk.toString());
      });

      child.on("exit", (code) => {
        console.warn(`[mars09-python] exited with code ${code}`);
        this.proc = null;
        this.readyPromise = null;
        for (const [, entry] of this.pending) {
          entry.reject(new Error(`Python simulation bridge exited (${code}).`));
        }
        this.pending.clear();
        if (!settledReady) {
          reject(new Error(`Python simulation bridge failed to start (${code}).`));
        }
      });
    });

    return this.readyPromise;
  }

  public call(cmd: string, args: Record<string, unknown> = {}, includeGrids = true): Promise<Record<string, unknown>> {
    const run = async (): Promise<Record<string, unknown>> => {
      await this.startProcess();
      if (!this.proc || !this.proc.stdin.writable) {
        throw new Error("Python simulation bridge stdin unavailable.");
      }
      const id = ++this.reqCounter;
      const payload = JSON.stringify({ id, cmd, args, include_grids: includeGrids });

      return await new Promise<Record<string, unknown>>((resolve, reject) => {
        const timer = setTimeout(() => {
          this.pending.delete(id);
          reject(new Error(`Simulation command '${cmd}' timed out.`));
        }, 12000);

        this.pending.set(id, {
          id,
          resolve: (val) => {
            clearTimeout(timer);
            resolve(val as Record<string, unknown>);
          },
          reject: (err) => {
            clearTimeout(timer);
            reject(err);
          },
        });

        this.proc!.stdin.write(payload + "\n");
      });
    };

    const next = this.queue.then(run, run);
    this.queue = next.catch(() => undefined);
    return next;
  }
}

async function bootstrapServer() {
  const app = express();
  app.use(express.json({ limit: "2mb" }));

  const bridge = new PythonSimulationBridge();

  const handleBridge = async (
    req: express.Request,
    res: express.Response,
    cmd: string,
    args: Record<string, unknown> = {},
    includeGrids = true
  ) => {
    try {
      const result = await bridge.call(cmd, args, includeGrids);
      if (!result.ok) {
        res.status(400).json({ ok: false, error: result.error || "Command rejected." });
        return;
      }
      res.json(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      res.status(500).json({ ok: false, error: message });
    }
  };

  app.get("/api/sim/state", (req, res) => {
    const includeGrids = req.query.grids !== "0";
    void handleBridge(req, res, "get_state", {}, includeGrids);
  });

  app.get("/api/sim/scenarios", (req, res) => {
    void handleBridge(req, res, "scenarios", {}, false);
  });

  app.post("/api/sim/scenario", (req, res) => {
    void handleBridge(req, res, "load_scenario", req.body || {}, true);
  });

  app.post("/api/sim/start", (req, res) => {
    void handleBridge(req, res, "start", {}, false);
  });

  app.post("/api/sim/pause", (req, res) => {
    void handleBridge(req, res, "pause", {}, false);
  });

  app.post("/api/sim/resume", (req, res) => {
    void handleBridge(req, res, "resume", {}, false);
  });

  app.post("/api/sim/reset", (req, res) => {
    void handleBridge(req, res, "reset", {}, true);
  });

  app.post("/api/sim/step", (req, res) => {
    const includeGrids = req.body?.include_grids === true;
    void handleBridge(req, res, "step", req.body || {}, includeGrids);
  });

  app.post("/api/sim/replan", (req, res) => {
    void handleBridge(req, res, "replan", {}, true);
  });

  app.post("/api/sim/controls", (req, res) => {
    void handleBridge(req, res, "set_controls", req.body || {}, false);
  });

  app.post("/api/sim/obstacle/add", (req, res) => {
    void handleBridge(req, res, "add_obstacle", req.body || {}, true);
  });

  app.post("/api/sim/obstacle/move", (req, res) => {
    void handleBridge(req, res, "move_obstacle", req.body || {}, true);
  });

  app.post("/api/sim/obstacle/remove", (req, res) => {
    void handleBridge(req, res, "remove_obstacle", req.body || {}, true);
  });

  app.post("/api/sim/terrain/paint", (req, res) => {
    void handleBridge(req, res, "paint_terrain", req.body || {}, true);
  });

  app.post("/api/sim/endpoints", (req, res) => {
    void handleBridge(req, res, "set_start_goal", req.body || {}, true);
  });

  if (process.env.NODE_ENV === "production") {
    const distDir = path.resolve(process.cwd(), "dist");
    app.use(express.static(distDir));
    app.get("/{*path}", (_req, res) => {
      res.sendFile(path.join(distDir, "index.html"));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  }

  const port = Number(process.env.PORT) || 3000;
  app.listen(port, "0.0.0.0", () => {
    console.log(`MARS-09 server listening on http://0.0.0.0:${port}`);
  });
}

void bootstrapServer();
