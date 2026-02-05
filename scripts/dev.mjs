import { spawn } from "node:child_process";

function run(command, args, opts = {}) {
  const child = spawn(command, args, {
    stdio: "inherit",
    env: { ...process.env, ...(opts.env ?? {}) },
    cwd: opts.cwd,
  });
  child.on("error", (err) => {
    console.error(`[dev] failed to start ${command}:`, err);
    process.exitCode = 1;
  });
  return child;
}

let shuttingDown = false;
function shutdown(children, code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;

  for (const child of children) {
    if (!child || child.killed) continue;
    try {
      child.kill("SIGINT");
    } catch {
      // ignore
    }
  }

  setTimeout(() => {
    for (const child of children) {
      if (!child || child.killed) continue;
      try {
        child.kill("SIGKILL");
      } catch {
        // ignore
      }
    }
    process.exit(code);
  }, 2500);
}

const children = [];
const frontend = run("pnpm", ["-C", "frontend", "dev"]);
children.push(frontend);

process.on("SIGINT", () => shutdown(children, 0));
process.on("SIGTERM", () => shutdown(children, 0));
frontend.on("exit", (code) => shutdown(children, code ?? 0));

// Give Next.js a moment to start before launching the desktop app.
setTimeout(() => {
  if (shuttingDown) return;

  const backend = run("uv", ["run", "python", "app.py"], {
    cwd: "backend",
    env: {
      MUSIC_LOOPER_DEV: "1",
    },
  });
  children.push(backend);

  backend.on("exit", (code) => shutdown(children, code ?? 0));
}, 700);
