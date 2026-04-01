import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";

const app = express();
const PORT = process.env.PORT || 8000;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const SEQREP_DIR = path.join(ROOT, "SeqRep");
const SCRIPTS_DIR = path.join(SEQREP_DIR, "scripts");
const INPUT_DIR = path.join(SEQREP_DIR, "data", "inputs");
const OUTPUT_DIR = path.join(SEQREP_DIR, "data", "outputs");
const LOCAL_BIN_DIR = path.join(SEQREP_DIR, ".local", "bin");
const PYDEPS_DIR = path.join(SEQREP_DIR, ".pydeps");
const PYTHON_EXEC = fs.existsSync(path.join(LOCAL_BIN_DIR, "python"))
  ? path.join(LOCAL_BIN_DIR, "python")
  : fs.existsSync(path.join(LOCAL_BIN_DIR, "python3"))
  ? path.join(LOCAL_BIN_DIR, "python3")
  : "python";
const CACHE_DIR = path.join(SCRIPTS_DIR, ".cache");
const MPL_CACHE_DIR = path.join(CACHE_DIR, "matplotlib");
const TORCH_CACHE_DIR = path.join(CACHE_DIR, "torch");
const TORCH_CHECKPOINTS_DIR = path.join(TORCH_CACHE_DIR, "hub", "checkpoints");
const XDG_CACHE_DIR = path.join(CACHE_DIR, "xdg");

app.use(cors());
app.use(express.json({ limit: "20mb" }));

function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function genPts(n, spread, seed) {
  const r = seededRandom(seed);
  return Array.from({ length: n }, (_, i) => ({
    x: (r() - 0.5) * spread * 2,
    y: (r() - 0.5) * spread * 2,
    label: `seq_${String(i + 1).padStart(3, "0")}`,
  }));
}

function parseFasta(text) {
  const lines = text.split("\n");
  const seqs = [];
  let cur = null;
  for (const line of lines) {
    const s = line.trim();
    if (!s) continue;
    if (s.startsWith(">")) {
      if (cur) seqs.push(cur);
      cur = { header: s.slice(1).trim(), sequence: "" };
    } else if (cur) {
      cur.sequence += s;
    }
  }
  if (cur) seqs.push(cur);
  return seqs.map((x, idx) => ({
    id: x.header || `seq_${String(idx + 1).padStart(3, "0")}`,
    length: x.sequence.length,
  }));
}

const REPRO = genPts(100, 12, 42);
const ESM = genPts(100, 8, 99);

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, service: "capstone-backend" });
});

app.get("/api/dashboard", (_req, res) => {
  res.json({
    hero: {
      sequences: 100,
      models: 2,
      replicas: 6,
      status: "reproducible",
    },
    reproducibility: {
      unirep: REPRO,
      esm2: ESM,
      bitwiseMatch: true,
    },
    speed: {
      single: { unirep: 1.2, esm2: 0.05 },
      multi100: { unirep: 45.3, esm2: 1.8 },
    },
  });
});

function findLatestFile(dir, exts) {
  if (!fs.existsSync(dir)) return null;
  const files = fs
    .readdirSync(dir)
    .filter((f) => exts.some((e) => f.toLowerCase().endsWith(e)))
    .map((name) => {
      const absPath = path.join(dir, name);
      const stat = fs.statSync(absPath);
      return { name, absPath, mtimeMs: stat.mtimeMs };
    })
    .sort((a, b) => b.mtimeMs - a.mtimeMs);
  return files[0] || null;
}

app.get("/api/download/unirep/first", (_req, res) => {
  const dir = path.join(OUTPUT_DIR, "mlstm", "runs");
  const latest = findLatestFile(dir, [".h5"]);
  if (!latest) {
    res.status(404).json({ error: "UniRep output not found" });
    return;
  }
  res.download(latest.absPath, latest.name);
});

app.get("/api/download/:model/:name", (req, res) => {
  const { model, name } = req.params;
  const fileName = decodeURIComponent(name || "embedding.h5");
  const subdir = model === "unirep" ? ["mlstm", "runs"] : ["esm", "runs"];
  const absPath = path.join(OUTPUT_DIR, ...subdir, fileName);

  if (!fs.existsSync(absPath)) {
    res.status(404).json({ error: "File not found" });
    return;
  }

  res.download(absPath, fileName);
});

app.get("/api/plot/unirep/latest", (_req, res) => {
  const dir = path.join(OUTPUT_DIR, "mlstm", "plot");
  const latest = findLatestFile(dir, [".png", ".jpg", ".jpeg", ".webp"]);
  if (!latest) {
    res.status(404).json({ error: "UniRep plot not found" });
    return;
  }
  res.sendFile(latest.absPath);
});

app.get("/api/plot/esm/latest", (_req, res) => {
  const absPath = path.join(OUTPUT_DIR, "esm", "plot", "tsne_esm.png");
  if (!fs.existsSync(absPath)) {
    res.status(404).json({ error: "ESM plot not found" });
    return;
  }
  res.sendFile(absPath);
});

app.get("/api/plot/:model/:name", (req, res) => {
  const { model, name } = req.params;
  const fileName = decodeURIComponent(name || "plot.png");
  const subdir = model === "unirep" ? ["mlstm", "plot"] : ["esm", "plot"];
  const absPath = path.join(OUTPUT_DIR, ...subdir, fileName);

  if (!fs.existsSync(absPath)) {
    res.status(404).json({ error: "Plot not found" });
    return;
  }

  res.sendFile(absPath);
});

const jobs = new Map();

function ensureDirs() {
  fs.mkdirSync(INPUT_DIR, { recursive: true });
  fs.mkdirSync(path.join(OUTPUT_DIR, "mlstm", "runs"), { recursive: true });
  fs.mkdirSync(path.join(OUTPUT_DIR, "mlstm", "plot"), { recursive: true });
  fs.mkdirSync(path.join(OUTPUT_DIR, "esm", "runs"), { recursive: true });
  fs.mkdirSync(path.join(OUTPUT_DIR, "esm", "plot"), { recursive: true });
  fs.mkdirSync(MPL_CACHE_DIR, { recursive: true });
  fs.mkdirSync(TORCH_CHECKPOINTS_DIR, { recursive: true });
  fs.mkdirSync(XDG_CACHE_DIR, { recursive: true });
}

function copyIfExists(src, dst) {
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, dst);
    return true;
  }
  return false;
}

function prepareRuntimeEnv(extraEnv = {}) {
  const modelA = "esm2_t33_650M_UR50D.pt";
  const modelB = "esm2_t33_650M_UR50D-contact-regression.pt";
  copyIfExists(path.join(SCRIPTS_DIR, modelA), path.join(TORCH_CHECKPOINTS_DIR, modelA));
  copyIfExists(path.join(SCRIPTS_DIR, modelB), path.join(TORCH_CHECKPOINTS_DIR, modelB));

  return {
    MPLCONFIGDIR: MPL_CACHE_DIR,
    TORCH_HOME: TORCH_CACHE_DIR,
    XDG_CACHE_HOME: XDG_CACHE_DIR,
    PYTHONPATH: PYDEPS_DIR,
    PATH: `${LOCAL_BIN_DIR}:${process.env.PATH || ""}`,
    ...extraEnv,
  };
}

function writeInputFasta(files, jobId) {
  ensureDirs();
  const fullPath = path.join(INPUT_DIR, `frontend_${jobId}.fasta`);

  // Keep frontend FASTA exactly as provided (no sanitization/reformat/replacement)
  // to ensure backend orchestration matches direct script execution.
  const content = files
    .map((f) => (typeof f?.content === "string" ? f.content : ""))
    .filter((txt) => txt.length > 0)
    .join("\n");

  fs.writeFileSync(fullPath, content, "utf8");
  return fullPath;
}

function parseProgressLine(line) {
  const m = line.match(/(\d{1,3})%/);
  if (m) return Math.min(99, Number(m[1]));
  return null;
}

function appendJobLog(jobId, line) {
  const job = jobs.get(jobId);
  if (!job || !line) return;
  const logs = Array.isArray(job.logs) ? job.logs : [];
  logs.push(line);
  job.logs = logs.length > 300 ? logs.slice(logs.length - 300) : logs;
  jobs.set(jobId, job);
}

function runScript(jobId, scriptName, phaseLabel, env = {}) {
  return new Promise((resolve, reject) => {
    const job = jobs.get(jobId);
    if (!job) return reject(new Error("Job not found"));

    job.status = phaseLabel;
    job.phase = phaseLabel;
    job.message = `Running ${scriptName}...`;
    jobs.set(jobId, job);
    appendJobLog(jobId, `[${phaseLabel}] Running ${scriptName}...`);

    const runtimeEnv = prepareRuntimeEnv(env);
    const proc = spawn(PYTHON_EXEC, [scriptName], {
      cwd: SCRIPTS_DIR,
      env: { ...process.env, ...runtimeEnv },
      stdio: ["ignore", "pipe", "pipe"],
    });

    const onData = (buf) => {
      const txt = buf.toString();
      const lines = txt.split(/\r?\n/).filter(Boolean);
      const current = jobs.get(jobId);
      if (!current) return;
      for (const ln of lines) {
        current.message = ln;
        appendJobLog(jobId, `[${phaseLabel}] ${ln}`);
        const p = parseProgressLine(ln);
        if (p !== null) current.progress = p;
      }
      jobs.set(jobId, current);
    };

    proc.stdout.on("data", onData);
    proc.stderr.on("data", onData);

    proc.on("close", (code) => {
      if (code === 0) resolve();
      else {
        const latest = jobs.get(jobId);
        const logs = latest?.logs || [];
        const tail = logs.slice(-12).join("\n");
        const detail = tail || latest?.message || "unknown error";
        reject(new Error(`${scriptName} exited with code ${code}\n${detail}`));
      }
    });
  });
}

function listModelOutputs(modelName) {
  const subdir = modelName === "unirep" ? ["mlstm", "runs"] : ["esm", "runs"];
  const dir = path.join(OUTPUT_DIR, ...subdir);
  if (!fs.existsSync(dir)) return [];

  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".h5"))
    .map((name) => {
      const absPath = path.join(dir, name);
      const stat = fs.statSync(absPath);
      return {
        model: modelName,
        name,
        absPath,
        mtimeMs: stat.mtimeMs,
        downloadUrl: `/api/download/${modelName}/${encodeURIComponent(name)}`,
      };
    })
    .sort((a, b) => b.mtimeMs - a.mtimeMs);
}

function collectOutputs(model) {
  const outputs = [];
  if (model === "both" || model === "unirep") outputs.push(...listModelOutputs("unirep"));
  if (model === "both" || model === "esm2") outputs.push(...listModelOutputs("esm2"));
  return outputs;
}

function buildResult(files, model) {
  const produced = collectOutputs(model);
  const unirepOutputs = produced.filter((x) => x.model === "unirep");
  const esmOutputs = produced.filter((x) => x.model === "esm2");

  const parsed = files.map((f) => ({
    fileName: f.name || "input.fasta",
    sequenceCount: parseFasta(f.content || "").length,
    previews: parseFasta(f.content || "").slice(0, 3),
    outputs: {
      unirep:
        model === "both" || model === "unirep"
          ? unirepOutputs.map((o) => ({
              name: o.name,
              downloadUrl: `/api/download/unirep/${encodeURIComponent(o.name)}`,
            }))
          : [],
      esm2:
        model === "both" || model === "esm2"
          ? esmOutputs.map((o) => ({
              name: o.name,
              downloadUrl: `/api/download/esm2/${encodeURIComponent(o.name)}`,
            }))
          : [],
    },
  }));

  const esmPlotPath = path.join(OUTPUT_DIR, "esm", "plot", "tsne_esm.png");
  const plots = {
    esm2Url: fs.existsSync(esmPlotPath)
      ? `/api/plot/esm/tsne_esm.png?ts=${Date.now()}`
      : null,
  };

  const downloads = [
    ...unirepOutputs.map((o) => ({
      model: "unirep",
      name: o.name,
      downloadUrl: `/api/download/unirep/${encodeURIComponent(o.name)}`,
      dim: "1900d",
    })),
    ...esmOutputs.map((o) => ({
      model: "esm2",
      name: o.name,
      downloadUrl: `/api/download/esm2/${encodeURIComponent(o.name)}`,
      dim: "1280d",
    })),
  ];

  return {
    model,
    files: parsed,
    downloads,
    plots,
    reproducibility: {
      unirep: model === "both" || model === "unirep" ? REPRO : null,
      esm2: model === "both" || model === "esm2" ? ESM : null,
      pipeline: "StandardScaler → PCA (95%) → t-SNE (perplexity=50, iter=10k, seed=42)",
    },
  };
}

app.post("/api/run", async (req, res) => {
  const { files = [], model = "both" } = req.body ?? {};
  const jobId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

  jobs.set(jobId, {
    status: "queued",
    phase: "queued",
    progress: 1,
    message: "Queued",
    result: null,
    error: null,
    logs: ["[queued] Queued"],
    createdAt: Date.now(),
  });

  res.json({ jobId });

  try {
    const fastaPath = writeInputFasta(files, jobId);
    const sharedEnv = { FASTA_FILE: fastaPath, FASTA: fastaPath };

    if (model === "both" || model === "unirep") {
      await runScript(jobId, "unirep_r_auto_pipeline.py", "running_unirep", sharedEnv);
      const j = jobs.get(jobId);
      if (j) {
        j.progress = Math.max(j.progress, 50);
        jobs.set(jobId, j);
      }
    }

    if (model === "both" || model === "esm2") {
      await runScript(jobId, "esm_auto_pipeline.py", "running_esm2", sharedEnv);
      const j = jobs.get(jobId);
      if (j) {
        j.progress = Math.max(j.progress, 90);
        jobs.set(jobId, j);
      }
    }

    const result = buildResult(files, model);
    const done = jobs.get(jobId);
    if (done) {
      done.status = "complete";
      done.phase = "complete";
      done.progress = 100;
      done.message = "Complete";
      done.result = result;
      done.logs = [...(done.logs || []), "[complete] Complete"];
      jobs.set(jobId, done);
      setTimeout(() => jobs.delete(jobId), 1000 * 60 * 10);
    }
  } catch (err) {
    const failed = jobs.get(jobId);
    if (failed) {
      failed.status = "error";
      failed.phase = "error";
      failed.error = err.message;
      failed.message = err.message;
      failed.logs = [...(failed.logs || []), `[error] ${err.message}`];
      jobs.set(jobId, failed);
    }
  }
});

app.get("/api/run/:jobId", (req, res) => {
  const { jobId } = req.params;
  const job = jobs.get(jobId);
  if (!job) {
    res.status(404).json({ error: "Job not found" });
    return;
  }
  res.json({
    status: job.status,
    phase: job.phase,
    progress: job.progress,
    message: job.message,
    logs: job.logs || [],
    result: job.result,
    error: job.error,
  });
});

app.listen(PORT, () => {
  console.log(`Backend running at http://localhost:${PORT}`);
});
