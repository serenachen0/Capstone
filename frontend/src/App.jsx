import { useState, useEffect, useRef, createContext, useContext } from "react";
import Plotly from "plotly.js-dist-min";

// ─── Theme Context ──────────────────────────────────────────────────────────
const ThemeContext = createContext();
const useTheme = () => useContext(ThemeContext);

const themes = {
  light: {
    bg: "#f8fafc",
    surface: "#ffffff",
    text: "#0f172a",
    textSecondary: "#64748b",
    textMuted: "#94a3b8",
    border: "#e2e8f0",
    borderLight: "#f1f5f9",
    accent: "#0284c7",
    accentBg: "#f0f9ff",
    accentBorder: "#bae6fd",
    green: "#059669",
    greenBg: "#ecfdf5",
    greenBorder: "#a7f3d0",
    red: "#dc2626",
    redBg: "#fef2f2",
    redBorder: "#fecaca",
    amber: "#d97706",
    amberBg: "#fffbeb",
    amberBorder: "#fde68a",
    blue: "#2563eb",
    blueBg: "#eff6ff",
    blueBorder: "#bfdbfe",
    cardShadow: "0 1px 3px rgba(0,0,0,0.04)",
    plotGrid: "#f1f5f9",
    plotZero: "#e2e8f0",
    plotFont: "#64748b",
  },
  dark: {
    bg: "#0a0e1a",
    surface: "rgba(255,255,255,0.03)",
    text: "#e2e8f0",
    textSecondary: "rgba(255,255,255,0.5)",
    textMuted: "rgba(255,255,255,0.3)",
    border: "rgba(255,255,255,0.08)",
    borderLight: "rgba(255,255,255,0.04)",
    accent: "#38bdf8",
    accentBg: "rgba(14,165,233,0.1)",
    accentBorder: "rgba(14,165,233,0.25)",
    green: "#34d399",
    greenBg: "rgba(16,185,129,0.1)",
    greenBorder: "rgba(16,185,129,0.25)",
    red: "#f87171",
    redBg: "rgba(239,68,68,0.1)",
    redBorder: "rgba(239,68,68,0.25)",
    amber: "#fbbf24",
    amberBg: "rgba(245,158,11,0.1)",
    amberBorder: "rgba(245,158,11,0.25)",
    blue: "#60a5fa",
    blueBg: "rgba(59,130,246,0.1)",
    blueBorder: "rgba(59,130,246,0.25)",
    cardShadow: "0 1px 3px rgba(0,0,0,0.3)",
    plotGrid: "rgba(255,255,255,0.05)",
    plotZero: "rgba(255,255,255,0.08)",
    plotFont: "rgba(255,255,255,0.5)",
  },
};

// ─── Mock Data ──────────────────────────────────────────────────────────────
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
const REPRO = genPts(100, 12, 42);
const ESM = genPts(100, 8, 99);
const SPEED = {
  single: { unirep: 1.2, esm2: 0.05 },
  multi100: { unirep: 45.3, esm2: 1.8 },
};
const CLR = { r1: "#94a3b8", r2: "#f59e0b", r3: "#3b82f6" };

// ─── Plotly ─────────────────────────────────────────────────────────────────
function Plot({ id, data, layout, style }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current)
      Plotly.newPlot(ref.current, data, layout, {
        responsive: true,
        displayModeBar: false,
      });
    return () => {
      if (ref.current) Plotly.purge(ref.current);
    };
  }, [data, layout]);
  return <div ref={ref} id={id} style={style} />;
}
function mkTrace(pts, name, color, sz = 8) {
  return {
    x: pts.map((p) => p.x),
    y: pts.map((p) => p.y),
    text: pts.map((p) => p.label),
    mode: "markers",
    type: "scatter",
    name,
    marker: {
      color,
      size: sz,
      opacity: 0.8,
      line: { width: 0.5, color: "rgba(255,255,255,0.6)" },
    },
    hovertemplate: `<b>${name}</b><br>%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>`,
  };
}

// ─── Navbar ─────────────────────────────────────────────────────────────────
function Navbar({ isDark, setIsDark }) {
  const t = useTheme();
  const secs = [
    { id: "hero", l: "Overview" },
    { id: "upload", l: "Run" },
    { id: "repro", l: "Reproducibility" },
    { id: "speed", l: "Speed" },
    { id: "quality", l: "Quality" },
    { id: "compare", l: "Comparison" },
    { id: "pipeline", l: "Pipeline" },
  ];
  return (
    <nav
      style={{
        position: "sticky",
        top: 0,
        zIndex: 100,
        width: "100%",
        background: isDark ? "rgba(10,14,26,0.85)" : "rgba(248,250,252,0.85)",
        backdropFilter: "blur(12px)",
        borderBottom: `1px solid ${t.border}`,
        padding: "0 5vw",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}
    >
      <div
        style={{ display: "flex", gap: 2, overflowX: "auto", padding: "8px 0" }}
      >
        {secs.map((s) => (
          <button
            key={s.id}
            onClick={() =>
              document
                .getElementById(s.id)
                ?.scrollIntoView({ behavior: "smooth" })
            }
            style={{
              padding: "7px 14px",
              fontSize: 14,
              fontFamily: "'JetBrains Mono'",
              color: t.textSecondary,
              background: "transparent",
              border: "none",
              cursor: "pointer",
              borderRadius: 6,
              whiteSpace: "nowrap",
              transition: "all 0.15s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = t.accentBg;
              e.currentTarget.style.color = t.accent;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "transparent";
              e.currentTarget.style.color = t.textSecondary;
            }}
          >
            {s.l}
          </button>
        ))}
      </div>
      <button
        onClick={() => setIsDark(!isDark)}
        style={{
          padding: "7px 12px",
          fontSize: 18,
          background: t.surface,
          border: `1px solid ${t.border}`,
          borderRadius: 8,
          cursor: "pointer",
          color: t.text,
          flexShrink: 0,
        }}
      >
        {isDark ? "☀️" : "🌙"}
      </button>
    </nav>
  );
}

// ─── UI Components ──────────────────────────────────────────────────────────
function MetricCard({ label, value, sub, accent }) {
  const t = useTheme();
  return (
    <div
      style={{
        background: accent
          ? `linear-gradient(135deg, ${t.accent}, #0369a1)`
          : t.surface,
        border: accent ? "none" : `1px solid ${t.border}`,
        borderRadius: 14,
        padding: "24px 20px",
        textAlign: "center",
        flex: 1,
        minWidth: 160,
        boxShadow: accent ? `0 4px 16px ${t.accent}30` : t.cardShadow,
      }}
    >
      <div
        style={{
          fontSize: 12,
          color: accent ? "rgba(255,255,255,0.7)" : t.textMuted,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          marginBottom: 8,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 36,
          fontWeight: 700,
          color: accent ? "#fff" : t.text,
          fontFamily: "'Space Mono'",
          lineHeight: 1.1,
        }}
      >
        {value}
      </div>
      {sub && (
        <div
          style={{
            fontSize: 13,
            color: accent ? "rgba(255,255,255,0.6)" : t.textMuted,
            marginTop: 6,
          }}
        >
          {sub}
        </div>
      )}
    </div>
  );
}
function SectionTitle({ number, title, subtitle }) {
  const t = useTheme();
  return (
    <div style={{ marginBottom: 36 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 14,
          marginBottom: 10,
        }}
      >
        <span
          style={{
            fontSize: 14,
            color: t.accent,
            background: t.accentBg,
            border: `1px solid ${t.accentBorder}`,
            borderRadius: 8,
            padding: "3px 12px",
            fontFamily: "'Space Mono'",
          }}
        >
          {number}
        </span>
        <div
          style={{
            flex: 1,
            height: 1,
            background: `linear-gradient(90deg, ${t.accentBorder}, transparent)`,
          }}
        />
      </div>
      <h2
        style={{
          fontSize: 32,
          fontWeight: 300,
          color: t.text,
          margin: 0,
          fontFamily: "'Instrument Serif', serif",
        }}
      >
        {title}
      </h2>
      {subtitle && (
        <p
          style={{
            fontSize: 15,
            color: t.textSecondary,
            margin: "6px 0 0",
            lineHeight: 1.6,
          }}
        >
          {subtitle}
        </p>
      )}
    </div>
  );
}
function Tag({ children, color }) {
  const t = useTheme();
  const c = color || t.accent;
  return (
    <span
      style={{
        fontSize: 13,
        color: c,
        background: `${c}15`,
        border: `1px solid ${c}30`,
        borderRadius: 6,
        padding: "2px 10px",
        fontFamily: "'JetBrains Mono'",
      }}
    >
      {children}
    </span>
  );
}
function Badge({ label, good }) {
  const t = useTheme();
  const c = good ? t.green : t.red;
  return (
    <span
      style={{
        fontSize: 12,
        color: c,
        background: good ? t.greenBg : t.redBg,
        border: `1px solid ${good ? t.greenBorder : t.redBorder}`,
        borderRadius: 4,
        padding: "2px 8px",
        textTransform: "uppercase",
      }}
    >
      {label}
    </span>
  );
}

// ─── FASTA Parser ───────────────────────────────────────────────────────────
function parseFasta(text) {
  const lines = text.split("\n");
  const seqs = [];
  let cur = null;
  for (const l of lines) {
    const s = l.trim();
    if (s.startsWith(">")) {
      if (cur) seqs.push(cur);
      cur = { header: s.slice(1).trim(), length: 0 };
    } else if (cur && s) cur.length += s.length;
  }
  if (cur) seqs.push(cur);
  return seqs;
}

// ─── Upload Section ─────────────────────────────────────────────────────────
function UploadSection({ model, setModel, onRunComplete }) {
  const t = useTheme();
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState({});
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [runMessage, setRunMessage] = useState("");
  const [runLogs, setRunLogs] = useState([]);
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);
  const busy = status === "running_unirep" || status === "running_esm2";
  const doU = model === "both" || model === "unirep";
  const doE = model === "both" || model === "esm2";
  const totalSeq = Object.values(previews).reduce((s, a) => s + a.length, 0);

  const sc = {
    idle: {
      label: "Waiting for upload",
      color: t.textMuted,
      bg: t.bg,
      icon: "📁",
    },
    uploaded: {
      label: `${files.length} file${
        files.length !== 1 ? "s" : ""
      } · ${totalSeq} sequences — Ready`,
      color: t.accent,
      bg: t.accentBg,
      icon: "✅",
    },
    running_unirep: {
      label: "Running UniRep…",
      color: t.amber,
      bg: t.amberBg,
      icon: "⏳",
    },
    running_esm2: {
      label: "Running ESM-2…",
      color: t.blue,
      bg: t.blueBg,
      icon: "⏳",
    },
    complete: {
      label: "Complete — download results",
      color: t.green,
      bg: t.greenBg,
      icon: "🎉",
    },
    error: {
      label: errorMsg || "Error",
      color: t.red,
      bg: t.redBg,
      icon: "❌",
    },
  }[status];

  const exts = [".fasta", ".fa", ".txt"];
  const ok = (f) => exts.some((e) => f.name.toLowerCase().endsWith(e));
  const add = async (fl) => {
    const v = Array.from(fl).filter(ok);
    const bad = Array.from(fl).filter((f) => !ok(f));
    if (bad.length) setErrorMsg(`Skipped ${bad.length} unsupported file(s)`);
    if (v.length) {
      setFiles((p) => {
        const n = new Set(p.map((f) => f.name));
        return [...p, ...v.filter((f) => !n.has(f.name))];
      });
      for (const f of v) {
        const txt = await f.text();
        setPreviews((p) => ({ ...p, [f.name]: parseFasta(txt) }));
      }
      setStatus("uploaded");
      setResults(null);
      setProgress(0);
      onRunComplete?.(null);
    }
  };
  const rm = (i) => {
    const n = files[i].name;
    setFiles((p) => p.filter((_, j) => j !== i));
    setPreviews((p) => {
      const c = { ...p };
      delete c[n];
      return c;
    });
    if (files.length <= 1) {
      setStatus("idle");
      setResults(null);
    }
  };

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const run = async () => {
    if (!files.length) return;
    try {
      setErrorMsg("");
      setRunMessage("Job queued...");
      setRunLogs(["[INIT] Job queued..."]);
      setStatus(doU ? "running_unirep" : "running_esm2");
      setProgress(1);

      const payloadFiles = await Promise.all(
        files.map(async (f) => ({ name: f.name, content: await f.text() }))
      );

      const startResp = await fetch("http://localhost:8000/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ files: payloadFiles, model }),
      });

      if (!startResp.ok) {
        throw new Error(`Backend error: ${startResp.status}`);
      }

      const { jobId } = await startResp.json();
      if (!jobId) throw new Error("Missing job id");

      let finalResult = null;
      while (!finalResult) {
        await sleep(300);
        const pollResp = await fetch(`http://localhost:8000/api/run/${jobId}`);
        if (!pollResp.ok) {
          throw new Error(`Poll failed: ${pollResp.status}`);
        }

        const poll = await pollResp.json();
        setProgress(Math.max(1, Math.min(100, poll.progress ?? 1)));
        if (poll.message) {
          setRunMessage(poll.message);
        }
        if (Array.isArray(poll.logs)) {
          setRunLogs(poll.logs.slice(-300));
        }
        if (poll.phase === "running_unirep" || poll.phase === "running_esm2") {
          setStatus(poll.phase);
        }

        if (poll.status === "complete") {
          finalResult = poll.result;
          break;
        }

        if (poll.status === "error") {
          throw new Error(poll.error || poll.message || "Run failed");
        }
      }

      const mapped = (finalResult.files || []).map((f) => ({
        name: f.fileName,
        u: f.outputs?.unirep?.name || null,
        uUrl: f.outputs?.unirep?.downloadUrl
          ? `http://localhost:8000${f.outputs.unirep.downloadUrl}`
          : null,
        e: f.outputs?.esm2?.name || null,
        eUrl: f.outputs?.esm2?.downloadUrl
          ? `http://localhost:8000${f.outputs.esm2.downloadUrl}`
          : null,
      }));

      setProgress(100);
      setStatus("complete");
      setRunMessage("Completed");
      setRunLogs((prev) => [...prev, "[complete] Completed"]);
      setResults(mapped);
      onRunComplete?.({
        model: finalResult.model || model,
        files: mapped,
        reproducibility: finalResult.reproducibility || null,
      });
    } catch (e) {
      const msg = e.message || "Run failed";
      setErrorMsg(msg);
      setRunMessage(msg);
      setRunLogs((prev) => [...prev, `[error] ${msg}`]);
      setStatus("error");
    }
  };

  const dl = (name, url) => {
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };
  const reset = () => {
    setFiles([]);
    setPreviews({});
    setStatus("idle");
    setProgress(0);
    setResults(null);
    setErrorMsg("");
    setRunMessage("");
    setRunLogs([]);
    onRunComplete?.(null);
  };

  const steps = [
    {
      s: `${files.length} file${
        files.length !== 1 ? "s" : ""
      } (${totalSeq} seq)`,
      done: status !== "idle",
    },
    ...(doU
      ? [
          {
            s: "UniRep extraction",
            done: status === "running_esm2" || status === "complete",
            active: status === "running_unirep",
          },
        ]
      : []),
    ...(doE
      ? [
          {
            s: "ESM-2 extraction",
            done: status === "complete",
            active: status === "running_esm2",
          },
        ]
      : []),
    { s: "Results ready", done: status === "complete" },
  ];

  return (
    <section
      id="upload"
      style={{
        position: "relative",
        zIndex: 1,
        width: "100%",
        padding: "60px 5vw",
      }}
    >
      <SectionTitle
        number="00"
        title="Run Embedding Extraction"
        subtitle="Upload FASTA files to extract protein embeddings"
      />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 24,
          alignItems: "start",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Dropzone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              add(e.dataTransfer.files);
            }}
            onClick={() => !busy && inputRef.current?.click()}
            style={{
              background: drag ? t.accentBg : t.surface,
              border: `2px dashed ${
                drag ? t.accent : files.length ? t.greenBorder : t.border
              }`,
              borderRadius: 14,
              padding: 28,
              textAlign: "center",
              cursor: busy ? "default" : "pointer",
              boxShadow: t.cardShadow,
              transition: "all 0.2s",
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".fasta,.fa,.txt"
              multiple
              onChange={(e) => {
                add(e.target.files);
                e.target.value = "";
              }}
              style={{ display: "none" }}
            />
            <div style={{ fontSize: 32, marginBottom: 6 }}>
              {files.length ? "📄" : "📂"}
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, color: t.text }}>
              {files.length
                ? `${files.length} file${
                    files.length !== 1 ? "s" : ""
                  } · ${totalSeq} sequences`
                : "Drop FASTA files here"}
            </div>
            <div style={{ fontSize: 13, color: t.textMuted, marginTop: 4 }}>
              {files.length
                ? "click to add more"
                : ".fasta .fa .txt — multiple OK"}
            </div>
          </div>

          {/* File list with FASTA preview */}
          {files.length > 0 && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
                maxHeight: 200,
                overflowY: "auto",
                paddingRight: 4,
              }}
            >
              {files.map((f, i) => {
                const seqs = previews[f.name] || [];
                return (
                  <div
                    key={f.name}
                    style={{
                      background: t.surface,
                      border: `1px solid ${t.border}`,
                      borderRadius: 10,
                      padding: "10px 14px",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 8,
                        }}
                      >
                        <span style={{ fontSize: 14 }}>📄</span>
                        <span
                          style={{
                            fontSize: 14,
                            color: t.text,
                            fontWeight: 500,
                          }}
                        >
                          {f.name}
                        </span>
                        <span style={{ fontSize: 12, color: t.textMuted }}>
                          {(f.size / 1024).toFixed(1)}KB
                        </span>
                        {seqs.length > 0 && (
                          <span
                            style={{
                              fontSize: 11,
                              color: t.accent,
                              background: t.accentBg,
                              borderRadius: 4,
                              padding: "1px 6px",
                            }}
                          >
                            {seqs.length} seq
                          </span>
                        )}
                      </div>
                      {!busy && status !== "complete" && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            rm(i);
                          }}
                          style={{
                            background: "none",
                            border: "none",
                            color: t.red,
                            cursor: "pointer",
                            fontSize: 15,
                          }}
                        >
                          ✕
                        </button>
                      )}
                    </div>
                    {seqs.length > 0 && (
                      <div style={{ marginTop: 6, paddingLeft: 22 }}>
                        {seqs.slice(0, 3).map((s, j) => (
                          <div
                            key={j}
                            style={{
                              fontSize: 11,
                              color: t.textMuted,
                              lineHeight: 1.7,
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                            }}
                          >
                            <span style={{ color: t.textSecondary }}>
                              {s.header.length > 45
                                ? s.header.slice(0, 45) + "…"
                                : s.header}
                            </span>
                            <span style={{ color: t.accent, marginLeft: 6 }}>
                              {s.length} aa
                            </span>
                          </div>
                        ))}
                        {seqs.length > 3 && (
                          <div
                            style={{
                              fontSize: 11,
                              color: t.textMuted,
                              fontStyle: "italic",
                            }}
                          >
                            +{seqs.length - 3} more
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Model selector */}
          <div>
            <div
              style={{
                fontSize: 12,
                color: t.textMuted,
                marginBottom: 6,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
              }}
            >
              Model
            </div>
            <div
              style={{
                display: "flex",
                background: t.surface,
                borderRadius: 8,
                overflow: "hidden",
                border: `1px solid ${t.border}`,
                width: "fit-content",
              }}
            >
              {[
                ["both", "Both"],
                ["unirep", "UniRep"],
                ["esm2", "ESM-2"],
              ].map(([k, l]) => (
                <button
                  key={k}
                  onClick={() => setModel(k)}
                  disabled={busy}
                  style={{
                    padding: "8px 16px",
                    fontSize: 14,
                    color: model === k ? t.accent : t.textMuted,
                    background: model === k ? t.accentBg : "transparent",
                    border: "none",
                    cursor: busy ? "not-allowed" : "pointer",
                    fontWeight: model === k ? 600 : 400,
                    fontFamily: "'JetBrains Mono'",
                  }}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            <button
              onClick={run}
              disabled={!files.length || busy}
              style={{
                padding: "11px 26px",
                fontSize: 15,
                fontWeight: 600,
                color: "#fff",
                background:
                  !files.length || busy
                    ? t.border
                    : "linear-gradient(135deg, #0284c7, #0369a1)",
                border: "none",
                borderRadius: 10,
                cursor: !files.length || busy ? "not-allowed" : "pointer",
                boxShadow:
                  files.length && !busy
                    ? "0 4px 12px rgba(2,132,199,0.3)"
                    : "none",
                fontFamily: "'JetBrains Mono'",
              }}
            >
              {busy ? "Running…" : "Run"}
            </button>
            {(status === "complete" || status === "error") && (
              <button
                onClick={reset}
                style={{
                  padding: "11px 18px",
                  fontSize: 15,
                  color: t.textSecondary,
                  background: t.surface,
                  border: `1px solid ${t.border}`,
                  borderRadius: 10,
                  cursor: "pointer",
                  fontFamily: "'JetBrains Mono'",
                }}
              >
                Reset
              </button>
            )}
          </div>
          {errorMsg && status !== "error" && (
            <div style={{ fontSize: 13, color: t.amber }}>⚠ {errorMsg}</div>
          )}
        </div>

        {/* Status */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div
            style={{
              background: sc.bg,
              border: `1px solid ${sc.color}25`,
              borderRadius: 14,
              padding: "18px 22px",
              boxShadow: t.cardShadow,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                marginBottom: 12,
              }}
            >
              <span style={{ fontSize: 20 }}>{sc.icon}</span>
              <span style={{ fontSize: 15, fontWeight: 600, color: sc.color }}>
                {sc.label}
              </span>
            </div>
            {(busy || status === "complete") && (
              <div style={{ marginBottom: 12 }}>
                <div
                  style={{
                    height: 6,
                    background: t.border,
                    borderRadius: 3,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${progress}%`,
                      borderRadius: 3,
                      transition: "width 0.4s",
                      background:
                        status === "complete"
                          ? t.green
                          : `linear-gradient(90deg, ${t.accent}, #38bdf8)`,
                    }}
                  />
                </div>
                <div
                  style={{
                    fontSize: 12,
                    color: t.textMuted,
                    marginTop: 4,
                    textAlign: "right",
                  }}
                >
                  {progress}%
                </div>
                {runMessage && (
                  <div
                    style={{
                      marginTop: 8,
                      fontSize: 12,
                      color: t.textSecondary,
                      fontFamily: "'JetBrains Mono'",
                      wordBreak: "break-word",
                    }}
                  >
                    {runMessage}
                  </div>
                )}

                {runLogs.length > 0 && (
                  <div
                    style={{
                      marginTop: 10,
                      padding: "8px 10px",
                      border: `1px solid ${t.border}`,
                      borderRadius: 8,
                      background: t.surface,
                      maxHeight: 130,
                      overflowY: "auto",
                      fontFamily: "'JetBrains Mono'",
                      fontSize: 11,
                      color: t.textSecondary,
                      lineHeight: 1.5,
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    {runLogs.map((line, idx) => (
                      <div key={idx}>{line}</div>
                    ))}
                  </div>
                )}
              </div>
            )}
            {status !== "idle" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {steps.map(({ s, done, active }, i) => (
                  <div
                    key={i}
                    style={{ display: "flex", alignItems: "center", gap: 10 }}
                  >
                    <div
                      style={{
                        width: 18,
                        height: 18,
                        borderRadius: "50%",
                        flexShrink: 0,
                        background: done
                          ? t.green
                          : active
                          ? t.accent
                          : t.border,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 10,
                        color: "#fff",
                        fontWeight: 700,
                        animation: active ? "pulse 1.5s infinite" : "none",
                      }}
                    >
                      {done ? "✓" : active ? "●" : ""}
                    </div>
                    <span
                      style={{
                        fontSize: 14,
                        color: done ? t.green : active ? t.accent : t.textMuted,
                        fontWeight: active ? 600 : 400,
                      }}
                    >
                      {s}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

        </div>
      </div>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}`}</style>
    </section>
  );
}

// ─── Main ───────────────────────────────────────────────────────────────────
export default function App() {
  const [isDark, setIsDark] = useState(false);
  const [speedMode, setSpeedMode] = useState("single");
  const [selectedModel, setSelectedModel] = useState("both");
  const [runSummary, setRunSummary] = useState(null);
  const t = isDark ? themes.dark : themes.light;
  const reproPts = runSummary?.reproducibility?.unirep || REPRO;
  const esmPts = runSummary?.reproducibility?.esm2 || ESM;
  const reproT = [
    mkTrace(reproPts, "Run 1", CLR.r1),
    mkTrace(reproPts, "Run 2", CLR.r2, 6),
    mkTrace(reproPts, "Run 3", CLR.r3, 4),
  ];
  const esmT = [
    mkTrace(esmPts, "Run 1", CLR.r1),
    mkTrace(esmPts, "Run 2", CLR.r2, 6),
    mkTrace(esmPts, "Run 3", CLR.r3, 4),
  ];
  const pL = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "JetBrains Mono", color: t.plotFont, size: 12 },
    margin: { l: 48, r: 16, t: 36, b: 48 },
    xaxis: {
      gridcolor: t.plotGrid,
      zerolinecolor: t.plotZero,
      title: { text: "t-SNE 1", font: { size: 12, color: t.textMuted } },
    },
    yaxis: {
      gridcolor: t.plotGrid,
      zerolinecolor: t.plotZero,
      title: { text: "t-SNE 2", font: { size: 12, color: t.textMuted } },
    },
    showlegend: true,
    legend: {
      font: { size: 11, color: t.textSecondary },
      bgcolor: "rgba(0,0,0,0)",
      x: 1,
      xanchor: "right",
      y: 1,
    },
  };
  const pT = (text) => ({
    text,
    font: { size: 13, color: t.textMuted, family: "JetBrains Mono" },
  });
  const sm = speedMode === "single" ? "single" : "multi100";
  const sT = [
    {
      x: ["UniRep", "ESM-2"],
      y: [SPEED[sm].unirep, SPEED[sm].esm2],
      type: "bar",
      marker: { color: ["#f59e0b", "#3b82f6"], opacity: 0.85 },
      hovertemplate: "<b>%{x}</b><br>%{y:.2f}s<extra></extra>",
    },
  ];
  const sL = {
    ...pL,
    title: {
      text: speedMode === "single" ? "Single Protein" : "100 Proteins",
      font: { size: 13, color: t.textSecondary, family: "JetBrains Mono" },
    },
    yaxis: {
      ...pL.yaxis,
      title: { text: "Time (s)", font: { size: 12, color: t.textMuted } },
    },
    xaxis: { ...pL.xaxis, title: null },
    showlegend: false,
    bargap: 0.5,
  };
  const ratio = (SPEED[sm].unirep / SPEED[sm].esm2).toFixed(0);
  const S = {
    position: "relative",
    zIndex: 1,
    width: "100%",
    padding: "60px 5vw",
  };

  return (
    <ThemeContext.Provider value={t}>
      <div
        style={{
          minHeight: "100vh",
          width: "100%",
          background: t.bg,
          color: t.text,
          fontFamily: "'JetBrains Mono', monospace",
          overflowX: "hidden",
          transition: "background 0.3s, color 0.3s",
        }}
      >
        <link
          href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap"
          rel="stylesheet"
        />
        <Navbar isDark={isDark} setIsDark={setIsDark} />

        {/* Hero */}
        <header id="hero" style={{ ...S, paddingTop: 48, paddingBottom: 36 }}>
          <div
            style={{
              display: "flex",
              gap: 8,
              marginBottom: 18,
              flexWrap: "wrap",
            }}
          >
            <Tag>NEU Capstone</Tag>
            <Tag color={t.amber}>Protein Engineering</Tag>
            <Tag color={t.green}>Machine Learning</Tag>
          </div>
          <h1
            style={{
              fontSize: "clamp(30px, 4.5vw, 52px)",
              fontWeight: 300,
              fontFamily: "'Instrument Serif', serif",
              lineHeight: 1.1,
              margin: 0,
            }}
          >
            Protein Language Model
            <br />
            <span style={{ color: t.accent }}>Evaluation & Comparison</span>
          </h1>
          <p
            style={{
              fontSize: 15,
              color: t.textSecondary,
              maxWidth: 660,
              lineHeight: 1.7,
              marginTop: 18,
            }}
          >
            Comparing UniRep (mLSTM) and ESM-2 (Transformer) for protein feature
            extraction across reproducibility, speed, and embedding quality.
          </p>
          <div
            style={{
              display: "flex",
              gap: 12,
              marginTop: 28,
              flexWrap: "wrap",
            }}
          >
            <MetricCard label="Sequences" value="100" sub="from 5Y0M" />
            <MetricCard label="Models" value="2" sub="UniRep · ESM-2" />
            <MetricCard label="Replicas" value="6" sub="per model" />
            <MetricCard label="Status" value="✓" sub="reproducible" accent />
          </div>
        </header>

        <UploadSection
          model={selectedModel}
          setModel={setSelectedModel}
          onRunComplete={setRunSummary}
        />

        {/* Reproducibility */}
        <section id="repro" style={S}>
          <SectionTitle
            number="01"
            title="Reproducibility"
            subtitle="PCA → t-SNE confirms bitwise-identical embeddings across runs"
          />

          {runSummary && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  runSummary.model === "both" ? "1fr 1fr" : "1fr",
                gap: 20,
              }}
            >
              {(runSummary.model === "both" || runSummary.model === "unirep") && (
                <div
                  style={{
                    background: t.surface,
                    border: `1px solid ${t.amberBorder}`,
                    borderRadius: 14,
                    padding: 18,
                    boxShadow: t.cardShadow,
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      marginBottom: 6,
                    }}
                  >
                    <span
                      style={{ fontSize: 16, fontWeight: 600, color: t.amber }}
                    >
                      UniRep (mLSTM)
                    </span>
                    <Badge label="Deterministic" good />
                  </div>
                  <div
                    style={{ fontSize: 13, color: t.textMuted, marginBottom: 4 }}
                  >
                    TensorFlow 2.18
                  </div>
                  <div
                    style={{
                      width: "100%",
                      height: 370,
                      borderRadius: 8,
                      overflow: "hidden",
                      border: `1px solid ${t.border}`,
                      background: t.bg,
                    }}
                  >
                    <img
                      src={`http://localhost:8000/api/plot/unirep/latest?ts=${Date.now()}`}
                      alt="UniRep t-SNE latest"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        display: "block",
                      }}
                    />
                  </div>
                  <div
                    onClick={() =>
                      window.open("http://localhost:8000/api/download/unirep/first", "_blank")
                    }
                    style={{
                      background: t.surface,
                      border: `1px solid ${t.amberBorder}`,
                      borderRadius: 8,
                      padding: "8px 12px",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      cursor: "pointer",
                      transition: "border-color 0.15s",
                      marginTop: 10,
                    }}
                    onMouseEnter={(e) =>
                      (e.currentTarget.style.borderColor = t.amber)
                    }
                    onMouseLeave={(e) =>
                      (e.currentTarget.style.borderColor = t.amberBorder)
                    }
                  >
                    <div>
                      <div
                        style={{
                          fontSize: 13,
                          fontWeight: 600,
                          color: t.text,
                        }}
                      >
                        first_unirep_run.h5
                      </div>
                      <div style={{ fontSize: 11, color: t.textMuted }}>
                        UniRep · 1900d
                      </div>
                    </div>
                    <span
                      style={{ fontSize: 13, color: t.amber, fontWeight: 600 }}
                    >
                      ↓
                    </span>
                  </div>
                </div>
              )}
              {(runSummary.model === "both" || runSummary.model === "esm2") && (
                <div
                  style={{
                    background: t.surface,
                    border: `1px solid ${t.blueBorder}`,
                    borderRadius: 14,
                    padding: 18,
                    boxShadow: t.cardShadow,
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      marginBottom: 6,
                    }}
                  >
                    <span style={{ fontSize: 16, fontWeight: 600, color: t.blue }}>
                      ESM-2 (Transformer)
                    </span>
                    <Badge label="Deterministic" good />
                  </div>
                  <div
                    style={{ fontSize: 13, color: t.textMuted, marginBottom: 4 }}
                  >
                    PyTorch — natively deterministic
                  </div>
                  <div
                    style={{
                      width: "100%",
                      height: 370,
                      borderRadius: 8,
                      overflow: "hidden",
                      border: `1px solid ${t.border}`,
                      background: t.bg,
                    }}
                  >
                    <img
                      src={`http://localhost:8000/api/plot/esm/latest?ts=${Date.now()}`}
                      alt="ESM t-SNE latest"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        display: "block",
                      }}
                    />
                  </div>
                  <div
                    onClick={() =>
                      window.open(
                        "http://localhost:8000/api/download/esm2/esm_run1.h5",
                        "_blank"
                      )
                    }
                    style={{
                      background: t.surface,
                      border: `1px solid ${t.blueBorder}`,
                      borderRadius: 8,
                      padding: "8px 12px",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      cursor: "pointer",
                      transition: "border-color 0.15s",
                      marginTop: 10,
                    }}
                    onMouseEnter={(e) =>
                      (e.currentTarget.style.borderColor = t.blue)
                    }
                    onMouseLeave={(e) =>
                      (e.currentTarget.style.borderColor = t.blueBorder)
                    }
                  >
                    <div>
                      <div
                        style={{
                          fontSize: 13,
                          fontWeight: 600,
                          color: t.text,
                        }}
                      >
                        esm_run1.h5
                      </div>
                      <div style={{ fontSize: 11, color: t.textMuted }}>
                        ESM-2 · 1280d
                      </div>
                    </div>
                    <span
                      style={{ fontSize: 13, color: t.blue, fontWeight: 600 }}
                    >
                      ↓
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}
          <p
            style={{
              fontSize: 13,
              color: t.textMuted,
              marginTop: 12,
              lineHeight: 1.7,
            }}
          >
            Pipeline:{" "}
            <span style={{ color: t.accent }}>
              StandardScaler → PCA (95%) → t-SNE (perplexity=50, iter=10k,
              seed=42)
            </span>
          </p>
        </section>

        {/* Speed */}
        <section id="speed" style={S}>
          <SectionTitle
            number="02"
            title="Inference Speed"
            subtitle="V100 GPU wall-clock comparison"
          />
          <div
            style={{
              display: "flex",
              marginBottom: 20,
              background: t.surface,
              borderRadius: 8,
              overflow: "hidden",
              border: `1px solid ${t.border}`,
              width: "fit-content",
            }}
          >
            {["single", "multi"].map((m) => (
              <button
                key={m}
                onClick={() => setSpeedMode(m)}
                style={{
                  padding: "8px 18px",
                  fontSize: 14,
                  color: speedMode === m ? t.accent : t.textMuted,
                  background: speedMode === m ? t.accentBg : "transparent",
                  border: "none",
                  cursor: "pointer",
                  fontWeight: speedMode === m ? 600 : 400,
                  fontFamily: "'JetBrains Mono'",
                }}
              >
                {m === "single" ? "1 Protein" : "100 Proteins"}
              </button>
            ))}
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 20,
              alignItems: "start",
            }}
          >
            <div
              style={{
                background: t.surface,
                border: `1px solid ${t.border}`,
                borderRadius: 14,
                padding: 18,
                boxShadow: t.cardShadow,
              }}
            >
              <Plot
                id="spd"
                data={sT}
                layout={sL}
                style={{ width: "100%", height: 320 }}
              />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <MetricCard
                label="Speed Advantage"
                value={`${ratio}×`}
                sub="ESM-2 faster"
                accent
              />
              <MetricCard
                label="UniRep"
                value={`${SPEED[sm].unirep}s`}
                sub="mLSTM sequential"
              />
              <MetricCard
                label="ESM-2"
                value={`${SPEED[sm].esm2}s`}
                sub="Transformer parallel"
              />
            </div>
          </div>
        </section>

        {/* Quality */}
        <section id="quality" style={S}>
          <SectionTitle
            number="03"
            title="Embedding Quality"
            subtitle="Representation characteristics"
          />
          <div
            style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}
          >
            {[
              {
                n: "UniRep",
                c: t.amber,
                b: t.amberBorder,
                tag: "mLSTM",
                dim: "1,900",
                arch: "Multiplicative LSTM",
                fw: "TensorFlow 2.18",
                match: "100% (after fix)",
                pca: "Required (1900→~50 PCs)",
              },
              {
                n: "ESM-2",
                c: t.blue,
                b: t.blueBorder,
                tag: "Transformer",
                dim: "1,280",
                arch: "Transformer (33L, 650M)",
                fw: "PyTorch (ESM)",
                match: "100% (native)",
                pca: "Recommended (1280→~40 PCs)",
              },
            ].map((m) => (
              <div
                key={m.n}
                style={{
                  background: t.surface,
                  border: `1px solid ${m.b}`,
                  borderRadius: 14,
                  padding: 24,
                  boxShadow: t.cardShadow,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginBottom: 18,
                  }}
                >
                  <div
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: "50%",
                      background: m.c,
                    }}
                  />
                  <span
                    style={{ fontSize: 17, fontWeight: 600, color: t.text }}
                  >
                    {m.n}
                  </span>
                  <Tag color={m.c}>{m.tag}</Tag>
                </div>
                {[
                  { l: "Embedding Dim", v: m.dim, big: true },
                  { l: "Architecture", v: m.arch },
                  { l: "Framework", v: m.fw },
                  { l: "Bitwise Match", v: m.match, g: true },
                  { l: "PCA Pre-reduction", v: m.pca },
                ].map((r) => (
                  <div key={r.l} style={{ marginBottom: 12 }}>
                    <div
                      style={{
                        fontSize: 11,
                        color: t.textMuted,
                        marginBottom: 2,
                        letterSpacing: "0.06em",
                        textTransform: "uppercase",
                      }}
                    >
                      {r.l}
                    </div>
                    <div
                      style={{
                        fontSize: r.big ? 26 : 15,
                        fontWeight: r.big ? 700 : 400,
                        color: r.g ? t.green : t.text,
                        fontFamily: r.big ? "'Space Mono'" : "inherit",
                      }}
                    >
                      {r.v}
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </section>

        {/* Comparison */}
        <section id="compare" style={S}>
          <SectionTitle
            number="04"
            title="Single vs. Multi-Protein"
            subtitle="Scaling behavior"
          />
          <div
            style={{
              background: t.surface,
              border: `1px solid ${t.border}`,
              borderRadius: 14,
              overflow: "hidden",
              boxShadow: t.cardShadow,
            }}
          >
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: 14,
              }}
            >
              <thead>
                <tr
                  style={{
                    borderBottom: `1px solid ${t.border}`,
                    background: t.borderLight,
                  }}
                >
                  {[
                    "Metric",
                    "UniRep (1)",
                    "UniRep (100)",
                    "ESM-2 (1)",
                    "ESM-2 (100)",
                  ].map((h, i) => (
                    <th
                      key={h}
                      style={{
                        padding: "12px 14px",
                        textAlign: i ? "center" : "left",
                        color: t.textMuted,
                        fontWeight: 500,
                        fontSize: 12,
                        letterSpacing: "0.06em",
                        textTransform: "uppercase",
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  ["Reproducible", "✓ (TF 2.18)", "✓ (TF 2.18)", "✓", "✓"],
                  ["Inference", "1.2s", "45.3s", "0.05s", "1.8s"],
                  ["Dim", "1900", "1900", "1280", "1280"],
                  ["PCA", "Yes", "Yes", "Recommended", "Recommended"],
                  ["GPU Mem", "~2GB", "~4GB", "~3GB", "~5GB"],
                ].map((row, ri) => (
                  <tr
                    key={ri}
                    style={{
                      borderBottom:
                        ri < 4 ? `1px solid ${t.borderLight}` : "none",
                    }}
                  >
                    {row.map((c, ci) => (
                      <td
                        key={ci}
                        style={{
                          padding: "10px 14px",
                          textAlign: ci ? "center" : "left",
                          color:
                            ci === 0
                              ? t.textSecondary
                              : c.startsWith("✓")
                              ? t.green
                              : t.text,
                          fontWeight: ci === 0 ? 500 : 400,
                        }}
                      >
                        {c}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Pipeline */}
        <section id="pipeline" style={S}>
          <SectionTitle
            number="05"
            title="Pipeline"
            subtitle="Embedding → 2D visualization workflow"
          />
          <div
            style={{
              display: "flex",
              alignItems: "center",
              flexWrap: "wrap",
              justifyContent: "center",
            }}
          >
            {[
              { l: "FASTA", d: "Sequences", i: "📄" },
              { l: "Model", d: "UniRep/ESM-2", i: "🧠" },
              { l: "Embed", d: "1900d/1280d", i: "📊" },
              { l: "Scaler", d: "StandardScaler", i: "⚖️" },
              { l: "PCA", d: "95% var", i: "📉" },
              { l: "t-SNE", d: "2D plot", i: "🎯" },
            ].map((s, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center" }}>
                <div
                  style={{
                    background: t.surface,
                    border: `1px solid ${t.border}`,
                    borderRadius: 10,
                    padding: "14px 18px",
                    textAlign: "center",
                    minWidth: 100,
                    boxShadow: t.cardShadow,
                    transition: "all 0.15s",
                    cursor: "default",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = t.accentBorder;
                    e.currentTarget.style.transform = "translateY(-2px)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = t.border;
                    e.currentTarget.style.transform = "none";
                  }}
                >
                  <div style={{ fontSize: 22, marginBottom: 4 }}>{s.i}</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: t.text }}>
                    {s.l}
                  </div>
                  <div style={{ fontSize: 11, color: t.textMuted }}>{s.d}</div>
                </div>
                {i < 5 && (
                  <div
                    style={{
                      color: t.textMuted,
                      fontSize: 18,
                      padding: "0 6px",
                    }}
                  >
                    →
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        <footer
          style={{
            ...S,
            paddingTop: 24,
            paddingBottom: 36,
            borderTop: `1px solid ${t.border}`,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 10,
          }}
        >
          <div>
            <div style={{ fontSize: 14, color: t.textSecondary }}>
              Northeastern University · Capstone Project 5
            </div>
            <div style={{ fontSize: 12, color: t.textMuted, marginTop: 3 }}>
              Protein Language Model Evaluation
            </div>
          </div>
          <a
            href="https://github.com/EricXue1991/protein-dashboard"
            target="_blank"
            rel="noopener noreferrer"
            style={{ fontSize: 14, color: t.accent, textDecoration: "none" }}
          >
            GitHub →
          </a>
        </footer>
      </div>
    </ThemeContext.Provider>
  );
}
