"use client";

import { useEffect, useState, useRef, useCallback, startTransition } from "react";
import { Virtuoso } from "react-virtuoso";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import { useProjectStore } from "@/store/projectStore";
import { fetchPipelineStatus, streamExecute } from "@/lib/api";
import type { PipelineStatus } from "@/lib/types";

// ── Stage definitions (static) ────────────────────────────────────────────

const STAGES = [
  { id: 1, name: "detect-text", label: "Text Detection",  icon: "🔍", phase: "a", desc: "EasyOCR — detect speech bubbles"    },
  { id: 2, name: "ocr",         label: "OCR",             icon: "📝", phase: "a", desc: "MangaOCR — extract Japanese text"   },
  { id: 3, name: "translate",   label: "Translation",     icon: "🌐", phase: "a", desc: "Argos/Google — Japanese → English"  },
  { id: 4, name: "inpaint",     label: "Inpainting",      icon: "🎨", phase: "b", desc: "OpenCV — erase original text"       },
  { id: 5, name: "typeset",     label: "Typesetting",     icon: "✒️",  phase: "b", desc: "Pillow — render English text"       },
];

type RunState = "idle" | "running" | "done" | "error";

// ── Stage node visual ─────────────────────────────────────────────────────

function StageNode({
  stage, phaseAdone, phaseBdone, isActive, total,
}: {
  stage: typeof STAGES[0];
  phaseAdone: number;
  phaseBdone: number;
  isActive: boolean;
  total: number;
}) {
  const done = stage.phase === "a"
    ? phaseAdone >= total && total > 0
    : phaseBdone >= total && total > 0;

  const partial = stage.phase === "a"
    ? phaseAdone > 0 && phaseAdone < total
    : phaseBdone > 0 && phaseBdone < total;

  const statusClass = done ? "done" : isActive ? "active" : "idle";

  return (
    <div className="stage-node" style={{ minWidth: 90 }}>
      <div
        className={`stage-icon ${statusClass} ${isActive ? "pulse-glow" : ""}`}
        title={stage.desc}
      >
        <span style={{ fontSize: "1.2rem" }}>{stage.icon}</span>
        {done && (
          <span style={{
            position: "absolute",
            top: -6, right: -6,
            width: 16, height: 16,
            background: "var(--accent)",
            borderRadius: "50%",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "0.55rem",
            color: "#000",
            fontWeight: 900,
            border: "2px solid var(--bg-card)",
          }}>✓</span>
        )}
        {partial && !done && (
          <span style={{
            position: "absolute",
            top: -6, right: -6,
            width: 16, height: 16,
            background: "var(--amber)",
            borderRadius: "50%",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "0.55rem",
            color: "#000",
            fontWeight: 900,
            border: "2px solid var(--bg-card)",
          }}>~</span>
        )}
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: "0.75rem", fontWeight: 600, color: done ? "var(--accent)" : isActive ? "var(--accent)" : "var(--text-primary)" }}>
          {stage.label}
        </div>
        <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: 1 }}>
          Stage {stage.id}
        </div>
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function PipelinePage() {
  const { projectDir, inputSubdir, modelVersion, setModelVersion } = useProjectStore();
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [runState, setRunState] = useState<RunState>("idle");
  const [activeStage, setActiveStage] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [force, setForce] = useState(false);
  const [limitN, setLimitN] = useState<number | null>(null);
  const [selectedMode, setSelectedMode] = useState<"all" | "single" | "range">("all");
  const [selectedStage, setSelectedStage] = useState("detect-text");
  const [fromStage, setFromStage] = useState("detect-text");
  const [toStage, setToStage] = useState("typeset");
  const cleanupRef = useRef<(() => void) | null>(null);

  const loadStatus = useCallback(async () => {
    if (!projectDir) return;
    try {
      const s = await fetchPipelineStatus(projectDir);
      setStatus(s);
    } catch { /* backend not up yet */ }
  }, [projectDir]);

  useEffect(() => { loadStatus(); }, [loadStatus]);

  // Detect active stage from log lines
  useEffect(() => {
    const last = [...logs].reverse().find(l => l.includes("Stage"));
    if (last) {
      const m = last.match(/Stage \d+: (\S+)/i);
      if (m) setActiveStage(m[1].toLowerCase());
    }
  }, [logs]);

  const handleRun = () => {
    if (runState === "running") {
      cleanupRef.current?.();
      setRunState("idle");
      return;
    }
    setLogs([]);
    setRunState("running");

    const req: Parameters<typeof streamExecute>[0] = {
      project_dir: projectDir,
      input_subdir: inputSubdir,
      model_version: modelVersion,
      force,
      ...(limitN ? { limit: limitN } : {}),
      ...(selectedMode === "all" ? { all_stages: true } : {}),
      ...(selectedMode === "single" ? { stage: selectedStage } : {}),
      ...(selectedMode === "range" ? { from_stage: fromStage, to_stage: toStage } : {}),
    };

    cleanupRef.current = streamExecute(
      req,
      // Wrap log updates in startTransition so they are background-priority —
      // user interactions (clicks, scrolling) always process instantly
      (line) => startTransition(() => setLogs((prev) => [...prev.slice(-2000), line])),
      (success) => {
        setRunState(success ? "done" : "error");
        setActiveStage(null);
        loadStatus();
      },
    );
  };

  const btnLabel = runState === "running" ? "⏹ Stop" : "▶ Run Pipeline";
  const btnClass = runState === "running" ? "btn btn-secondary" : "btn btn-primary";

  return (
    <div>
      <Sidebar />
      <Header title="Pipeline Runner" />
      <main className="main-content fade-in">

        {/* Page header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.75rem" }}>
          <div>
            <h1 style={{ fontSize: "1.4rem", marginBottom: 2 }}>Pipeline Runner</h1>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.82rem" }}>
              Trigger individual stages or the full pipeline — streamed live.
            </p>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-secondary" onClick={loadStatus}>↻ Refresh</button>
            <button
              className={btnClass}
              onClick={handleRun}
              disabled={!status?.manifest_exists && runState !== "running"}
            >
              {runState === "running" ? <span className="spin">↻</span> : null}
              {btnLabel}
            </button>
          </div>
        </div>

        {/* Stage pipeline visualization */}
        <div className="card" style={{ marginBottom: "1.5rem" }}>
          <div className="section-header" style={{ marginBottom: "1.25rem" }}>
            <span className="section-title">Pipeline Stages</span>
            <div style={{ display: "flex", gap: 8 }}>
              <span className="badge badge-blue">Phase A: Read</span>
              <span className="badge badge-green">Phase B: Write</span>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "flex-start", overflowX: "auto", paddingBottom: 8 }}>
            {STAGES.map((stage, idx) => (
              <div key={stage.id} style={{ display: "flex", alignItems: "center" }}>
                {/* Phase label break */}
                {idx === 3 && (
                  <div style={{ display: "flex", alignItems: "center", marginRight: 8 }}>
                    <div style={{
                      height: 40, width: 2,
                      background: "var(--border)",
                      borderRadius: 1,
                      marginRight: 8,
                    }} />
                  </div>
                )}
                <StageNode
                  stage={stage}
                  phaseAdone={status?.phase_a_completed ?? 0}
                  phaseBdone={status?.phase_b_completed ?? 0}
                  isActive={runState === "running" && activeStage === stage.name}
                  total={status?.total ?? 0}
                />
                {idx < STAGES.length - 1 && (
                  <div className={`stage-connector ${
                    (idx < 2 && (status?.phase_a_completed ?? 0) > 0) ||
                    (idx >= 3 && (status?.phase_b_completed ?? 0) > 0)
                      ? "done" : ""
                  }`} />
                )}
              </div>
            ))}
          </div>

          {/* Phase progress bars */}
          {status?.manifest_exists && (
            <div style={{ marginTop: "1.25rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: "0.75rem" }}>
                  <span style={{ color: "var(--text-muted)" }}>Phase A</span>
                  <span style={{ color: "var(--blue)", fontWeight: 600 }}>{status.phase_a_pct}%</span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${status.phase_a_pct}%`, background: "linear-gradient(90deg, #3b82f6, #60a5fa)" }} />
                </div>
              </div>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: "0.75rem" }}>
                  <span style={{ color: "var(--text-muted)" }}>Phase B</span>
                  <span style={{ color: "var(--accent)", fontWeight: 600 }}>{status.phase_b_pct}%</span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${status.phase_b_pct}%` }} />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="two-col" style={{ marginBottom: "1.5rem" }}>
          {/* Run options */}
          <div className="card">
            <div className="section-title" style={{ marginBottom: "0.9rem" }}>Run Options</div>

            {/* Mode selector */}
            <div style={{ display: "flex", gap: 6, marginBottom: "1rem" }}>
              {(["all", "single", "range"] as const).map((m) => (
                <button
                  key={m}
                  className={`btn ${selectedMode === m ? "btn-primary" : "btn-secondary"}`}
                  style={{ fontSize: "0.75rem", flex: 1 }}
                  onClick={() => setSelectedMode(m)}
                >
                  {m === "all" ? "All Stages" : m === "single" ? "Single" : "Range"}
                </button>
              ))}
            </div>

            {selectedMode === "single" && (
              <div style={{ marginBottom: "0.75rem" }}>
                <label style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: 4 }}>Stage</label>
                <select
                  className="input-field"
                  value={selectedStage}
                  onChange={(e) => setSelectedStage(e.target.value)}
                  style={{ cursor: "pointer" }}
                >
                  {STAGES.map((s) => <option key={s.name} value={s.name}>{s.label}</option>)}
                </select>
              </div>
            )}

            {selectedMode === "range" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: "0.75rem" }}>
                <div>
                  <label style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: 4 }}>From</label>
                  <select className="input-field" value={fromStage} onChange={(e) => setFromStage(e.target.value)} style={{ cursor: "pointer" }}>
                    {STAGES.map((s) => <option key={s.name} value={s.name}>{s.label}</option>)}
                  </select>
                </div>
                <div>
                  <label style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: 4 }}>To</label>
                  <select className="input-field" value={toStage} onChange={(e) => setToStage(e.target.value)} style={{ cursor: "pointer" }}>
                    {STAGES.map((s) => <option key={s.name} value={s.name}>{s.label}</option>)}
                  </select>
                </div>
              </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <div>
                <label style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: 4 }}>Model Version</label>
                <select className="input-field" value={modelVersion} onChange={(e) => setModelVersion(e.target.value as "v1" | "v2")} style={{ cursor: "pointer" }}>
                  <option value="v2">v2 — Merged Bubble</option>
                  <option value="v1">v1 — Single Word</option>
                </select>
              </div>
              <div>
                <label style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: 4 }}>Page Limit</label>
                <input
                  className="input-field"
                  type="number"
                  min={1}
                  placeholder="All"
                  value={limitN ?? ""}
                  onChange={(e) => setLimitN(e.target.value ? parseInt(e.target.value) : null)}
                />
              </div>
            </div>

            <div style={{ marginTop: "0.75rem", display: "flex", alignItems: "center", gap: 8 }}>
              <input
                type="checkbox"
                id="force-cb"
                checked={force}
                onChange={(e) => setForce(e.target.checked)}
                style={{ accentColor: "var(--accent)", width: 14, height: 14 }}
              />
              <label htmlFor="force-cb" style={{ fontSize: "0.8rem", cursor: "pointer", color: "var(--text-secondary)" }}>
                Force re-run (skip completed pages)
              </label>
            </div>
          </div>

          {/* Status summary */}
          <div className="card">
            <div className="section-title" style={{ marginBottom: "0.9rem" }}>Status</div>
            {status?.manifest_exists ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem" }}>
                  <span style={{ color: "var(--text-muted)" }}>Total Pages</span>
                  <span style={{ fontWeight: 700 }}>{status.total}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem" }}>
                  <span style={{ color: "var(--text-muted)" }}>Phase A Complete</span>
                  <span style={{ fontWeight: 700, color: "var(--blue)" }}>{status.phase_a_completed}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem" }}>
                  <span style={{ color: "var(--text-muted)" }}>Phase B Complete</span>
                  <span style={{ fontWeight: 700, color: "var(--accent)" }}>{status.phase_b_completed}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem" }}>
                  <span style={{ color: "var(--text-muted)" }}>Run State</span>
                  <span className={`badge ${
                    runState === "running" ? "badge-amber" :
                    runState === "done" ? "badge-green" :
                    runState === "error" ? "badge-red" : "badge-muted"
                  }`}>
                    {runState}
                  </span>
                </div>
              </div>
            ) : (
              <div style={{ color: "var(--red)", fontSize: "0.82rem" }}>
                No manifest found. Go to Dashboard and click "Initialize Project" first.
              </div>
            )}
          </div>
        </div>

        {/* Live log console */}
        <div className="section-header" style={{ marginBottom: "0.5rem" }}>
          <span className="section-title">Live Output</span>
          <div style={{ display: "flex", gap: 6 }}>
            {runState === "running" && <span className="badge badge-amber">● Running</span>}
            {runState === "done" && <span className="badge badge-green">● Complete</span>}
            {runState === "error" && <span className="badge badge-red">● Error</span>}
            <button className="btn btn-ghost" style={{ fontSize: "0.72rem" }} onClick={() => setLogs([])}>
              Clear
            </button>
          </div>
        </div>

        <div
          className="log-console"
          style={{ height: 320 }}
        >
          {logs.length === 0 ? (
            <span style={{ color: "var(--text-muted)" }}>Pipeline output will appear here…</span>
          ) : (
            <Virtuoso
              style={{ height: 300 }}
              totalCount={logs.length}
              followOutput="smooth"
              overscan={50}
              itemContent={(index) => {
                const line = logs[index];
                return (
                  <div
                    style={{
                      color: line.includes("ERROR") || line.includes("failed") ? "var(--red)"
                        : line.includes("WARN") ? "var(--amber)"
                        : line.includes("complete") || line.includes("saved") ? "var(--accent)"
                        : "#a3e4a3",
                    }}
                  >
                    {line}
                  </div>
                );
              }}
            />
          )}
        </div>
      </main>
    </div>
  );
}
