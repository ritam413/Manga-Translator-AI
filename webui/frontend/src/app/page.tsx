"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import { useProjectStore } from "@/store/projectStore";
import {
  fetchProjectInfo,
  planProject,
  fetchImageList,
  streamExecute,
  fetchStages,
  fetchPageDetail,
  fetchBackups,
  restoreBackup,
  fetchCorrectionCount,
  streamTraining,
} from "@/lib/api";
import type { ProjectInfo, StageDefinition, BackupInfo, TextRegion } from "@/lib/types";
import { thumbUrl, imageUrl } from "@/lib/api";
import { usePageRegions } from "@/hooks/usePageRegions";
import CanvasEditor from "@/components/CanvasEditor";

const ImageViewer = dynamic(() => import("@/components/ImageViewer"), { ssr: false });

// ── Tiny helpers ────────────────────────────────────────────────────────────

function Badge({ color, children }: { color: "green" | "amber" | "red" | "blue" | "muted"; children: React.ReactNode }) {
  const cls = { green: "badge-green", amber: "badge-amber", red: "badge-red", blue: "badge-blue", muted: "badge-muted" }[color];
  return <span className={`badge ${cls}`}>{children}</span>;
}

function ProgressBar({ value, max, glow }: { value: number; max: number; glow?: boolean }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="progress-track">
      <div
        className="progress-fill"
        style={{
          width: `${pct}%`,
          background: pct === 100
            ? "linear-gradient(90deg,#22c55e,#4ade80)"
            : "linear-gradient(90deg,#3b82f6,#60a5fa)",
        }}
      />
    </div>
  );
}

// ── Left Sidebar: Image Thumbnails ──────────────────────────────────────────

function ImageThumbnailSidebar({
  projectDir,
  inputSubdir,
  files,
  activeFile,
  onSelect,
  pages,
}: {
  projectDir: string;
  inputSubdir: string;
  files: string[];
  activeFile: string | null;
  onSelect: (f: string, idx: number) => void;
  pages: ProjectInfo["pages"];
}) {
  const pageMap = Object.fromEntries(pages.map(p => [p.filename, p]));

  return (
    <aside
      className="thumbnail-sidebar clay-panel glass-backdrop"
      style={{
        width: 148,
        minWidth: 148,
        display: "flex",
        flexDirection: "column",
        overflowY: "auto",
        overflowX: "hidden",
        padding: "16px 8px",
        gap: 12,
      }}
    >
      {files.length === 0 && (
        <div style={{ color: "var(--text-muted)", fontSize: "0.7rem", textAlign: "center", padding: "2rem 0.5rem", lineHeight: 1.6 }}>
          No images.<br />Set input folder in Settings.
        </div>
      )}
      {files.map((f, idx) => {
        const isActive = f === activeFile;
        const page = pageMap[f];
        const status = page?.phase_b_completed ? "done" : page?.phase_a_completed ? "partial" : "pending";
        return (
          <div
            key={f}
            onClick={() => onSelect(f, idx)}
            title={f}
            style={{
              position: "relative",
              cursor: "pointer",
              borderRadius: 16,
              border: isActive ? "2px solid var(--accent)" : "2px solid transparent",
              overflow: "hidden",
              background: "var(--bg-clay-well)",
              transition: "border-color 0.15s, box-shadow 0.15s",
              boxShadow: isActive 
                ? "inset 2px 2px 4px rgba(0,0,0,0.5), inset -2px -2px 4px rgba(255,255,255,0.05)" 
                : "4px 4px 8px rgba(6,7,12,0.4)",
              flexShrink: 0,
            }}
          >
            {/* Page number */}
            <span style={{
              position: "absolute", top: 4, left: 4, zIndex: 2,
              fontSize: "0.6rem", fontFamily: "var(--font-mono)", fontWeight: 700,
              background: "rgba(0,0,0,0.7)", color: "#fff", padding: "1px 5px", borderRadius: 3,
            }}>
              #{idx + 1}
            </span>
            {/* Status dot */}
            <span style={{
              position: "absolute", top: 4, right: 4, zIndex: 2,
              width: 7, height: 7, borderRadius: "50%",
              background: status === "done" ? "#4ade80" : status === "partial" ? "#60a5fa" : "rgba(255,255,255,0.15)",
              boxShadow: status === "done" ? "0 0 6px #4ade80" : undefined,
            }} />
            {/* Review badge */}
            {(page?.needs_review_count ?? 0) > 0 && (
              <span style={{
                position: "absolute", bottom: 28, right: 4, zIndex: 2,
                fontSize: "0.55rem", fontWeight: 700, background: "#fbbf24", color: "#000",
                padding: "1px 4px", borderRadius: 3,
              }}>
                ⚠{page.needs_review_count}
              </span>
            )}
            <img
              src={thumbUrl(projectDir, "input", f, inputSubdir, 140)}
              alt={f}
              loading="lazy"
              style={{
                width: "100%", height: 160, objectFit: "cover", display: "block",
                opacity: isActive ? 1 : 0.65,
                transition: "opacity 0.15s",
              }}
            />
            <div style={{
              padding: "3px 5px",
              fontSize: "0.6rem", fontFamily: "var(--font-mono)", color: "rgba(255,255,255,0.55)",
              textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap",
              background: "#080a0c",
            }}>
              {f}
            </div>
          </div>
        );
      })}
    </aside>
  );
}

// ── Center: Image Preview Canvas ────────────────────────────────────────────



// ── Right Sidebar: Pipeline + Controls ─────────────────────────────────────

const STAGE_ICONS: Record<string, string> = {
  "detect-text": "🔍",
  "ocr":         "📖",
  "translate":   "🌐",
  "inpaint":     "🖌️",
  "typeset":     "✍️",
};

function PipelinePanel({
  info,
  stages,
  projectDir,
  inputSubdir,
  modelVersion,
  onModelChange,
  onRefresh,
  onPlan,
  regions,
  selectedRegionId,
  setSelectedRegionId,
  updateRegionText,
  toggleNeedsReview,
  deleteRegion,
  undo,
  redo,
  canUndo,
  canRedo,
  isDirty,
  saveRegions,
  filename,
}: {
  info: ProjectInfo | null;
  stages: StageDefinition[];
  projectDir: string;
  inputSubdir: string;
  modelVersion: "v1" | "v2";
  onModelChange: (v: "v1" | "v2") => void;
  onRefresh: () => void;
  onPlan: () => void;
  regions: TextRegion[];
  selectedRegionId: string | null;
  setSelectedRegionId: (id: string | null) => void;
  updateRegionText: (id: string, text: string) => void;
  toggleNeedsReview: (id: string) => void;
  deleteRegion: (id: string) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  isDirty: boolean;
  saveRegions: (projectDir: string, filename: string) => Promise<boolean>;
  filename: string | null;
}) {
  const [runMode, setRunMode] = useState<"all" | "stage" | "range">("all");
  const [selectedStage, setSelectedStage] = useState("detect-text");
  const [fromStage, setFromStage] = useState("detect-text");
  const [toStage, setToStage] = useState("typeset");
  const [forceRun, setForceRun] = useState(false);
  const [limitPages, setLimitPages] = useState("");
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [planMsg, setPlanMsg] = useState<string | null>(null);
  const [planning, setPlanning] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);
  const stopRef = useRef<(() => void) | null>(null);

  const [rightTab, setRightTab] = useState<"pipeline" | "overview" | "dirs" | "region">("pipeline");

  const [correctionCount, setCorrectionCount] = useState(0);
  const [backups, setBackups] = useState<BackupInfo[]>([]);

  const loadCorrectionCount = useCallback(async () => {
    if (!projectDir) return;
    try {
      const res = await fetchCorrectionCount(projectDir);
      setCorrectionCount(res.count);
    } catch (e) {
      console.error(e);
    }
  }, [projectDir]);

  const loadBackups = useCallback(async () => {
    if (!projectDir) return;
    try {
      const list = await fetchBackups(projectDir);
      setBackups(list);
    } catch (e) {
      console.error(e);
    }
  }, [projectDir]);

  // Auto-switch to Region tab when a region is selected
  useEffect(() => {
    if (selectedRegionId) {
      setRightTab("region");
    }
  }, [selectedRegionId]);

  useEffect(() => {
    loadCorrectionCount();
  }, [loadCorrectionCount, isDirty]);

  useEffect(() => {
    if (rightTab === "dirs") {
      loadBackups();
    }
  }, [rightTab, loadBackups]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const handleRun = () => {
    if (!projectDir) return;
    setRunning(true);
    setLogs(["▶ Starting pipeline..."]);

    const req: any = {
      project_dir: projectDir,
      input_subdir: inputSubdir,
      model_version: modelVersion,
      force: forceRun,
      limit: limitPages ? parseInt(limitPages) : undefined,
    };

    if (runMode === "all") req.all_stages = true;
    else if (runMode === "stage") req.stage = selectedStage;
    else { req.from_stage = fromStage; req.to_stage = toStage; }

    const stop = streamExecute(req,
      (line) => setLogs(prev => [...prev.slice(-500), line]),
      (success) => {
        setRunning(false);
        setLogs(prev => [...prev, success ? "✓ Pipeline completed successfully." : "✗ Pipeline exited with errors."]);
        onRefresh();
      }
    );
    stopRef.current = stop;
  };

  const handleStop = () => {
    stopRef.current?.();
    setRunning(false);
    setLogs(prev => [...prev, "⏹ Stopped by user."]);
  };

  const handlePlan = async () => {
    setPlanning(true);
    setPlanMsg(null);
    try {
      const result = await onPlan() as any;
      setPlanMsg(result?.success ? "✓ Manifest created/updated" : `✗ Failed`);
    } finally {
      setPlanning(false);
    }
  };

  const handleTrain = () => {
    if (!projectDir) return;
    setRunning(true);
    setLogs(["▶ Starting layout model training..."]);
    const stop = streamTraining(projectDir,
      (line) => setLogs(prev => [...prev.slice(-500), line]),
      (success) => {
        setRunning(false);
        setLogs(prev => [...prev, success ? "✓ Training complete." : "✗ Training failed."]);
        loadCorrectionCount();
        onRefresh();
      }
    );
    stopRef.current = stop;
  };

  const totalPages = info?.total_pages ?? 0;
  const phaseADone = info?.phase_a_done ?? 0;
  const phaseBDone = info?.phase_b_done ?? 0;

  return (
    <aside 
      className="clay-panel glass-backdrop"
      style={{
        width: 320, minWidth: 320,
        display: "flex", flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* Tabs */}
      <div style={{ display: "flex", padding: "12px 12px 0 12px", gap: "6px", flexShrink: 0 }}>
        {(["pipeline", "overview", "dirs", "region"] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setRightTab(tab)}
            className={rightTab === tab ? "clay-btn-active font-bold" : "clay-btn"}
            style={{
              flex: 1, padding: "8px 4px", fontSize: "0.68rem",
              textTransform: "uppercase", letterSpacing: "0.06em",
              cursor: "pointer",
            }}
          >
            {tab === "pipeline" ? "⚡ Pipeline" : tab === "overview" ? "📊 Stats" : tab === "dirs" ? "📁 Dirs" : "✏️ Region"}
          </button>
        ))}
      </div>

      <div style={{ flex: 1, overflowY: "auto", overflowX: "hidden" }}>

        {/* ── Pipeline Tab ── */}
        {rightTab === "pipeline" && (
          <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 14 }}>

            {/* Stage progress mini-view */}
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div className="section-title" style={{ marginBottom: 6 }}>Stage Progress</div>
              {stages.map(st => {
                const stageKey = st.name;
                const doneCount = stageKey === "detect-text" || stageKey === "ocr" || stageKey === "translate"
                  ? phaseADone : phaseBDone;
                const pct = totalPages > 0 ? Math.round(doneCount / totalPages * 100) : 0;
                return (
                  <div key={st.id} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: "1rem", width: 22, flexShrink: 0 }}>{STAGE_ICONS[st.name] ?? "⚙"}</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                        <span style={{ fontSize: "0.68rem", color: "rgba(255,255,255,0.6)" }}>{st.label}</span>
                        <span style={{ fontSize: "0.68rem", fontWeight: 600, color: pct === 100 ? "#4ade80" : "rgba(255,255,255,0.35)" }}>{pct}%</span>
                      </div>
                      <ProgressBar value={doneCount} max={totalPages} />
                    </div>
                  </div>
                );
              })}
            </div>

            <hr style={{ border: "none", borderTop: "1px solid rgba(255,255,255,0.05)" }} />

            {/* Run mode */}
            <div>
              <div className="section-title" style={{ marginBottom: 8 }}>Run Mode</div>
              <div style={{ display: "flex", gap: 4 }}>
                {(["all", "stage", "range"] as const).map(m => (
                  <button
                    key={m}
                    onClick={() => setRunMode(m)}
                    className={runMode === m ? "clay-btn-active" : "clay-btn"}
                    style={{
                      flex: 1, padding: "6px 0", fontSize: "0.68rem", fontWeight: 600,
                    }}
                  >
                    {m === "all" ? "All" : m === "stage" ? "Single" : "Range"}
                  </button>
                ))}
              </div>

              {runMode === "stage" && (
                <select
                  value={selectedStage}
                  onChange={e => setSelectedStage(e.target.value)}
                  className="input-field"
                  style={{ marginTop: 8, fontSize: "0.75rem" }}
                >
                  {stages.map(s => <option key={s.name} value={s.name}>{STAGE_ICONS[s.name]} {s.label}</option>)}
                </select>
              )}

              {runMode === "range" && (
                <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: "0.65rem", color: "rgba(255,255,255,0.3)", marginBottom: 3 }}>From</div>
                    <select value={fromStage} onChange={e => setFromStage(e.target.value)} className="input-field" style={{ fontSize: "0.72rem" }}>
                      {stages.map(s => <option key={s.name} value={s.name}>{s.label}</option>)}
                    </select>
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: "0.65rem", color: "rgba(255,255,255,0.3)", marginBottom: 3 }}>To</div>
                    <select value={toStage} onChange={e => setToStage(e.target.value)} className="input-field" style={{ fontSize: "0.72rem" }}>
                      {stages.map(s => <option key={s.name} value={s.name}>{s.label}</option>)}
                    </select>
                  </div>
                </div>
              )}
            </div>

            {/* Model version */}
            <div>
              <div className="section-title" style={{ marginBottom: 8 }}>Model Version</div>
              <div style={{ display: "flex", gap: 4 }}>
                {(["v2", "v1"] as const).map(v => (
                  <button
                    key={v}
                    onClick={() => onModelChange(v)}
                    className={modelVersion === v ? "clay-btn-active" : "clay-btn"}
                    style={{
                      flex: 1, padding: "6px 0", fontSize: "0.68rem", fontWeight: 600,
                    }}
                  >
                    {v === "v2" ? "v2 Merged" : "v1 Single"}
                  </button>
                ))}
              </div>
            </div>

            {/* Options */}
            <div>
              <div className="section-title" style={{ marginBottom: 8 }}>Options</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={forceRun}
                    onChange={e => setForceRun(e.target.checked)}
                    style={{ accentColor: "#4ade80", width: 14, height: 14 }}
                  />
                  <span style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.6)" }}>Force re-run (overwrite)</span>
                </label>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.6)", whiteSpace: "nowrap" }}>Limit pages:</span>
                  <input
                    type="number"
                    value={limitPages}
                    onChange={e => setLimitPages(e.target.value)}
                    placeholder="All"
                    min={1}
                    className="input-field"
                    style={{ width: 70, fontSize: "0.72rem", padding: "4px 8px" }}
                  />
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {!running ? (
                <button
                  className="clay-btn-primary"
                  onClick={handleRun}
                  disabled={!info || !projectDir}
                  style={{ width: "100%", justifyContent: "center", fontSize: "0.82rem", padding: "10px 16px" }}
                >
                  ⚡ Run Pipeline
                </button>
              ) : (
                <button
                  className="clay-btn"
                  onClick={handleStop}
                  style={{ width: "100%", justifyContent: "center", fontSize: "0.82rem", color: "#f87171", padding: "10px 16px" }}
                >
                  ⏹ Stop
                </button>
              )}
              
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <button
                  className={correctionCount > 0 ? "clay-btn-active font-bold" : "clay-btn font-bold"}
                  onClick={handleTrain}
                  disabled={running || !projectDir}
                  style={{
                    width: "100%",
                    justifyContent: "center",
                    fontSize: "0.75rem",
                    padding: "8px 16px",
                  }}
                >
                  🧠 Train Layout Model ({correctionCount})
                </button>
              </div>

              <div style={{ display: "flex", gap: 6 }}>
                <button
                  className="clay-btn"
                  onClick={handlePlan}
                  disabled={planning || !projectDir}
                  style={{ flex: 1, justifyContent: "center", fontSize: "0.75rem", padding: "8px 16px" }}
                >
                  {planning ? <span className="spin">↻</span> : "⬡"} {info?.manifest_exists ? "Re-Plan" : "Initialize"}
                </button>
                <button
                  className="clay-btn"
                  onClick={onRefresh}
                  style={{ flex: 1, justifyContent: "center", fontSize: "0.75rem", padding: "8px 16px" }}
                >
                  ↻ Refresh
                </button>
              </div>
              {planMsg && (
                <div style={{
                  padding: "6px 10px", borderRadius: 12, fontSize: "0.72rem",
                  background: planMsg.startsWith("✓") ? "rgba(163,230,53,0.08)" : "rgba(248,113,113,0.08)",
                  border: `1px solid ${planMsg.startsWith("✓") ? "rgba(163,230,53,0.25)" : "rgba(248,113,113,0.25)"}`,
                  color: planMsg.startsWith("✓") ? "var(--accent)" : "#f87171",
                }}>
                  {planMsg}
                </div>
              )}
            </div>

            {/* Log Console */}
            {logs.length > 0 && (
              <div>
                <div className="section-title" style={{ marginBottom: 6 }}>Live Output</div>
                <div
                  ref={logRef}
                  className="log-console"
                  style={{ maxHeight: 180, fontSize: "0.68rem", padding: "8px 10px", lineHeight: "1.7", borderRadius: "16px" }}
                >
                  {logs.map((l, i) => (
                    <div key={i} style={{
                      color: l.startsWith("✓") ? "var(--accent)" : l.startsWith("✗") || l.startsWith("⏹") ? "#f87171" : "#a3e4a3",
                    }}>
                      {l}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Overview Tab ── */}
        {rightTab === "overview" && (
          <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 12 }}>
            {!info ? (
              <div style={{ color: "rgba(255,255,255,0.3)", fontSize: "0.78rem" }}>No project loaded.</div>
            ) : (
              <>
                {/* Stat cards */}
                {[
                  { label: "Total Pages", value: info.total_pages, color: "var(--text-primary)" },
                  { label: "Phase A Done", value: info.phase_a_done, color: "#4ade80" },
                  { label: "Phase B Done", value: info.phase_b_done, color: "#4ade80" },
                  { label: "Needs Review", value: info.needs_review, color: "#fbbf24" },
                ].map(s => (
                  <div key={s.label} className="card" style={{ padding: "10px 14px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.5)" }}>{s.label}</span>
                    <span style={{ fontSize: "1.4rem", fontWeight: 700, color: s.color }}>{s.value}</span>
                  </div>
                ))}
                {/* Language */}
                <div className="card" style={{ padding: "10px 14px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: "0.72rem", color: "rgba(255,255,255,0.4)" }}>Source</span>
                    <Badge color="blue">{info.source_language.toUpperCase()}</Badge>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ fontSize: "0.72rem", color: "rgba(255,255,255,0.4)" }}>Target</span>
                    <Badge color="green">{info.target_language.toUpperCase()}</Badge>
                  </div>
                </div>
                {/* Progress */}
                <div className="card" style={{ padding: "10px 14px" }}>
                  <div className="section-title" style={{ marginBottom: 10 }}>Page Execution Progress</div>
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", marginBottom: 4 }}>
                      <span style={{ color: "rgba(255,255,255,0.5)" }}>Phase A (OCR/Match)</span>
                      <span style={{ color: "rgba(255,255,255,0.8)", fontWeight: 600 }}>{info.phase_a_done}/{info.total_pages}</span>
                    </div>
                    <ProgressBar value={info.phase_a_done} max={info.total_pages} />
                  </div>
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", marginBottom: 4 }}>
                      <span style={{ color: "rgba(255,255,255,0.5)" }}>Phase B (Typeset)</span>
                      <span style={{ color: "rgba(255,255,255,0.8)", fontWeight: 600 }}>{info.phase_b_done}/{info.total_pages}</span>
                    </div>
                    <ProgressBar value={info.phase_b_done} max={info.total_pages} />
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* ── Dirs Tab ── */}
        {rightTab === "dirs" && (
          <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 12 }}>
            {[
              { label: "📥 Input", dir: info?.input_dir ?? null },
              { label: "📤 Output", dir: info?.output_dir ?? null },
            ].map(({ label, dir }) => (
              <div key={label} className="card" style={{ padding: "10px 14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <span style={{ fontSize: "0.78rem", fontWeight: 600 }}>{label}</span>
                  {dir && <Badge color={dir.exists ? "green" : "red"}>{dir.exists ? "Found" : "Missing"}</Badge>}
                </div>
                {dir ? (
                  <>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "rgba(255,255,255,0.3)", wordBreak: "break-all", marginBottom: 8 }}>
                      {dir.path}
                    </div>
                    <div style={{ display: "flex", gap: 16 }}>
                      <div>
                        <div style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--text-primary)" }}>{dir.image_count}</div>
                        <div style={{ fontSize: "0.65rem", color: "rgba(255,255,255,0.3)", textTransform: "uppercase" }}>Images</div>
                      </div>
                      <div>
                        <div style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--text-primary)" }}>{dir.total_size_mb}</div>
                        <div style={{ fontSize: "0.65rem", color: "rgba(255,255,255,0.3)", textTransform: "uppercase" }}>MB</div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="skeleton" style={{ height: 50 }} />
                )}
              </div>
            ))}

            {/* Backups List */}
            <div className="card" style={{ padding: "10px 14px" }}>
              <div className="section-title" style={{ marginBottom: 10, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>Manifest Backups</span>
                <button onClick={loadBackups} style={{ background: "none", border: "none", color: "#4ade80", cursor: "pointer", fontSize: "0.65rem" }}>↻ Refresh</button>
              </div>
              {backups.length === 0 ? (
                <div style={{ fontSize: "0.68rem", color: "rgba(255,255,255,0.3)" }}>No backups found. Backups are automatically created on page edits.</div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 150, overflowY: "auto", paddingRight: 4 }}>
                  {backups.map(b => (
                    <div key={b.filename} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.65rem", padding: "4px 6px", background: "rgba(255,255,255,0.02)", borderRadius: 4 }}>
                      <span title={b.filename} style={{ color: "rgba(255,255,255,0.7)" }}>{b.timestamp}</span>
                      <button
                        onClick={async () => {
                          if (confirm(`Are you sure you want to restore the manifest to this state (${b.timestamp})? Any edits since this backup will be overwritten.`)) {
                            try {
                              const res = await restoreBackup(projectDir, b.filename);
                              if (res.success) {
                                alert("Manifest restored successfully!");
                                onRefresh();
                                loadBackups();
                              }
                            } catch (err) {
                              alert("Failed to restore backup.");
                            }
                          }
                        }}
                        style={{ background: "transparent", border: "none", color: "#4ade80", cursor: "pointer", fontSize: "0.6rem" }}
                      >
                        Restore
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Quick settings */}
            <div className="card" style={{ padding: "10px 14px" }}>
              <div className="section-title" style={{ marginBottom: 10 }}>Quick Settings</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <div>
                  <label style={{ fontSize: "0.68rem", color: "rgba(255,255,255,0.35)", display: "block", marginBottom: 4 }}>Input Subfolder</label>
                  <InputSubdirField />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Region Tab ── */}
        {rightTab === "region" && (
          <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 12 }}>
            {!selectedRegionId ? (
              <div style={{ color: "rgba(255,255,255,0.3)", fontSize: "0.78rem", textAlign: "center", padding: "2rem 0" }}>
                Select a speech bubble box on the page to edit its translation.
              </div>
            ) : (() => {
              const region = regions.find(r => r.id === selectedRegionId);
              if (!region) return null;
              return (
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: "0.82rem", fontWeight: 700, color: "#4ade80" }}>Region {region.id}</span>
                    <button
                      onClick={() => deleteRegion(region.id)}
                      style={{ color: "#f87171", background: "transparent", border: "none", cursor: "pointer", fontSize: "0.75rem" }}
                    >
                      🗑 Delete
                    </button>
                  </div>
                  <hr style={{ border: "none", borderTop: "1px solid rgba(255,255,255,0.05)" }} />
                  
                  {/* Original JA text */}
                  <div>
                    <label style={{ fontSize: "0.68rem", color: "rgba(255,255,255,0.35)", display: "block", marginBottom: 4 }}>
                      ORIGINAL (JA) — READ ONLY
                    </label>
                    <textarea
                      className="input-field font-sans"
                      readOnly
                      value={region.raw_text || "(no OCR text)"}
                      style={{ width: "100%", height: 60, fontSize: "0.75rem", background: "rgba(255,255,255,0.02)", color: "rgba(255,255,255,0.65)" }}
                    />
                  </div>
                  
                  {/* Translated EN text */}
                  <div>
                    <label style={{ fontSize: "0.68rem", color: "rgba(255,255,255,0.35)", display: "block", marginBottom: 4 }}>
                      TRANSLATION (EN) ✏️
                    </label>
                    <textarea
                      className="input-field font-sans"
                      value={region.translated_text}
                      onChange={e => updateRegionText(region.id, e.target.value)}
                      placeholder="Enter translation..."
                      style={{
                        width: "100%",
                        height: 100,
                        fontSize: "0.75rem",
                        borderColor: "rgba(74, 222, 128, 0.2)",
                      }}
                    />
                  </div>
                  
                  {/* Needs review checkbox */}
                  <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", margin: "4px 0" }}>
                    <input
                      type="checkbox"
                      checked={region.needs_review}
                      onChange={() => toggleNeedsReview(region.id)}
                      style={{ accentColor: "#fbbf24", width: 14, height: 14 }}
                    />
                    <span style={{ fontSize: "0.75rem", color: "#fbbf24", fontWeight: 600 }}>⚠ Needs Review</span>
                  </label>
                  
                  {/* Box coordinates */}
                  <div style={{ fontSize: "0.65rem", color: "rgba(255,255,255,0.3)", display: "flex", flexDirection: "column", gap: 2 }}>
                    <div>Box Bounds: ({region.box.x_min}, {region.box.y_min}) → ({region.box.x_max}, {region.box.y_max})</div>
                    <div>Dimensions: {region.box.x_max - region.box.x_min}w × {region.box.y_max - region.box.y_min}h px</div>
                    <div>OCR Confidence: {region.ocr_confidence.toFixed(2)}</div>
                  </div>
                  
                  {/* Shortcuts help */}
                  <div className="card" style={{ padding: "8px 10px", fontSize: "0.62rem", color: "rgba(255,255,255,0.3)", lineHeight: 1.5 }}>
                    <strong>Shortcuts:</strong><br />
                    • <kbd>Ctrl+S</kbd>: Save page<br />
                    • <kbd>Ctrl+Z</kbd> / <kbd>Ctrl+Shift+Z</kbd>: Undo / Redo<br />
                    • <kbd>Del</kbd>: Delete region
                  </div>
                </div>
              );
            })()}
          </div>
        )}

      </div>
    </aside>
  );
}

// ── Small inline components ─────────────────────────────────────────────────

function InputSubdirField() {
  const { inputSubdir, setInputSubdir } = useProjectStore();
  return (
    <input
      className="input-field"
      value={inputSubdir}
      onChange={e => setInputSubdir(e.target.value)}
      placeholder="input"
      style={{ fontSize: "0.75rem" }}
    />
  );
}

// ── Top Navbar ──────────────────────────────────────────────────────────────

function TopNavbar({
  info,
  loading,
  projectDir,
  onNavigate,
  backendAlive,
  engineAlive,
  isDirty,
}: {
  info: ProjectInfo | null;
  loading: boolean;
  projectDir: string;
  onNavigate: (path: string) => void;
  backendAlive: boolean;
  engineAlive: boolean;
  isDirty?: boolean;
}) {
  return (
    <header 
      className="glass-backdrop"
      style={{
        height: 56,
        borderRadius: "0 0 24px 24px",
        boxShadow: "12px 12px 24px rgba(6,7,12,0.5), -8px -8px 16px rgba(255,255,255,0.02)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 24px",
        margin: "0 16px",
        flexShrink: 0,
        zIndex: 10,
      }}
    >
      {/* Logo + page count */}
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <div style={{
            width: 26, height: 26, borderRadius: 10,
            background: "rgba(163,230,53,0.15)", border: "1px solid rgba(163,230,53,0.3)",
            display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.9rem",
          }}>🎌</div>
          <div>
            <div style={{ fontSize: "0.75rem", fontWeight: 700, lineHeight: 1.1 }}>MangaFlow</div>
            <div style={{ fontSize: "0.55rem", color: "rgba(255,255,255,0.3)", letterSpacing: "0.05em" }}>TRANSLATOR</div>
          </div>
        </div>
        <div style={{ width: 1, height: 20, background: "rgba(255,255,255,0.06)" }} />
        <span style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.4)", fontWeight: 500 }}>
          {loading ? "Loading…" : info ? `Pages (${info.total_pages})` : "No project loaded"}
        </span>
        {(info?.needs_review ?? 0) > 0 && (
          <Badge color="amber">⚠ {info?.needs_review} review</Badge>
        )}
      </div>

      {/* Nav links */}
      <nav style={{ display: "flex", gap: 2 }}>
        {[
          { href: "/", label: "Dashboard" },
          { href: "/pipeline", label: "Pipeline" },
          { href: "/pages", label: "Pages" },
          { href: "/settings", label: "Settings" },
        ].map(n => (
          <a
            key={n.href}
            href={n.href}
            style={{
              padding: "4px 12px", borderRadius: 5, fontSize: "0.72rem", fontWeight: 500,
              color: "rgba(255,255,255,0.5)", textDecoration: "none",
              transition: "all 0.15s",
            }}
            onMouseEnter={e => (e.currentTarget.style.color = "#fff")}
            onMouseLeave={e => (e.currentTarget.style.color = "rgba(255,255,255,0.5)")}
          >
            {n.label}
          </a>
        ))}
      </nav>

      {/* Right section */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span style={{ fontSize: "0.62rem", fontFamily: "var(--font-mono)", color: "rgba(255,255,255,0.25)" }}>
          {projectDir ? projectDir.split(/[/\\]/).pop() : "—"}
        </span>
        
        {/* Backend Status */}
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <div style={{
            width: 7, height: 7, borderRadius: "50%",
            background: backendAlive ? "var(--accent)" : "#f87171",
            boxShadow: backendAlive ? "0 0 6px var(--accent)" : "0 0 6px #f87171",
          }} />
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.4)" }}>API</span>
        </div>

        {/* Engine Status */}
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <div style={{
            width: 7, height: 7, borderRadius: "50%",
            background: (backendAlive && engineAlive) ? "var(--accent)" : "#fbbf24",
            boxShadow: (backendAlive && engineAlive) ? "0 0 6px var(--accent)" : "0 0 6px #fbbf24",
          }} />
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.4)" }}>Engine</span>
        </div>
      </div>
    </header>
  );
}

// ── Main Page ───────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { projectDir, inputSubdir, modelVersion, setModelVersion } = useProjectStore();
  const [info, setInfo] = useState<ProjectInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stages, setStages] = useState<StageDefinition[]>([]);
  const [inputFiles, setInputFiles] = useState<string[]>([]);
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [viewerIndex, setViewerIndex] = useState<number | null>(null);

  // Connection health indicator
  const [backendAlive, setBackendAlive] = useState(true);
  const [engineAlive, setEngineAlive] = useState(false);

  // Region editor hook
  const {
    regions,
    selectedRegionId,
    setSelectedRegionId,
    isDirty,
    canUndo,
    canRedo,
    reset: resetRegions,
    addRegion,
    updateRegionBox,
    updateRegionText,
    toggleNeedsReview,
    deleteRegion,
    undo,
    redo,
    save: saveRegions,
  } = usePageRegions();

  const [pageWidth, setPageWidth] = useState(800);
  const [pageHeight, setPageHeight] = useState(1200);
  const [drawMode, setDrawMode] = useState(false);

  // Establish WebSocket connection for real-time backend health check
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: NodeJS.Timeout;

    const connect = () => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      // Connect directly to the FastAPI port 8000
      const wsUrl = `${protocol}//localhost:8000/api/ws/health`;

      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setBackendAlive(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.status === "healthy") {
            setBackendAlive(true);
            setEngineAlive(data.engine_connected ?? false);
          }
        } catch {}
      };

      ws.onclose = () => {
        setBackendAlive(false);
        setEngineAlive(false);
        reconnectTimer = setTimeout(connect, 3000); // Reconnect in 3s
      };

      ws.onerror = () => {
        setBackendAlive(false);
        setEngineAlive(false);
        ws?.close();
      };
    };

    connect();

    return () => {
      ws?.close();
      clearTimeout(reconnectTimer);
    };
  }, []);

  const load = useCallback(async () => {
    if (!projectDir) return;
    setLoading(true);
    setError(null);
    try {
      const [data, imgData] = await Promise.all([
        fetchProjectInfo(projectDir, inputSubdir),
        fetchImageList(projectDir, "input", inputSubdir).catch(() => ({ files: [] as string[] })),
      ]);
      setInfo(data);
      setInputFiles(imgData.files);
      if (!activeFile && imgData.files.length > 0) {
        setActiveFile(imgData.files[0]);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load project info");
    } finally {
      setLoading(false);
    }
  }, [projectDir, inputSubdir, activeFile]);

  useEffect(() => {
    fetchStages().then(d => setStages(d.stages)).catch(() => {});
  }, []);

  useEffect(() => { load(); }, [load]);

  // Load detailed page manifest regions on page switch
  useEffect(() => {
    if (!projectDir || !activeFile) return;
    const fetchDetail = async () => {
      try {
        const detail = await fetchPageDetail(projectDir, activeFile);
        setPageWidth(detail.width || 800);
        setPageHeight(detail.height || 1200);
        resetRegions(detail.text_regions || []);
      } catch (err) {
        console.error("Failed to load page details", err);
      }
    };
    fetchDetail();
  }, [projectDir, activeFile, resetRegions]);

  const handlePlan = async () => {
    const result = await planProject(projectDir, inputSubdir);
    await load();
    return result;
  };

  const activeIndex = activeFile ? inputFiles.indexOf(activeFile) : -1;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "transparent", overflow: "hidden" }}>
      <TopNavbar info={info} loading={loading} projectDir={projectDir} onNavigate={() => {}} isDirty={isDirty} backendAlive={backendAlive} engineAlive={engineAlive} />

      {error && (
        <div style={{
          padding: "6px 16px", background: "rgba(248,113,113,0.08)", borderBottom: "1px solid rgba(248,113,113,0.2)",
          fontSize: "0.75rem", color: "#f87171", flexShrink: 0,
          zIndex: 5,
        }}>
          ⚠ {error} — Make sure the backend is running on port 8000.
        </div>
      )}

      {/* Main 3-panel layout */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", padding: "16px", gap: "16px", zIndex: 1, position: "relative" }}>
        {/* Left: Thumbnails */}
        <ImageThumbnailSidebar
          projectDir={projectDir}
          inputSubdir={inputSubdir}
          files={inputFiles}
          activeFile={activeFile}
          onSelect={(f, idx) => setActiveFile(f)}
          pages={info?.pages ?? []}
        />

        {/* Center: Interactive Canvas Editor */}
        {activeFile ? (
          <CanvasEditor
            projectDir={projectDir}
            inputSubdir={inputSubdir}
            filename={activeFile}
            imageWidth={pageWidth}
            imageHeight={pageHeight}
            regions={regions}
            selectedRegionId={selectedRegionId}
            setSelectedRegionId={setSelectedRegionId}
            addRegion={addRegion}
            updateRegionBox={updateRegionBox}
            deleteRegion={deleteRegion}
            drawMode={drawMode}
            setDrawMode={setDrawMode}
            undo={undo}
            redo={redo}
            canUndo={canUndo}
            canRedo={canRedo}
            save={saveRegions}
            isDirty={isDirty}
          />
        ) : (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: "rgba(255,255,255,0.2)", gap: 12 }}>
            <div style={{ fontSize: "4rem" }}>🎌</div>
            <div style={{ fontSize: "0.9rem", fontWeight: 600 }}>Select an image to edit regions</div>
            <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.12)" }}>
              {info ? `${info.input_dir.image_count} images in input folder` : "No project loaded"}
            </div>
          </div>
        )}

        {/* Right: Pipeline / Overview / Dirs / Region Editor Panel */}
        <PipelinePanel
          info={info}
          stages={stages}
          projectDir={projectDir}
          inputSubdir={inputSubdir}
          modelVersion={modelVersion}
          onModelChange={setModelVersion}
          onRefresh={load}
          onPlan={handlePlan}
          regions={regions}
          selectedRegionId={selectedRegionId}
          setSelectedRegionId={setSelectedRegionId}
          updateRegionText={updateRegionText}
          toggleNeedsReview={toggleNeedsReview}
          deleteRegion={deleteRegion}
          undo={undo}
          redo={redo}
          canUndo={canUndo}
          canRedo={canRedo}
          isDirty={isDirty}
          saveRegions={saveRegions}
          filename={activeFile}
        />
      </div>

      {/* Full-page viewer overlay */}
      {viewerIndex !== null && info && (
        <ImageViewer
          projectDir={projectDir}
          inputSubdir={inputSubdir}
          pages={info.pages}
          initialIndex={viewerIndex}
          onClose={() => setViewerIndex(null)}
        />
      )}
    </div>
  );
}
