"use client";

import { useEffect, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import { useProjectStore } from "@/store/projectStore";
import { fetchProjectInfo } from "@/lib/api";
import type { ProjectInfo, PageStatus } from "@/lib/types";

const ImageViewer = dynamic(() => import("@/components/ImageViewer"), { ssr: false });
const ChapterGridView = dynamic(() => import("@/components/ChapterGridView"), { ssr: false });


export default function PagesPage() {
  const { projectDir, inputSubdir } = useProjectStore();
  const [info, setInfo] = useState<ProjectInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<"all" | "review" | "done" | "pending">("all");
  const [search, setSearch] = useState("");
  const [viewerIndex, setViewerIndex] = useState<number | null>(null);
  const [viewType, setViewType] = useState<"grid" | "table">("grid");

  const load = useCallback(async () => {
    if (!projectDir) return;
    setLoading(true);
    try {
      const data = await fetchProjectInfo(projectDir, inputSubdir);
      setInfo(data);
    } catch { /* ignore */ }
    finally { setLoading(false); }
  }, [projectDir, inputSubdir]);

  useEffect(() => { load(); }, [load]);

  const pages = info?.pages ?? [];
  const filtered = pages
    .filter((p) => {
      if (filter === "review") return p.needs_review_count > 0;
      if (filter === "done") return p.phase_b_completed;
      if (filter === "pending") return !p.phase_b_completed;
      return true;
    })
    .filter((p) => !search || p.filename.toLowerCase().includes(search.toLowerCase()));

  return (
    <div>
      <Sidebar />
      <Header title="Pages" />
      <main className="main-content fade-in" style={{ padding: "0.75rem 1rem" }}>
        {/* Filters, Search and Actions Toolbar */}
        <div style={{ display: "flex", gap: 8, marginBottom: "0.75rem", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap" }}>
          <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
            {(["all", "done", "pending", "review"] as const).map((f) => (
              <button
                key={f}
                className={`btn ${filter === f ? "btn-primary" : "btn-secondary"}`}
                style={{ fontSize: "0.7rem", padding: "4px 8px", borderRadius: "4px" }}
                onClick={() => setFilter(f)}
              >
                {f === "all" ? "All" : f === "done" ? "✓ Done" : f === "pending" ? "⏳ Pending" : "⚠ Review"}
              </button>
            ))}
            <input
              className="input-field"
              style={{ maxWidth: 180, fontSize: "0.75rem", padding: "4px 8px" }}
              placeholder="Search filename…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <span style={{ color: "var(--text-muted)", fontSize: "0.72rem", marginLeft: 4 }}>
              {filtered.length} / {pages.length} pages
            </span>
          </div>

          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <button className="btn btn-secondary" style={{ fontSize: "0.7rem", padding: "4px 8px", borderRadius: "4px" }} onClick={load} disabled={loading}>
              {loading ? <span className="spin">↻</span> : "↻"} Refresh
            </button>
            <div style={{ display: "flex", background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "4px", padding: "2px" }}>
              <button
                className={`btn ${viewType === "grid" ? "btn-primary" : "btn-secondary"}`}
                style={{ fontSize: "0.7rem", padding: "4px 8px", border: "none", borderRadius: "3px" }}
                onClick={() => setViewType("grid")}
              >
                Grid
              </button>
              <button
                className={`btn ${viewType === "table" ? "btn-primary" : "btn-secondary"}`}
                style={{ fontSize: "0.7rem", padding: "4px 8px", border: "none", borderRadius: "3px" }}
                onClick={() => setViewType("table")}
              >
                Table
              </button>
            </div>
          </div>
        </div>

        {/* Dynamic View rendering */}
        {viewType === "grid" ? (
          <ChapterGridView
            projectDir={projectDir}
            inputSubdir={inputSubdir}
            pages={filtered}
            onSelectPage={(idx) => setViewerIndex(idx)}
          />
        ) : (
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ overflowY: "auto", maxHeight: "calc(100vh - 260px)" }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Filename</th>
                    <th>Phase A</th>
                    <th>Phase B</th>
                    <th>Regions</th>
                    <th>Needs Review</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.length === 0 ? (
                    <tr>
                      <td colSpan={7} style={{ textAlign: "center", padding: "2rem", color: "var(--text-muted)" }}>
                        {loading ? "Loading…" : "No pages found"}
                      </td>
                    </tr>
                  ) : filtered.map((p: PageStatus, idx) => {
                    const status = p.phase_b_completed ? "done"
                      : p.phase_a_completed ? "inpaint"
                      : "pending";
                    return (
                      <tr 
                        key={p.filename} 
                        onClick={() => setViewerIndex(idx)}
                        className="cursor-pointer hover:bg-white/5"
                      >
                        <td style={{ color: "var(--text-muted)", width: 40 }}>{idx + 1}</td>
                        <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)", fontWeight: 500 }}>
                          {p.filename}
                        </td>
                        <td>
                          <span className={`badge ${p.phase_a_completed ? "badge-green" : "badge-muted"}`}>
                            {p.phase_a_completed ? "✓ Done" : "Pending"}
                          </span>
                        </td>
                        <td>
                          <span className={`badge ${p.phase_b_completed ? "badge-green" : "badge-muted"}`}>
                            {p.phase_b_completed ? "✓ Done" : "Pending"}
                          </span>
                        </td>
                        <td style={{ fontWeight: 600 }}>{p.region_count}</td>
                        <td>
                          {p.needs_review_count > 0
                            ? <span className="badge badge-amber">⚠ {p.needs_review_count}</span>
                            : <span style={{ color: "var(--text-muted)" }}>—</span>}
                        </td>
                        <td>
                          <span className={`badge ${
                            status === "done" ? "badge-green" :
                            status === "inpaint" ? "badge-blue" : "badge-muted"
                          }`}>
                            {status === "done" ? "Complete" : status === "inpaint" ? "Phase B" : "Pending"}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>

      {viewerIndex !== null && (
        <ImageViewer
          projectDir={projectDir}
          inputSubdir={inputSubdir}
          pages={filtered}
          initialIndex={viewerIndex}
          onClose={() => setViewerIndex(null)}
        />
      )}
    </div>
  );
}
