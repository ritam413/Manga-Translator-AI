"use client";

import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import { useProjectStore } from "@/store/projectStore";

export default function SettingsPage() {
  const {
    projectDir, setProjectDir,
    inputSubdir, setInputSubdir,
    modelVersion, setModelVersion,
  } = useProjectStore();

  return (
    <div>
      <Sidebar />
      <Header title="Settings" />
      <main className="main-content fade-in">
        <div style={{ maxWidth: 640 }}>
          <h1 style={{ fontSize: "1.4rem", marginBottom: 4 }}>Settings</h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.82rem", marginBottom: "2rem" }}>
            Configure the project directory and pipeline options. Changes are saved to localStorage automatically.
          </p>

          <div className="card" style={{ marginBottom: "1.25rem" }}>
            <div className="section-title" style={{ marginBottom: "1rem" }}>Project</div>

            <div style={{ marginBottom: "1rem" }}>
              <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", display: "block", marginBottom: 6 }}>
                Project Directory
              </label>
              <input
                className="input-field"
                value={projectDir}
                onChange={(e) => setProjectDir(e.target.value)}
                placeholder="e.g. e:\u torrentz\Manga Translator"
              />
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 4 }}>
                Absolute path to your manga project folder (must contain input/ and project_manifest.json)
              </div>
            </div>

            <div style={{ marginBottom: "1rem" }}>
              <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", display: "block", marginBottom: 6 }}>
                Input Subdirectory
              </label>
              <input
                className="input-field"
                value={inputSubdir}
                onChange={(e) => setInputSubdir(e.target.value)}
                placeholder="input"
              />
            </div>
          </div>

          <div className="card" style={{ marginBottom: "1.25rem" }}>
            <div className="section-title" style={{ marginBottom: "1rem" }}>Pipeline</div>

            <div>
              <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", display: "block", marginBottom: 6 }}>
                Default Model Version
              </label>
              <div style={{ display: "flex", gap: 8 }}>
                {(["v2", "v1"] as const).map((v) => (
                  <button
                    key={v}
                    className={`btn ${modelVersion === v ? "btn-primary" : "btn-secondary"}`}
                    onClick={() => setModelVersion(v)}
                  >
                    {v === "v2" ? "v2 — Merged Bubble (recommended)" : "v1 — Single Word / Column"}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="section-title" style={{ marginBottom: "1rem" }}>Backend</div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{
                width: 9, height: 9,
                background: "var(--accent)",
                borderRadius: "50%",
                boxShadow: "0 0 6px var(--accent)",
                display: "inline-block",
              }} />
              <div>
                <div style={{ fontSize: "0.82rem", fontWeight: 600 }}>FastAPI Backend</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--text-muted)" }}>
                  http://localhost:8000
                </div>
              </div>
            </div>
            <div style={{ marginTop: "0.75rem", fontSize: "0.78rem", color: "var(--text-secondary)" }}>
              Start with: <code style={{ fontFamily: "var(--font-mono)", background: "var(--bg-elevated)", padding: "1px 6px", borderRadius: 3 }}>
                python -m uvicorn main:app --reload
              </code> from <code style={{ fontFamily: "var(--font-mono)", background: "var(--bg-elevated)", padding: "1px 6px", borderRadius: 3 }}>webui/backend/</code>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
