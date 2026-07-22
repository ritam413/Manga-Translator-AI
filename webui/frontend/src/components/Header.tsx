"use client";

import { useProjectStore } from "@/store/projectStore";

export default function Header({ title }: { title: string }) {
  const projectDir = useProjectStore((s) => s.projectDir);

  return (
    <header className="main-header">
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <h2 style={{ fontSize: "0.95rem", fontWeight: 600 }}>{title}</h2>
        {projectDir && (
          <span className="badge badge-muted" style={{ fontFamily: "var(--font-mono)", maxWidth: 340, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {projectDir}
          </span>
        )}
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <span className="badge badge-green">● Live</span>
      </div>
    </header>
  );
}
