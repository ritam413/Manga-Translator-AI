"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/",         icon: "⬡",  label: "Dashboard"  },
  { href: "/pipeline", icon: "⚡", label: "Pipeline"   },
  { href: "/pages",    icon: "📄", label: "Pages"      },
  { href: "/settings", icon: "⚙", label: "Settings"   },
];

export default function Sidebar() {
  const path = usePathname();

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div style={{
          width: 30, height: 30,
          background: "var(--accent-muted)",
          border: "1px solid var(--border-accent)",
          borderRadius: 8,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "0.95rem",
        }}>🎌</div>
        <div>
          <div style={{ fontWeight: 700, fontSize: "0.85rem", lineHeight: 1.2 }}>
            Manga Translator
          </div>
          <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", letterSpacing: "0.05em" }}>
            PIPELINE v2
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <div className="section-title" style={{ padding: "0.4rem 0.75rem 0.5rem", marginBottom: 2 }}>
          Navigation
        </div>
        {NAV.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={`nav-item ${path === item.href ? "active" : ""}`}
          >
            <span style={{ width: 18, textAlign: "center" }}>{item.icon}</span>
            {item.label}
          </Link>
        ))}
      </nav>

      {/* Footer */}
      <div style={{
        padding: "0.75rem 1rem",
        borderTop: "1px solid var(--border)",
        fontSize: "0.7rem",
        color: "var(--text-muted)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{
            width: 7, height: 7,
            borderRadius: "50%",
            background: "var(--accent)",
            boxShadow: "0 0 6px var(--accent)",
            display: "inline-block",
          }} />
          Backend: localhost:8000
        </div>
      </div>
    </aside>
  );
}
