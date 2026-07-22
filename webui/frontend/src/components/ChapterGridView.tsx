import { useEffect, useRef, forwardRef } from "react";
import gsap from "gsap";
import { VirtuosoGrid } from "react-virtuoso";
import { imageUrl } from "@/lib/api";
import { PageStatus } from "@/lib/types";

interface ChapterGridViewProps {
  projectDir: string;
  inputSubdir: string;
  pages: PageStatus[];
  onSelectPage: (index: number) => void;
}

// ── Single Thumbnail Card ───────────────────────────────────────────────────

function ThumbnailCard({
  page,
  index,
  projectDir,
  inputSubdir,
  onClick,
}: {
  page: PageStatus;
  index: number;
  projectDir: string;
  inputSubdir: string;
  onClick: () => void;
}) {
  const tUrl = imageUrl(projectDir, "input", page.filename, inputSubdir);
  const status = page.phase_b_completed ? "done" : page.phase_a_completed ? "inpaint" : "pending";

  return (
    <div
      onClick={onClick}
      className="thumbnail-card group relative bg-[#0d1117] rounded-lg border border-white/5 hover:border-[#4ade80]/40 overflow-hidden cursor-pointer shadow-md hover:shadow-[0_0_15px_rgba(74,222,128,0.2)] transition-all duration-300 flex flex-col"
      style={{ width: "150px" }}
    >
      {/* Thumbnail Image Container */}
      <div className="relative w-full bg-[#070a0d] overflow-hidden flex items-center justify-center border-b border-white/5" style={{ height: "200px" }}>
        <img
          src={tUrl}
          alt={page.filename}
          loading="lazy"
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
          style={{ imageRendering: "high-quality" as any }}
        />

        {/* Floating Index Badge */}
        <span className="absolute top-1.5 left-1.5 px-2 py-0.5 bg-black/75 backdrop-blur-md text-[10px] font-mono font-bold rounded border border-white/10 text-white/80">
          #{index + 1}
        </span>

        {/* Needs Review Badge */}
        {page.needs_review_count > 0 && (
          <span className="absolute top-1.5 right-1.5 px-2 py-0.5 bg-amber/90 backdrop-blur-md text-[10px] font-bold rounded border border-amber/30 text-black flex items-center shadow-lg animate-pulse">
            ⚠ {page.needs_review_count}
          </span>
        )}
      </div>

      {/* Info Bottom Area */}
      <div className="p-2 flex-1 flex flex-col justify-between gap-1.5 bg-[#0a0c0f]">
        <div className="font-mono text-xs font-bold text-white/90 truncate text-center" title={page.filename}>
          {page.filename}
        </div>
        <div className="flex items-center justify-between text-[10px] text-white/50">
          <span>{page.region_count} regions</span>
          <span className={`badge ${
            status === "done" ? "badge-green" :
            status === "inpaint" ? "badge-blue" : "badge-muted"
          }`} style={{ fontSize: "9px", padding: "2px 6px" }}>
            {status === "done" ? "Done" : status === "inpaint" ? "Ph.B" : "Pend"}
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Virtuoso Grid Custom Layout Components ─────────────────────────────────

const GridList = forwardRef<HTMLDivElement, any>(({ style, children, ...props }, ref) => (
  <div
    ref={ref}
    {...props}
    style={{
      ...style,
      display: "grid",
      gridTemplateColumns: "repeat(auto-fill, 150px)",
      gap: "0.75rem",
      paddingLeft: "0.25rem",
      paddingRight: "0.25rem",
    }}
  >
    {children}
  </div>
));
GridList.displayName = "GridList";

const GridItem = ({ children, ...props }: any) => (
  <div {...props} className="flex flex-col">
    {children}
  </div>
);
GridItem.displayName = "GridItem";

// ── Main ChapterGridView Component ──────────────────────────────────────────

export default function ChapterGridView({
  projectDir,
  inputSubdir,
  pages,
  onSelectPage,
}: ChapterGridViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // GSAP Stagger Entrance Animation for static grid
  useEffect(() => {
    if (pages.length <= 100 && containerRef.current) {
      const cards = containerRef.current.querySelectorAll(".thumbnail-card");
      if (cards.length > 0) {
        gsap.killTweensOf(cards);
        gsap.fromTo(
          cards,
          { y: 30, opacity: 0 },
          {
            y: 0,
            opacity: 1,
            stagger: 0.02,
            duration: 0.4,
            ease: "power2.out",
            overwrite: "auto",
          }
        );
      }
    }
  }, [pages]);

  if (pages.length === 0) {
    return (
      <div className="flex items-center justify-center p-12 text-white/40 text-sm">
        No pages to display.
      </div>
    );
  }

  // Use virtualization for 100+ pages
  if (pages.length > 100) {
    return (
      <div className="w-full relative" style={{ height: "calc(100vh - 120px)" }}>
        <VirtuosoGrid
          style={{ height: "100%", outline: "none" }}
          totalCount={pages.length}
          components={{
            List: GridList,
            Item: GridItem,
          }}
          itemContent={(index) => (
            <ThumbnailCard
              page={pages[index]}
              index={index}
              projectDir={projectDir}
              inputSubdir={inputSubdir}
              onClick={() => onSelectPage(index)}
            />
          )}
        />
      </div>
    );
  }

  // Use simple grid with GSAP animations for <= 100 pages
  return (
    <div
      ref={containerRef}
      className="grid gap-4 overflow-y-auto pr-1"
      style={{
        gridTemplateColumns: "repeat(auto-fill, 150px)",
        maxHeight: "calc(100vh - 120px)",
      }}
    >
      {pages.map((page, index) => (
        <ThumbnailCard
          key={page.filename}
          page={page}
          index={index}
          projectDir={projectDir}
          inputSubdir={inputSubdir}
          onClick={() => onSelectPage(index)}
        />
      ))}
    </div>
  );
}
