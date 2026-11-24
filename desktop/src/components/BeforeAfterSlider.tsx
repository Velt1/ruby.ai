import { useEffect, useRef, useState } from "react";
import {
  Columns2,
  MoveHorizontal,
  Pause,
  Play,
  SplitSquareHorizontal,
  Volume2,
  VolumeX,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

interface BeforeAfterSliderProps {
  beforeUrl?: string | null;
  afterUrl?: string | null;
  beforeVideoUrl?: string | null;
  afterVideoUrl?: string | null;
  beforeLabel?: string;
  afterLabel?: string;
  aspectRatio?: number;
}

export function BeforeAfterSlider({
  beforeUrl,
  afterUrl,
  beforeVideoUrl,
  afterVideoUrl,
  beforeLabel = "Baseline (Lanczos)",
  afterLabel = "ruby.ai Enhanced",
  aspectRatio,
}: BeforeAfterSliderProps) {
  const [videoFailed, setVideoFailed] = useState(false);
  const hasVideo = Boolean(beforeVideoUrl && afterVideoUrl) && !videoFailed;

  const [viewMode, setViewMode] = useState<"split" | "side-by-side">("split");
  const [splitPct, setSplitPct] = useState(50);
  const [zoomOneToOne, setZoomOneToOne] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [stageSize, setStageSize] = useState<{ width: number; height: number } | null>(null);

  const stageRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef(false);
  const beforeVideoRef = useRef<HTMLVideoElement | null>(null);
  const afterVideoRef = useRef<HTMLVideoElement | null>(null);
  const lastTimeRef = useRef(0);

  useEffect(() => {
    setVideoFailed(false);
    setPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    lastTimeRef.current = 0;
  }, [beforeVideoUrl, afterVideoUrl]);

  useEffect(() => {
    const el = stageRef.current;
    if (!el || viewMode !== "split") return;

    const updateSize = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setStageSize({ width: rect.width, height: rect.height });
      }
    };

    updateSize();
    const observer = new ResizeObserver(() => updateSize());
    observer.observe(el);
    return () => observer.disconnect();
  }, [viewMode]);

  const updateFromClientX = (clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSplitPct(Math.max(2, Math.min(98, pct)));
  };

  const togglePlay = () => {
    const b = beforeVideoRef.current;
    const a = afterVideoRef.current;
    if (!b || !a || !hasVideo) return;
    if (!a.paused) {
      b.pause();
      a.pause();
      setPlaying(false);
    } else {
      b.currentTime = a.currentTime;
      void Promise.all([b.play(), a.play()])
        .then(() => setPlaying(true))
        .catch(() => setPlaying(false));
    }
  };

  const togglePlayRef = useRef(togglePlay);
  togglePlayRef.current = togglePlay;

  useEffect(() => {
    if (!playing || !hasVideo) return;
    let rafId = 0;
    const tick = () => {
      const a = afterVideoRef.current;
      const b = beforeVideoRef.current;
      if (a) {
        lastTimeRef.current = a.currentTime;
        setCurrentTime(a.currentTime);
        if (b && Math.abs(b.currentTime - a.currentTime) > 0.04) {
          b.currentTime = a.currentTime;
        }
      }
      rafId = window.requestAnimationFrame(tick);
    };
    rafId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(rafId);
  }, [playing, hasVideo]);

  useEffect(() => {
    if (!hasVideo) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      const target = e.target as HTMLElement | null;
      if (target) {
        const tag = target.tagName;
        if (
          tag === "INPUT" ||
          tag === "TEXTAREA" ||
          tag === "SELECT" ||
          tag === "BUTTON" ||
          target.isContentEditable ||
          target.getAttribute("role") === "slider" ||
          target.getAttribute("role") === "tab"
        ) {
          return;
        }
      }
      e.preventDefault();
      togglePlayRef.current();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [hasVideo]);

  const seekTo = (sec: number) => {
    const b = beforeVideoRef.current;
    const a = afterVideoRef.current;
    if (a) a.currentTime = sec;
    if (b) b.currentTime = sec;
    lastTimeRef.current = sec;
    setCurrentTime(sec);
  };

  const handleViewModeToggle = (nextMode: "split" | "side-by-side") => {
    if (nextMode === viewMode) return;
    beforeVideoRef.current?.pause();
    afterVideoRef.current?.pause();
    setPlaying(false);
    setViewMode(nextMode);
  };

  const handleAfterMetadata = () => {
    const a = afterVideoRef.current;
    const b = beforeVideoRef.current;
    if (!a) return;
    if (Number.isFinite(a.duration) && a.duration > 0) {
      setDuration(a.duration);
    }
    if (lastTimeRef.current > 0) {
      a.currentTime = lastTimeRef.current;
      if (b) b.currentTime = lastTimeRef.current;
    }
  };

  const handleAfterTimeUpdate = () => {
    const a = afterVideoRef.current;
    const b = beforeVideoRef.current;
    if (!a) return;
    lastTimeRef.current = a.currentTime;
    setCurrentTime(a.currentTime);
    if (b && Math.abs(b.currentTime - a.currentTime) > 0.04) {
      b.currentTime = a.currentTime;
    }
  };

  const handleVideoError = () => {
    setPlaying(false);
    setVideoFailed(true);
  };

  const fitMediaClass = "block max-h-full max-w-full object-contain";
  const oneToOneMediaClass = "block max-w-none";

  const splitContainerStyle = (() => {
    if (zoomOneToOne) return undefined;
    if (aspectRatio && aspectRatio > 0 && stageSize && stageSize.width > 0 && stageSize.height > 0) {
      const fitHeight = Math.min(stageSize.height, stageSize.width / aspectRatio);
      const fitWidth = fitHeight * aspectRatio;
      return {
        width: `${Math.floor(fitWidth)}px`,
        height: `${Math.floor(fitHeight)}px`,
      };
    }
    return aspectRatio && aspectRatio > 0 ? { aspectRatio: String(aspectRatio) } : undefined;
  })();

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      {/* Toolbar */}
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 px-1">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>Links: {beforeLabel}</span>
          <span>·</span>
          <span className="font-medium text-foreground">Rechts: {afterLabel}</span>
          {viewMode === "split" && (
            <Badge variant="outline" className="tabular-nums text-[10px] px-1.5 py-0">
              {Math.round(splitPct)}%
            </Badge>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          <div className="inline-flex rounded-md border bg-background/60 p-0.5">
            <Button
              type="button"
              variant={viewMode === "split" ? "secondary" : "ghost"}
              size="sm"
              className="h-6 gap-1 px-2 text-[11px]"
              onClick={() => handleViewModeToggle("split")}
            >
              <SplitSquareHorizontal className="h-3 w-3" />
              Split
            </Button>
            <Button
              type="button"
              variant={viewMode === "side-by-side" ? "secondary" : "ghost"}
              size="sm"
              className="h-6 gap-1 px-2 text-[11px]"
              onClick={() => handleViewModeToggle("side-by-side")}
            >
              <Columns2 className="h-3 w-3" />
              Nebeneinander
            </Button>
          </div>

          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-7 gap-1.5 text-xs"
            onClick={() => setZoomOneToOne((v) => !v)}
          >
            {zoomOneToOne ? (
              <>
                <ZoomOut className="h-3.5 w-3.5" />
                Einpassen
              </>
            ) : (
              <>
                <ZoomIn className="h-3.5 w-3.5" />
                1:1 Pixel
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Main Comparison Canvas */}
      {viewMode === "split" ? (
        <div
          ref={stageRef}
          className={cn(
            "relative flex-1 min-h-0 w-full rounded-lg border bg-black/85 select-none",
            zoomOneToOne
              ? "overflow-auto"
              : "flex items-center justify-center overflow-hidden"
          )}
        >
          <div
            ref={containerRef}
            style={splitContainerStyle}
            onPointerDown={(e) => {
              draggingRef.current = true;
              (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
              updateFromClientX(e.clientX);
            }}
            onPointerMove={(e) => {
              if (!draggingRef.current) return;
              updateFromClientX(e.clientX);
            }}
            onPointerUp={() => {
              draggingRef.current = false;
            }}
            onPointerCancel={() => {
              draggingRef.current = false;
            }}
            className={cn(
              "relative cursor-ew-resize select-none",
              zoomOneToOne
                ? "inline-block"
                : splitContainerStyle && "width" in splitContainerStyle
                  ? "shrink-0"
                  : "inline-flex h-full max-h-full max-w-full items-center justify-center"
            )}
          >
            {hasVideo ? (
              <>
                <video
                  key={`split-after-${afterVideoUrl}`}
                  ref={afterVideoRef}
                  src={afterVideoUrl ?? undefined}
                  poster={afterUrl ?? undefined}
                  preload="auto"
                  playsInline
                  loop
                  muted={muted}
                  onLoadedMetadata={handleAfterMetadata}
                  onTimeUpdate={handleAfterTimeUpdate}
                  onError={handleVideoError}
                  className={cn(
                    "pointer-events-none",
                    zoomOneToOne ? oneToOneMediaClass : "block h-full w-full object-contain"
                  )}
                />
                <video
                  key={`split-before-${beforeVideoUrl}`}
                  ref={beforeVideoRef}
                  src={beforeVideoUrl ?? undefined}
                  poster={beforeUrl ?? undefined}
                  preload="auto"
                  playsInline
                  loop
                  muted
                  onError={handleVideoError}
                  style={{ clipPath: `inset(0 ${100 - splitPct}% 0 0)` }}
                  className={cn(
                    "pointer-events-none absolute inset-0 h-full w-full",
                    zoomOneToOne ? oneToOneMediaClass : "object-contain"
                  )}
                />
              </>
            ) : (
              <>
                <img
                  src={afterUrl ?? ""}
                  alt={afterLabel}
                  draggable={false}
                  className={cn(
                    "pointer-events-none",
                    zoomOneToOne ? oneToOneMediaClass : "block h-full w-full object-contain"
                  )}
                />
                <img
                  src={beforeUrl ?? ""}
                  alt={beforeLabel}
                  draggable={false}
                  style={{ clipPath: `inset(0 ${100 - splitPct}% 0 0)` }}
                  className={cn(
                    "pointer-events-none absolute inset-0 h-full w-full",
                    zoomOneToOne ? oneToOneMediaClass : "object-contain"
                  )}
                />
              </>
            )}

            <div
              style={{ left: `${splitPct}%` }}
              className="pointer-events-none absolute inset-y-0 w-0.5 -translate-x-1/2 bg-white/90 shadow-[0_0_8px_rgba(0,0,0,0.8)]"
            >
              <div className="absolute left-1/2 top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-white/60 bg-background/90 text-foreground shadow-lg">
                <MoveHorizontal className="h-4 w-4" />
              </div>
            </div>

            {hasVideo && !playing && (
              <button
                type="button"
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  togglePlay();
                }}
                className="absolute bottom-4 left-1/2 z-20 flex -translate-x-1/2 items-center gap-2 rounded-full border border-white/25 bg-black/75 px-3.5 py-1.5 text-xs font-medium text-white shadow-lg backdrop-blur transition hover:bg-black/90"
              >
                <Play className="h-3.5 w-3.5 fill-white" />
                Live-Vergleich abspielen
              </button>
            )}

            <Badge
              variant="secondary"
              className="pointer-events-none absolute left-2.5 top-2.5 bg-black/75 text-[11px] text-white backdrop-blur"
            >
              {beforeLabel}
            </Badge>
            <Badge
              variant="default"
              className="pointer-events-none absolute right-2.5 top-2.5 text-[11px] shadow"
            >
              {afterLabel}
            </Badge>
          </div>
        </div>
      ) : (
        <div
          className={cn(
            "grid flex-1 min-h-0 w-full grid-cols-2 gap-2 rounded-lg border bg-black/85 p-2",
            zoomOneToOne && "overflow-auto"
          )}
        >
          <div
            className={cn(
              "relative flex min-h-0 min-w-0 items-center justify-center rounded-md bg-black/50",
              zoomOneToOne ? "overflow-auto" : "overflow-hidden"
            )}
          >
            {hasVideo ? (
              <video
                key={`sbs-before-${beforeVideoUrl}`}
                ref={beforeVideoRef}
                src={beforeVideoUrl ?? undefined}
                poster={beforeUrl ?? undefined}
                preload="auto"
                playsInline
                loop
                muted
                onError={handleVideoError}
                className={zoomOneToOne ? oneToOneMediaClass : fitMediaClass}
              />
            ) : (
              <img
                src={beforeUrl ?? ""}
                alt={beforeLabel}
                draggable={false}
                className={zoomOneToOne ? oneToOneMediaClass : fitMediaClass}
              />
            )}
            <Badge
              variant="secondary"
              className="pointer-events-none absolute left-2.5 top-2.5 bg-black/75 text-[11px] text-white backdrop-blur"
            >
              {beforeLabel}
            </Badge>
          </div>

          <div
            className={cn(
              "relative flex min-h-0 min-w-0 items-center justify-center rounded-md bg-black/50",
              zoomOneToOne ? "overflow-auto" : "overflow-hidden"
            )}
          >
            {hasVideo ? (
              <video
                key={`sbs-after-${afterVideoUrl}`}
                ref={afterVideoRef}
                src={afterVideoUrl ?? undefined}
                poster={afterUrl ?? undefined}
                preload="auto"
                playsInline
                loop
                muted={muted}
                onLoadedMetadata={handleAfterMetadata}
                onTimeUpdate={handleAfterTimeUpdate}
                onError={handleVideoError}
                className={zoomOneToOne ? oneToOneMediaClass : fitMediaClass}
              />
            ) : (
              <img
                src={afterUrl ?? ""}
                alt={afterLabel}
                draggable={false}
                className={zoomOneToOne ? oneToOneMediaClass : fitMediaClass}
              />
            )}
            <Badge
              variant="default"
              className="pointer-events-none absolute left-2.5 top-2.5 text-[11px] shadow"
            >
              {afterLabel}
            </Badge>
          </div>
        </div>
      )}

      {/* Synchronized Transport Controls */}
      {hasVideo && (
        <div className="flex shrink-0 items-center gap-3 rounded-lg border bg-card/70 px-3 py-1.5">
          <Button
            type="button"
            variant="secondary"
            size="sm"
            className="h-7 gap-1.5 px-2.5 text-xs"
            onClick={togglePlay}
          >
            {playing ? (
              <>
                <Pause className="h-3.5 w-3.5" />
                Pause
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5" />
                Abspielen
              </>
            )}
          </Button>

          <Slider
            value={[currentTime]}
            min={0}
            max={Math.max(0.1, duration)}
            step={0.02}
            onValueChange={([t]) => seekTo(t)}
            className="flex-1"
          />

          <span className="w-24 text-right text-[11px] tabular-nums text-muted-foreground">
            {currentTime.toFixed(2)}s / {duration.toFixed(2)}s
          </span>

          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            title={muted ? "Ton einschalten" : "Ton stummschalten"}
            onClick={() => setMuted((m) => !m)}
          >
            {muted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
          </Button>
        </div>
      )}
    </div>
  );
}
