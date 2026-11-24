import { useEffect, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  FolderOpen,
  Info,
  Loader2,
  Square,
  Video,
  Wand2,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BeforeAfterSlider } from "@/components/BeforeAfterSlider";
import { VideoDropzone } from "@/components/VideoDropzone";
import { engineClient } from "@/lib/engine-client";
import type { JobRecord, VideoAnalysis, VideoProbe } from "@/lib/types";
import { formatBitrate, formatBytes, formatDuration } from "@/lib/utils";

interface PreviewStageProps {
  probe: VideoProbe | null;
  probing: boolean;
  analysis: VideoAnalysis | null;
  analyzing: boolean;
  currentPreset: string;
  probeError: string | null;
  activeJob: JobRecord | null;
  isDragging: boolean;
  onPickVideo: () => void;
  onApplyAutoEstimate: () => void;
  onCancelJob: (jobId: string) => void;
  onRevealPath: (path: string) => void;
}

export function PreviewStage({
  probe,
  probing,
  analysis,
  analyzing,
  currentPreset,
  probeError,
  activeJob,
  isDragging,
  onPickVideo,
  onApplyAutoEstimate,
  onCancelJob,
  onRevealPath,
}: PreviewStageProps) {
  const [resultTab, setResultTab] = useState("compare");
  const [compareSource, setCompareSource] = useState<"social" | "master">("social");

  useEffect(() => {
    if (activeJob?.status === "completed") {
      setResultTab("compare");
    }
  }, [activeJob?.job_id, activeJob?.status]);

  if (probing) {
    return (
      <div className="flex h-full min-h-[380px] flex-col items-center justify-center rounded-xl border bg-card/30 p-8">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <p className="mt-3 text-sm font-medium">Video wird analysiert…</p>
        <p className="mt-1 text-xs text-muted-foreground">
          Auflösung, Farbraum, Helligkeit und optimale Social-Einstellungen werden ermittelt.
        </p>
      </div>
    );
  }

  if (!probe) {
    return (
      <div className="flex h-full flex-col gap-3">
        {probeError && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Video konnte nicht geladen werden</AlertTitle>
            <AlertDescription>{probeError}</AlertDescription>
          </Alert>
        )}
        <div className="flex-1">
          <VideoDropzone isDragging={isDragging} onPickVideo={onPickVideo} />
        </div>
      </div>
    );
  }

  const posterUrl = engineClient.toMediaUrl(probe.poster_path);
  const enhanceRes = activeJob?.status === "completed" ? activeJob.enhance_result : undefined;
  const benchRes = activeJob?.status === "completed" ? activeJob.benchmark_result : undefined;

  const beforePngUrl = engineClient.toMediaUrl(
    enhanceRes?.before_frame_png ?? benchRes?.before_frame_png ?? null
  );
  const afterPngUrl = engineClient.toMediaUrl(
    enhanceRes?.after_frame_png ?? benchRes?.after_frame_png ?? null
  );

  const baselineSocialUrl = engineClient.toMediaUrl(
    benchRes?.artifacts?.baseline_social?.path ?? null
  );
  const enhancedSocialUrl = engineClient.toMediaUrl(
    benchRes?.artifacts?.enhanced_social?.path ?? null
  );
  const baselineMasterUrl = engineClient.toMediaUrl(
    benchRes?.artifacts?.baseline?.path ?? null
  );
  const enhancedMasterUrl = engineClient.toMediaUrl(
    benchRes?.artifacts?.enhanced?.path ?? null
  );

  const enhancedVideoPath =
    enhanceRes?.output_path ?? benchRes?.artifacts?.enhanced?.path ?? null;
  const enhancedVideoUrl = engineClient.toMediaUrl(enhancedVideoPath);
  const enhanceBeforeVideoUrl = engineClient.toMediaUrl(
    enhanceRes?.before_video_path ?? null
  );

  const hasMasterPair = Boolean(baselineMasterUrl && enhancedMasterUrl);
  const hasSocialPair = Boolean(baselineSocialUrl && enhancedSocialUrl);
  const activeBeforeVideoUrl = enhanceRes
    ? enhanceBeforeVideoUrl
    : compareSource === "master" && hasMasterPair
      ? baselineMasterUrl
      : baselineSocialUrl;
  const activeAfterVideoUrl = enhanceRes
    ? enhancedVideoUrl
    : compareSource === "master" && hasMasterPair
      ? enhancedMasterUrl
      : enhancedSocialUrl;

  const targetW =
    enhanceRes?.target_width ?? benchRes?.enhance_result.target_width ?? probe.target_width;
  const targetH =
    enhanceRes?.target_height ?? benchRes?.enhance_result.target_height ?? probe.target_height;
  const compareAspectRatio = targetW > 0 && targetH > 0 ? targetW / targetH : undefined;

  const comparisonSheetPath = benchRes?.artifacts?.comparison_png?.path ?? null;
  const revealTarget = enhancedVideoPath ?? benchRes?.output_dir ?? probe.path;

  const hasCompletedResult = Boolean(enhanceRes || benchRes);
  const hasCompareVideos = Boolean(activeBeforeVideoUrl && activeAfterVideoUrl);
  const hasCompareMedia = Boolean((beforePngUrl && afterPngUrl) || hasCompareVideos);
  const effectiveTab =
    resultTab === "compare" && !hasCompareMedia
      ? enhancedVideoUrl
        ? "enhanced"
        : "source"
      : resultTab;

  const showAnalysisBanner =
    analyzing ||
    (analysis !== null &&
      (analysis.recommended_preset !== currentPreset ||
        analysis.is_hdr ||
        analysis.is_high_fps));

  return (
    <div className="flex h-full min-h-0 flex-col gap-2.5">
      {/* Input metadata bar */}
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 rounded-lg border bg-card/70 px-3 py-1.5">
        <div className="flex flex-wrap items-center gap-2 min-w-0">
          <Video className="h-4 w-4 shrink-0 text-primary" />
          <span className="max-w-[260px] truncate text-xs font-semibold" title={probe.path}>
            {probe.filename}
          </span>
          <Badge variant="secondary" className="tabular-nums text-[11px] font-normal">
            {probe.display_width}×{probe.display_height} · {probe.codec_name.toUpperCase()} ·{" "}
            {probe.fps} fps
          </Badge>
          {probe.is_hdr && (
            <Badge variant="warning" className="text-[10px]">
              HDR → BT.709
            </Badge>
          )}
          <ArrowRight className="h-3.5 w-3.5 text-muted-foreground" />
          <Badge variant="outline" className="tabular-nums text-[11px] font-normal">
            {probe.target_width}×{probe.target_height} H.264
          </Badge>
          <span className="text-[11px] text-muted-foreground tabular-nums">
            {formatDuration(probe.duration)} · {formatBytes(probe.file_size_bytes)} ·{" "}
            {formatBitrate(probe.total_bitrate_kbps)}
          </span>
        </div>

        <Button
          type="button"
          variant="outline"
          size="sm"
          className="h-7 gap-1.5 text-xs"
          onClick={onPickVideo}
        >
          <FolderOpen className="h-3.5 w-3.5" />
          Anderes Video
        </Button>
      </div>

      {/* Compact Auto-Analysis Banner */}
      {showAnalysisBanner && (
        <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 rounded-lg border border-border/80 bg-card/40 px-3 py-1.5 text-xs">
          {analyzing ? (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
              <span>Auto-Analyse prüft Helligkeit & Kompressionsdichte…</span>
            </div>
          ) : analysis ? (
            <>
              <div className="flex flex-wrap items-center gap-2 min-w-0">
                <Wand2 className="h-3.5 w-3.5 shrink-0 text-primary" />
                {analysis.is_low_light && (
                  <Badge variant="warning" className="text-[10px]">
                    Nacht / Low-Light
                  </Badge>
                )}
                {analysis.is_heavily_compressed && (
                  <Badge variant="warning" className="text-[10px]">
                    Stark komprimiert
                  </Badge>
                )}
                {analysis.is_high_fps && (
                  <Badge variant="secondary" className="text-[10px]">
                    High-FPS
                  </Badge>
                )}
                <span className="text-muted-foreground truncate">
                  {analysis.reasons[0]}
                </span>
              </div>
              {analysis.recommended_preset !== currentPreset && (
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  className="h-6 px-2.5 text-[11px]"
                  onClick={onApplyAutoEstimate}
                >
                  Empfehlung ({analysis.recommended_preset}) anwenden
                </Button>
              )}
            </>
          ) : null}
        </div>
      )}

      {probeError && (
        <Alert variant="destructive" className="shrink-0">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Fehler</AlertTitle>
          <AlertDescription>{probeError}</AlertDescription>
        </Alert>
      )}

      {/* Running progress bar */}
      {activeJob?.status === "running" && (
        <div className="shrink-0 rounded-lg border border-primary/30 bg-card/80 p-3">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
              <span className="font-medium">
                {activeJob.mode === "preview"
                  ? "2s Social-Vorschau"
                  : activeJob.mode === "benchmark"
                    ? "Benchmark A/B"
                    : "Video-Export"}{" "}
                läuft:{" "}
                <span className="text-muted-foreground">{activeJob.preset}</span>
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-muted-foreground tabular-nums">
                {activeJob.frames_done}
                {activeJob.total_frames ? ` / ${activeJob.total_frames}` : ""} Frames ·{" "}
                {activeJob.fps.toFixed(2)} fps · {activeJob.elapsed_sec.toFixed(1)}s
                {activeJob.eta_sec != null ? ` · ETA ${activeJob.eta_sec.toFixed(0)}s` : ""}
              </span>
              <Button
                type="button"
                variant="destructive"
                size="sm"
                className="h-7 gap-1 text-xs"
                onClick={() => onCancelJob(activeJob.job_id)}
              >
                <Square className="h-3 w-3" />
                Abbrechen
              </Button>
            </div>
          </div>
          <Progress value={activeJob.percent} className="mt-2.5 h-2" />
        </div>
      )}

      {activeJob?.status === "failed" && (
        <Alert variant="destructive" className="shrink-0">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Job fehlgeschlagen</AlertTitle>
          <AlertDescription>{activeJob.error_message}</AlertDescription>
        </Alert>
      )}

      {/* Main viewport */}
      <div className="relative flex-1 min-h-0 overflow-hidden rounded-xl border bg-card/30 p-2.5">
        {hasCompletedResult ? (
          <Tabs
            value={effectiveTab}
            onValueChange={setResultTab}
            className="flex h-full min-h-0 flex-col"
          >
            <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 pb-2">
              <div className="flex flex-wrap items-center gap-2">
                <TabsList className="h-8">
                  {hasCompareMedia && (
                    <TabsTrigger value="compare" className="text-xs px-3">
                      {hasCompareVideos ? "Vorher / Nachher (Video)" : "Split-Vergleich"}
                    </TabsTrigger>
                  )}
                  {enhancedVideoUrl && (
                    <TabsTrigger value="enhanced" className="text-xs px-3">
                      {benchRes ? "Vorschau Master MP4" : "Export MP4"}
                    </TabsTrigger>
                  )}
                  {enhancedSocialUrl && (
                    <TabsTrigger value="social" className="text-xs px-3">
                      Simuliert ({benchRes?.bitrate_kbps ?? 3000} kbps)
                    </TabsTrigger>
                  )}
                  <TabsTrigger value="source" className="text-xs px-3">
                    Quell-Vorschau
                  </TabsTrigger>
                </TabsList>

                {effectiveTab === "compare" && hasSocialPair && hasMasterPair && (
                  <div className="inline-flex rounded-md border bg-background/60 p-0.5">
                    <Button
                      type="button"
                      variant={compareSource === "social" ? "secondary" : "ghost"}
                      size="sm"
                      className="h-6 px-2 text-[11px]"
                      onClick={() => setCompareSource("social")}
                    >
                      Nach IG/TikTok ({benchRes?.bitrate_kbps ?? 3000} kbps)
                    </Button>
                    <Button
                      type="button"
                      variant={compareSource === "master" ? "secondary" : "ghost"}
                      size="sm"
                      className="h-6 px-2 text-[11px]"
                      onClick={() => setCompareSource("master")}
                    >
                      Master (vor Kompression)
                    </Button>
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2">
                {comparisonSheetPath && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => onRevealPath(comparisonSheetPath)}
                  >
                    Vergleichs-PNG zeigen
                  </Button>
                )}
                {revealTarget && (
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    className="h-7 gap-1.5 text-xs"
                    onClick={() => onRevealPath(revealTarget)}
                  >
                    <FolderOpen className="h-3.5 w-3.5" />
                    Im Explorer öffnen
                  </Button>
                )}
              </div>
            </div>

            {hasCompareMedia && (
              <TabsContent
                value="compare"
                className="mt-0 flex-1 min-h-0 data-[state=active]:flex data-[state=active]:flex-col"
              >
                <BeforeAfterSlider
                  beforeUrl={beforePngUrl}
                  afterUrl={afterPngUrl}
                  beforeVideoUrl={activeBeforeVideoUrl}
                  afterVideoUrl={activeAfterVideoUrl}
                  aspectRatio={compareAspectRatio}
                  beforeLabel={
                    benchRes
                      ? compareSource === "master"
                        ? "Ohne ruby.ai (Master)"
                        : "Ohne ruby.ai (nach IG/TikTok)"
                      : "Original (Lanczos)"
                  }
                  afterLabel={
                    benchRes
                      ? compareSource === "master"
                        ? "ruby.ai (Master)"
                        : "ruby.ai (nach IG/TikTok)"
                      : "ruby.ai Export"
                  }
                />
              </TabsContent>
            )}

            {enhancedVideoUrl && (
              <TabsContent
                value="enhanced"
                className="mt-0 flex-1 min-h-0 items-center justify-center rounded-lg bg-black/80 p-2 data-[state=active]:flex"
              >
                <video
                  key={enhancedVideoUrl}
                  src={enhancedVideoUrl}
                  controls
                  className="h-full w-full rounded object-contain"
                />
              </TabsContent>
            )}

            {enhancedSocialUrl && (
              <TabsContent
                value="social"
                className="mt-0 flex-1 min-h-0 items-center justify-center rounded-lg bg-black/80 p-2 data-[state=active]:flex"
              >
                <video
                  key={enhancedSocialUrl}
                  src={enhancedSocialUrl}
                  controls
                  className="h-full w-full rounded object-contain"
                />
              </TabsContent>
            )}

            <TabsContent
              value="source"
              className="mt-0 flex-1 min-h-0 items-center justify-center rounded-lg bg-black/80 p-2 data-[state=active]:flex"
            >
              {posterUrl ? (
                <img
                  src={posterUrl}
                  alt={probe.filename}
                  className="h-full w-full rounded object-contain"
                />
              ) : (
                <p className="text-xs text-muted-foreground">Kein Poster-Frame verfügbar.</p>
              )}
            </TabsContent>
          </Tabs>
        ) : (
          <div className="flex h-full w-full items-center justify-center rounded-lg bg-black/75 p-2">
            {posterUrl ? (
              <img
                src={posterUrl}
                alt={probe.filename}
                className="h-full w-full rounded object-contain"
              />
            ) : (
              <p className="text-xs text-muted-foreground">Kein Vorschaubild verfügbar.</p>
            )}
          </div>
        )}
      </div>

      {/* Metrics summary + Social Upload Tip */}
      {hasCompletedResult && (
        <div className="shrink-0 space-y-1">
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border bg-card/70 px-3 py-1.5 text-xs">
            {(() => {
              const core = enhanceRes ?? benchRes?.enhance_result;
              if (!core) return null;
              const sizeBytes =
                enhanceRes?.file_size_bytes ??
                benchRes?.artifacts?.enhanced?.size_bytes ??
                0;
              const totalKbps =
                enhanceRes?.total_bitrate_kbps ??
                benchRes?.artifacts?.enhanced?.total_bitrate_kbps ??
                0;
              const tileLabel =
                core.effective_tile_size === 0
                  ? "full-frame"
                  : core.effective_tile_size != null
                    ? `${core.effective_tile_size}px`
                    : "auto";
              return (
                <>
                  <div className="flex flex-wrap items-center gap-3 tabular-nums">
                    <Badge variant="success">{core.fps.toFixed(2)} fps</Badge>
                    <span>
                      <span className="text-muted-foreground">Laufzeit:</span>{" "}
                      {core.elapsed_sec.toFixed(2)}s ({core.frames_processed} Frames)
                    </span>
                    <span>
                      <span className="text-muted-foreground">Profil/Modell:</span> {core.model_key}
                    </span>
                    <span>
                      <span className="text-muted-foreground">Codec:</span> {core.codec_used}
                    </span>
                    <span>
                      <span className="text-muted-foreground">Tiling:</span> {tileLabel} · Batch{" "}
                      {core.batch_size}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 tabular-nums text-muted-foreground">
                    <span>{formatBytes(sizeBytes)}</span>
                    <span>{formatBitrate(totalKbps)}</span>
                  </div>
                </>
              );
            })()}
          </div>
          {enhanceRes && (
            <div className="flex items-center gap-2 px-1 text-[11px] text-muted-foreground">
              <Info className="h-3.5 w-3.5 shrink-0 text-primary" />
              <span>
                Upload-Tipp: Exportiert als TikTok/Instagram-Ingest (Rec.709, Closed GOP, 48 kHz AAC).
                In der App trotzdem <strong className="text-foreground">„Upload in höchster Qualität“</strong> aktivieren.
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
