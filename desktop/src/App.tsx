import { useCallback, useEffect, useMemo, useState } from "react";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { AlertCircle } from "lucide-react";
import { HeaderBar } from "@/components/HeaderBar";
import { JobQueueBar } from "@/components/JobQueueBar";
import { ModelsDoctorView } from "@/components/ModelsDoctorView";
import { PreviewStage } from "@/components/PreviewStage";
import { RuntimeSetupOverlay } from "@/components/RuntimeSetupOverlay";
import { SettingsPanel } from "@/components/SettingsPanel";
import { UpdateNotification } from "@/components/UpdateNotification";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { TooltipProvider } from "@/components/ui/tooltip";
import { engineClient } from "@/lib/engine-client";
import { useAppUpdater } from "@/lib/use-updater";
import type {
  BenchmarkJobResult,
  EngineInfo,
  EnhanceJobResult,
  JobEvent,
  JobMode,
  JobRecord,
  PipelineSettings,
  RuntimeProgressEvent,
  RuntimeStatus,
  VideoAnalysis,
  VideoProbe,
} from "@/lib/types";

const DEFAULT_SETTINGS: PipelineSettings = {
  preset: "social-crisp",
  model: "de-h264",
  export_target: "social-1080p",
  max_short_edge: 1080,
  temporal_denoise: 0.03,
  ai_strength: 0.7,
  sharpen: 0.42,
  clarity: 0.22,
  deband: true,
  grain: 2,
  hdr_to_sdr: true,
  maxrate_kbps: null,
  crf: 16,
  codec: "auto",
  x264_preset: "slow",
  nvenc_preset: "p6",
  hwaccel: "auto",
  tile_size: null,
  tile_overlap: 32,
  tile_batch_size: 4,
  batch_size: 1,
  optimize_gpu: true,
  compile_model: false,
  async_io: true,
  start_sec: null,
  duration_sec: null,
  bitrate_kbps: 3000,
  reference_path: null,
  output_path: null,
};

export function App() {
  const {
    appVersion,
    updateState,
    showNotification,
    checkForUpdates,
    installUpdate,
    dismissNotification,
    reopenNotification,
  } = useAppUpdater({ enabled: true });

  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus | null>(null);
  const [runtimeSettingUp, setRuntimeSettingUp] = useState(false);
  const [runtimeProgress, setRuntimeProgress] = useState<RuntimeProgressEvent | null>(null);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);

  const [engineInfo, setEngineInfo] = useState<EngineInfo | null>(null);
  const [loadingInfo, setLoadingInfo] = useState(true);
  const [infoError, setInfoError] = useState<string | null>(null);

  const [activeView, setActiveView] = useState<"studio" | "system">("studio");
  const [settings, setSettings] = useState<PipelineSettings>(DEFAULT_SETTINGS);

  const [probe, setProbe] = useState<VideoProbe | null>(null);
  const [probing, setProbing] = useState(false);
  const [analysis, setAnalysis] = useState<VideoAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [probeError, setProbeError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const loadEngineInfo = useCallback(async () => {
    setLoadingInfo(true);
    setInfoError(null);
    try {
      const info = await engineClient.getEngineInfo();
      setEngineInfo(info);
      setSettings((prev) => ({
        ...prev,
        hwaccel: info.cuda_available ? prev.hwaccel : "none",
      }));
    } catch (err) {
      setInfoError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingInfo(false);
    }
  }, []);

  const runRuntimeBootstrap = useCallback(
    async (forceReinstall = false) => {
      setRuntimeSettingUp(true);
      setRuntimeError(null);
      try {
        const next = await engineClient.ensureRuntimeReady(forceReinstall);
        setRuntimeStatus(next);
        if (next.ready) {
          await loadEngineInfo();
        }
      } catch (err) {
        setRuntimeError(err instanceof Error ? err.message : String(err));
      } finally {
        setRuntimeSettingUp(false);
      }
    },
    [loadEngineInfo]
  );

  // Initialize runtime check & bootstrap if needed
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let mounted = true;

    void engineClient
      .subscribeRuntimeEvents((ev) => {
        if (mounted) setRuntimeProgress(ev);
      })
      .then((fn) => {
        if (mounted) {
          unlisten = fn;
        } else {
          fn();
        }
      })
      .catch(() => {});

    void (async () => {
      try {
        const status = await engineClient.getRuntimeStatus();
        if (!mounted) return;
        setRuntimeStatus(status);
        if (status.ready) {
          await loadEngineInfo();
        } else {
          await runRuntimeBootstrap(false);
        }
      } catch {
        // Fallback for non-runtime contexts: try loading engine info directly.
        if (mounted) await loadEngineInfo();
      }
    })();

    return () => {
      mounted = false;
      unlisten?.();
    };
  }, [loadEngineInfo, runRuntimeBootstrap]);

  const handleLoadVideo = useCallback(
    async (filePath: string) => {
      setProbing(true);
      setAnalyzing(true);
      setProbeError(null);
      setAnalysis(null);
      try {
        const probeResult = await engineClient.probeVideo(
          filePath,
          settings.max_short_edge
        );
        setProbe(probeResult);
        setActiveView("studio");
        setProbing(false);

        try {
          const analysisResult = await engineClient.analyzeVideo(filePath);
          setAnalysis(analysisResult);
        } catch {
          setAnalysis(null);
        } finally {
          setAnalyzing(false);
        }
      } catch (err) {
        setProbeError(err instanceof Error ? err.message : String(err));
        setProbing(false);
        setAnalyzing(false);
      }
    },
    [settings.max_short_edge]
  );

  // Subscribe to native Tauri drag-and-drop
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let mounted = true;
    void getCurrentWebview()
      .onDragDropEvent((event) => {
        if (!mounted) return;
        const payload = event.payload;
        if (payload.type === "enter" || payload.type === "over") {
          setIsDragging(true);
        } else if (payload.type === "leave") {
          setIsDragging(false);
        } else if (payload.type === "drop") {
          setIsDragging(false);
          const first = payload.paths[0];
          if (first) {
            void handleLoadVideo(first);
          }
        }
      })
      .then((fn) => {
        if (mounted) {
          unlisten = fn;
        } else {
          fn();
        }
      })
      .catch(() => {});

    return () => {
      mounted = false;
      unlisten?.();
    };
  }, [handleLoadVideo]);

  // Subscribe to Python NDJSON worker events from Tauri
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let mounted = true;

    const handleEvent = (ev: JobEvent) => {
      if (!ev.job_id) return;
      setJobs((prev) =>
        prev.map((job) => {
          if (job.job_id !== ev.job_id) return job;
          if (ev.type === "started") {
            return { ...job, status: "running" };
          }
          if (ev.type === "progress") {
            return {
              ...job,
              status: "running",
              frames_done: ev.frames_done,
              total_frames: ev.total_frames,
              percent: ev.percent ?? job.percent,
              fps: ev.fps,
              elapsed_sec: ev.elapsed_sec,
              eta_sec: ev.eta_sec,
            };
          }
          if (ev.type === "completed") {
            if (ev.mode === "enhance") {
              const res = ev.result as EnhanceJobResult;
              return {
                ...job,
                status: "completed",
                percent: 100,
                frames_done: res.frames_processed,
                total_frames: res.frames_processed,
                fps: res.fps,
                elapsed_sec: res.elapsed_sec,
                eta_sec: 0,
                enhance_result: res,
              };
            }
            const res = ev.result as BenchmarkJobResult;
            return {
              ...job,
              status: "completed",
              percent: 100,
              frames_done: res.enhance_result.frames_processed,
              total_frames: res.enhance_result.frames_processed,
              fps: res.enhance_result.fps,
              elapsed_sec: res.enhance_result.elapsed_sec,
              eta_sec: 0,
              benchmark_result: res,
            };
          }
          if (ev.type === "cancelled") {
            return { ...job, status: "cancelled" };
          }
          if (ev.type === "error") {
            return {
              ...job,
              status: "failed",
              error_message: ev.message,
            };
          }
          return job;
        })
      );
    };

    void engineClient
      .subscribeJobEvents(handleEvent)
      .then((fn) => {
        if (mounted) {
          unlisten = fn;
        } else {
          fn();
        }
      });

    return () => {
      mounted = false;
      unlisten?.();
    };
  }, []);

  const activeJob = useMemo(
    () => jobs.find((j) => j.job_id === activeJobId) ?? null,
    [jobs, activeJobId]
  );

  const runningJob = useMemo(
    () => jobs.find((j) => j.status === "running") ?? null,
    [jobs]
  );

  const handleSelectPreset = (presetKey: string) => {
    const found = engineInfo?.presets.find((p) => p.key === presetKey);
    if (!found) return;
    setSettings((prev) => ({
      ...prev,
      preset: found.key,
      model: found.model,
      temporal_denoise: found.temporal_denoise,
      ai_strength: found.ai_strength,
      sharpen: found.sharpen,
      clarity: found.clarity,
      deband: found.deband,
      grain: found.grain,
      batch_size: found.key === "fast" ? 2 : 1,
    }));
  };

  const handleSelectExportTarget = (targetKey: string) => {
    const target = engineInfo?.export_targets.find((t) => t.key === targetKey);
    if (!target) return;
    setSettings((prev) => ({
      ...prev,
      export_target: target.key,
      max_short_edge: target.max_short_edge,
    }));
    if (probe) {
      void engineClient
        .probeVideo(probe.path, target.max_short_edge)
        .then(setProbe)
        .catch(() => {});
    }
  };

  const handleApplyAutoEstimate = () => {
    if (!analysis) return;
    const ov = analysis.recommended_overrides;
    setSettings((prev) => ({
      ...prev,
      preset: analysis.recommended_preset,
      export_target: analysis.recommended_export_target,
      max_short_edge: 1080,
      model: typeof ov.model === "string" ? ov.model : prev.model,
      ai_strength:
        typeof ov.ai_strength === "number" ? ov.ai_strength : prev.ai_strength,
      temporal_denoise:
        typeof ov.temporal_denoise === "number"
          ? ov.temporal_denoise
          : prev.temporal_denoise,
      sharpen: typeof ov.sharpen === "number" ? ov.sharpen : prev.sharpen,
      clarity: typeof ov.clarity === "number" ? ov.clarity : prev.clarity,
      deband: typeof ov.deband === "boolean" ? ov.deband : prev.deband,
      grain: typeof ov.grain === "number" ? ov.grain : prev.grain,
      batch_size: analysis.recommended_preset === "fast" ? 2 : 1,
    }));
  };

  const handleUpdateSettings = (patch: Partial<PipelineSettings>) => {
    setSettings((prev) => ({ ...prev, ...patch }));
  };

  const handlePickVideo = async () => {
    const selected = await engineClient.pickVideoFile();
    if (selected) {
      await handleLoadVideo(selected);
    }
  };

  const handlePickOutput = async () => {
    const defaultName = probe
      ? `${probe.filename.replace(/\.[^.]+$/, "")}_ruby_${settings.preset}.mp4`
      : undefined;
    const file = await engineClient.pickOutputFile(defaultName);
    if (file) handleUpdateSettings({ output_path: file });
  };

  const handlePickReference = async () => {
    const selected = await engineClient.pickVideoFile();
    if (selected) handleUpdateSettings({ reference_path: selected });
  };

  const launchJob = async (mode: JobMode) => {
    if (!probe || runningJob) return;
    setProbeError(null);

    const payload = {
      input_path: probe.path,
      output_path: mode === "enhance" ? settings.output_path : undefined,
      preset: settings.preset,
      start_sec: settings.start_sec,
      duration_sec:
        mode === "preview"
          ? (settings.duration_sec ?? Math.min(2.0, Math.max(0.5, probe.duration || 2.0)))
          : settings.duration_sec,
      bitrate_kbps: settings.bitrate_kbps,
      reference_path: settings.reference_path,
      config: {
        model: settings.model,
        export_target: settings.export_target,
        max_short_edge: settings.max_short_edge,
        temporal_denoise: settings.temporal_denoise,
        ai_strength: settings.ai_strength,
        sharpen: settings.sharpen,
        clarity: settings.clarity,
        deband: settings.deband,
        grain: settings.grain,
        hdr_to_sdr: settings.hdr_to_sdr,
        maxrate_kbps: settings.maxrate_kbps,
        crf: settings.crf,
        codec: settings.codec,
        x264_preset: settings.x264_preset,
        nvenc_preset: settings.nvenc_preset,
        hwaccel: settings.hwaccel,
        tile_size: settings.tile_size,
        tile_overlap: settings.tile_overlap,
        tile_batch_size: settings.tile_batch_size,
        batch_size: settings.batch_size,
        optimize_gpu: settings.optimize_gpu,
        compile_model: settings.compile_model,
        async_io: settings.async_io,
      },
    };

    try {
      const { job_id } = await engineClient.startJob(mode, payload);
      const record: JobRecord = {
        job_id,
        mode,
        input_path: probe.path,
        filename: probe.filename,
        preset: settings.preset,
        model: settings.model,
        status: "running",
        frames_done: 0,
        total_frames: null,
        percent: 0,
        fps: 0,
        elapsed_sec: 0,
        eta_sec: null,
      };
      setJobs((prev) => [record, ...prev]);
      setActiveJobId(job_id);
    } catch (err) {
      setProbeError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleCancelJob = async (jobId: string) => {
    await engineClient.cancelJob(jobId);
  };

  const handleRevealPath = async (targetPath: string) => {
    await engineClient.revealInExplorer(targetPath);
  };

  const needsSetupOverlay = runtimeStatus !== null && !runtimeStatus.ready;

  return (
    <TooltipProvider>
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => e.preventDefault()}
        className="flex h-screen w-screen flex-col overflow-hidden bg-background text-foreground"
      >
        <HeaderBar
          appVersion={appVersion}
          engineInfo={engineInfo}
          loadingInfo={loadingInfo}
          updateState={updateState}
          activeView={activeView}
          onActiveViewChange={setActiveView}
          onRefreshInfo={() => void loadEngineInfo()}
          onOpenUpdate={reopenNotification}
        />

        {needsSetupOverlay ? (
          <RuntimeSetupOverlay
            appVersion={appVersion}
            status={runtimeStatus}
            progress={runtimeProgress}
            settingUp={runtimeSettingUp}
            error={runtimeError}
            onRetry={() => void runRuntimeBootstrap(false)}
          />
        ) : (
          <>
            {infoError && (
              <div className="px-4 pt-3">
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Python-Engine nicht erreichbar</AlertTitle>
                  <AlertDescription>{infoError}</AlertDescription>
                </Alert>
              </div>
            )}

            {activeView === "system" ? (
              <ModelsDoctorView
                appVersion={appVersion}
                engineInfo={engineInfo}
                loadingInfo={loadingInfo}
                runtimeStatus={runtimeStatus}
                updateState={updateState}
                isJobRunning={Boolean(runningJob)}
                onRefreshInfo={() => void loadEngineInfo()}
                onCheckUpdates={() => void checkForUpdates(false)}
                onInstallUpdate={() => {
                  if (!runningJob) void installUpdate();
                }}
                onRepairRuntime={() => {
                  setRuntimeStatus((prev) => (prev ? { ...prev, ready: false } : prev));
                  void runRuntimeBootstrap(true);
                }}
              />
            ) : (
              <div className="flex min-h-0 flex-1 overflow-hidden">
                <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
                  <main className="flex-1 min-h-0 overflow-hidden p-4">
                    <PreviewStage
                      probe={probe}
                      probing={probing}
                      analysis={analysis}
                      analyzing={analyzing}
                      currentPreset={settings.preset}
                      probeError={probeError}
                      activeJob={activeJob}
                      isDragging={isDragging}
                      onPickVideo={() => void handlePickVideo()}
                      onApplyAutoEstimate={handleApplyAutoEstimate}
                      onCancelJob={(id) => void handleCancelJob(id)}
                      onRevealPath={(p) => void handleRevealPath(p)}
                    />
                  </main>
                  <JobQueueBar
                    jobs={jobs}
                    activeJobId={activeJobId}
                    onSelectJob={setActiveJobId}
                    onRevealPath={(p) => void handleRevealPath(p)}
                  />
                </div>

                <SettingsPanel
                  engineInfo={engineInfo}
                  probe={probe}
                  analysis={analysis}
                  settings={settings}
                  isRunning={Boolean(runningJob)}
                  onSelectPreset={handleSelectPreset}
                  onSelectExportTarget={handleSelectExportTarget}
                  onApplyAutoEstimate={handleApplyAutoEstimate}
                  onUpdateSettings={handleUpdateSettings}
                  onPickOutput={() => void handlePickOutput()}
                  onPickReference={() => void handlePickReference()}
                  onStartPreview={() => void launchJob("preview")}
                  onStartExport={() => void launchJob("enhance")}
                  onCancel={() => {
                    if (runningJob) void handleCancelJob(runningJob.job_id);
                  }}
                />
              </div>
            )}
          </>
        )}

        {!needsSetupOverlay && showNotification && (
          <UpdateNotification
            updateState={updateState}
            isJobRunning={Boolean(runningJob)}
            onInstall={() => {
              if (!runningJob) void installUpdate();
            }}
            onRetryCheck={() => void checkForUpdates(false)}
            onDismiss={dismissNotification}
          />
        )}
      </div>
    </TooltipProvider>
  );
}

export default App;
