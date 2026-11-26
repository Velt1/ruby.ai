import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import type {
  EngineInfo,
  JobEvent,
  JobMode,
  RuntimeProgressEvent,
  RuntimeStatus,
  VideoAnalysis,
  VideoProbe,
} from "./types";

export interface StartJobPayload {
  input_path: string;
  output_path?: string | null;
  output_dir?: string | null;
  preset: string;
  start_sec?: number | null;
  duration_sec?: number | null;
  bitrate_kbps?: number;
  reference_path?: string | null;
  config: Record<string, unknown>;
}

export interface EngineClient {
  getRuntimeStatus(): Promise<RuntimeStatus>;
  ensureRuntimeReady(forceReinstall?: boolean): Promise<RuntimeStatus>;
  subscribeRuntimeEvents(handler: (event: RuntimeProgressEvent) => void): Promise<UnlistenFn>;
  getEngineInfo(): Promise<EngineInfo>;
  probeVideo(path: string, maxShortEdge?: number, model?: string): Promise<VideoProbe>;
  analyzeVideo(path: string): Promise<VideoAnalysis>;
  startJob(mode: JobMode, payload: StartJobPayload): Promise<{ job_id: string }>;
  cancelJob(jobId: string): Promise<boolean>;
  subscribeJobEvents(handler: (event: JobEvent) => void): Promise<UnlistenFn>;
  pickVideoFile(): Promise<string | null>;
  pickOutputFile(defaultPath?: string): Promise<string | null>;
  pickOutputDir(): Promise<string | null>;
  revealInExplorer(path: string): Promise<void>;
  toMediaUrl(path: string | null | undefined): string | null;
}

export class TauriEngineClient implements EngineClient {
  async getRuntimeStatus(): Promise<RuntimeStatus> {
    return invoke<RuntimeStatus>("get_runtime_status");
  }

  async ensureRuntimeReady(forceReinstall = false): Promise<RuntimeStatus> {
    return invoke<RuntimeStatus>("ensure_runtime_ready", { forceReinstall });
  }

  async subscribeRuntimeEvents(
    handler: (event: RuntimeProgressEvent) => void
  ): Promise<UnlistenFn> {
    return listen<RuntimeProgressEvent>("ruby://runtime-event", (ev) => {
      handler(ev.payload);
    });
  }

  async getEngineInfo(): Promise<EngineInfo> {
    return invoke<EngineInfo>("get_engine_info");
  }

  async probeVideo(
    path: string,
    maxShortEdge = 1080,
    model?: string
  ): Promise<VideoProbe> {
    return invoke<VideoProbe>("probe_video", {
      path,
      maxShortEdge,
      model,
    });
  }

  async analyzeVideo(path: string): Promise<VideoAnalysis> {
    return invoke<VideoAnalysis>("analyze_video", { path });
  }

  async startJob(mode: JobMode, payload: StartJobPayload): Promise<{ job_id: string }> {
    return invoke<{ job_id: string }>("start_job", {
      mode,
      payload,
    });
  }

  async cancelJob(jobId: string): Promise<boolean> {
    return invoke<boolean>("cancel_job", { jobId });
  }

  async subscribeJobEvents(handler: (event: JobEvent) => void): Promise<UnlistenFn> {
    return listen<JobEvent>("ruby://job-event", (ev) => {
      handler(ev.payload);
    });
  }

  async pickVideoFile(): Promise<string | null> {
    const selected = await open({
      multiple: false,
      directory: false,
      filters: [
        {
          name: "Video",
          extensions: ["mp4", "mov", "mkv", "webm", "avi", "m4v"],
        },
      ],
    });
    return typeof selected === "string" ? selected : null;
  }

  async pickOutputFile(defaultPath?: string): Promise<string | null> {
    const isMov = defaultPath?.toLowerCase().endsWith(".mov");
    const selected = await save({
      defaultPath,
      filters: isMov
        ? [{ name: "ProRes MOV", extensions: ["mov"] }]
        : [{ name: "MP4 Video", extensions: ["mp4"] }],
    });
    return selected ?? null;
  }

  async pickOutputDir(): Promise<string | null> {
    const selected = await open({
      multiple: false,
      directory: true,
    });
    return typeof selected === "string" ? selected : null;
  }

  async revealInExplorer(path: string): Promise<void> {
    await invoke("reveal_in_explorer", { path });
  }

  toMediaUrl(path: string | null | undefined): string | null {
    if (!path) return null;
    return convertFileSrc(path);
  }
}

export const engineClient: EngineClient = new TauriEngineClient();
