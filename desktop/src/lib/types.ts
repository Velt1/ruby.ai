export type JobMode = "enhance" | "benchmark" | "preview";

export interface DeviceInfoPayload {
  backend: string;
  name: string;
  default_fp16: boolean;
  total_memory_gb: number | null;
  error?: string | null;
}

export interface PresetSpec {
  key: string;
  model: string;
  temporal_denoise: number;
  ai_strength: number;
  sharpen: number;
  clarity: number;
  deband: boolean;
  grain: number;
}

export interface ProfileSpec {
  key: string;
  label: string;
  badge: string;
  style: string;
  tagline: string;
  primary: boolean;
}

export interface ExportTargetSpec {
  key: string;
  label: string;
  badge: string;
  max_short_edge: number;
  maxrate_kbps_30fps: number | null;
  maxrate_kbps_60fps: number | null;
  gop_seconds: number;
  audio_sample_rate: number;
  description: string;
}

export interface ModelSpec {
  key: string;
  architecture: string;
  scale: number;
  author: string;
  license: string;
  description: string;
  source_url: string;
  source_kind: string;
  commercial_use: boolean;
}

export interface EngineOptions {
  codecs: string[];
  x264_presets: string[];
  nvenc_presets: string[];
  hwaccels: string[];
  export_targets: string[];
}

export interface EngineInfo {
  version: string;
  platform: string;
  python: string;
  pytorch: string;
  spandrel: string;
  device: DeviceInfoPayload;
  cuda_available: boolean;
  mps_available: boolean;
  ffmpeg_available: boolean;
  ffprobe_available: boolean;
  ffmpeg_path: string | null;
  ffprobe_path: string | null;
  encoders: string[];
  presets: PresetSpec[];
  profiles: ProfileSpec[];
  export_targets: ExportTargetSpec[];
  models: ModelSpec[];
  options: EngineOptions;
}

export interface VideoProbe {
  path: string;
  filename: string;
  width: number;
  height: number;
  display_width: number;
  display_height: number;
  rotation: number;
  fps_num: number;
  fps_den: number;
  fps: number;
  fps_str: string;
  duration: number;
  frame_count: number;
  has_audio: boolean;
  pix_fmt: string;
  codec_name: string;
  color_space?: string;
  color_transfer?: string;
  is_hdr?: boolean;
  file_size_bytes: number;
  total_bitrate_kbps: number;
  video_bitrate_kbps: number;
  target_width: number;
  target_height: number;
  poster_path: string | null;
}

export interface VideoAnalysis {
  path: string;
  avg_luma: number;
  avg_saturation: number;
  bits_per_pixel: number;
  is_low_light: boolean;
  is_heavily_compressed: boolean;
  is_hdr: boolean;
  is_high_fps: boolean;
  recommended_preset: string;
  recommended_export_target: string;
  recommended_overrides: Record<string, unknown>;
  reasons: string[];
}

export interface PipelineSettings {
  preset: string;
  model: string;
  export_target: string;
  max_short_edge: number;
  temporal_denoise: number;
  ai_strength: number;
  sharpen: number;
  clarity: number;
  deband: boolean;
  grain: number;
  hdr_to_sdr: boolean;
  maxrate_kbps: number | null;
  crf: number;
  codec: string;
  x264_preset: string;
  nvenc_preset: string;
  hwaccel: string;
  tile_size: number | null;
  tile_overlap: number;
  tile_batch_size: number;
  batch_size: number;
  optimize_gpu: boolean;
  compile_model: boolean;
  async_io: boolean;
  start_sec: number | null;
  duration_sec: number | null;
  bitrate_kbps: number;
  reference_path: string | null;
  output_path: string | null;
}

export interface EnhanceJobResult {
  output_path: string;
  frames_processed: number;
  elapsed_sec: number;
  fps: number;
  device_label: string;
  target_width: number;
  target_height: number;
  model_key: string;
  codec_used: string;
  effective_tile_size: number | null;
  batch_size: number;
  file_size_bytes: number;
  total_bitrate_kbps: number;
  video_bitrate_kbps: number;
  before_frame_png: string | null;
  after_frame_png: string | null;
  before_video_path?: string | null;
}

export interface BenchmarkArtifactEntry {
  path: string;
  size_bytes: number;
  total_bitrate_kbps: number;
  video_bitrate_kbps: number;
}

export interface BenchmarkJobResult {
  output_dir: string;
  bitrate_kbps: number;
  artifacts: Record<string, BenchmarkArtifactEntry>;
  before_frame_png: string | null;
  after_frame_png: string | null;
  enhance_result: Omit<
    EnhanceJobResult,
    "file_size_bytes" | "total_bitrate_kbps" | "video_bitrate_kbps" | "before_frame_png" | "after_frame_png"
  >;
}

export type JobEvent =
  | {
      type: "started";
      job_id: string;
      mode: JobMode;
      input_path: string;
      output_path?: string;
      output_dir?: string;
      preset: string;
      model: string;
    }
  | {
      type: "progress";
      job_id: string;
      stage: string;
      frames_done: number;
      total_frames: number | null;
      percent: number | null;
      fps: number;
      elapsed_sec: number;
      eta_sec: number | null;
    }
  | {
      type: "completed";
      job_id: string;
      mode: JobMode;
      result: EnhanceJobResult | BenchmarkJobResult;
    }
  | {
      type: "error";
      job_id?: string | null;
      message: string;
    }
  | {
      type: "cancelled";
      job_id: string;
    };

export type JobStatus = "running" | "completed" | "failed" | "cancelled";

export interface JobRecord {
  job_id: string;
  mode: JobMode;
  input_path: string;
  filename: string;
  preset: string;
  model: string;
  status: JobStatus;
  frames_done: number;
  total_frames: number | null;
  percent: number;
  fps: number;
  elapsed_sec: number;
  eta_sec: number | null;
  error_message?: string;
  enhance_result?: EnhanceJobResult;
  benchmark_result?: BenchmarkJobResult;
}

export interface RuntimeStatus {
  mode: "workspace" | "managed";
  ready: boolean;
  needs_setup: boolean;
  needs_sync: boolean;
  accelerator: string;
  runtime_dir: string;
  engine_src_dir: string | null;
  python_bin: string | null;
  uv_installed: boolean;
  ffmpeg_installed: boolean;
  venv_installed: boolean;
  installed_deps_hash: string | null;
  expected_deps_hash: string | null;
  reason: string | null;
}

export interface RuntimeProgressEvent {
  stage: string;
  step: number;
  total_steps: number;
  percent: number;
  message: string;
  detail: string;
}

export type UpdatePhase =
  | "idle"
  | "checking"
  | "up-to-date"
  | "available"
  | "downloading"
  | "installing"
  | "error";

export interface AvailableUpdateInfo {
  version: string;
  currentVersion: string;
  date?: string;
  body?: string;
}

export interface UpdateState {
  phase: UpdatePhase;
  available: AvailableUpdateInfo | null;
  downloadedBytes: number;
  totalBytes: number | null;
  error: string | null;
  lastCheckedAt: number | null;
}
