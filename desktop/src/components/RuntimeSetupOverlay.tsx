import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Gem,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { RuntimeProgressEvent, RuntimeStatus } from "@/lib/types";

interface RuntimeSetupOverlayProps {
  appVersion: string | null;
  status: RuntimeStatus;
  progress: RuntimeProgressEvent | null;
  settingUp: boolean;
  error: string | null;
  onRetry: () => void;
}

const SETUP_STAGES = [
  { key: "uv", step: 1, label: "uv Paketmanager" },
  { key: "ffmpeg", step: 2, label: "FFmpeg & FFprobe (NVENC + zscale)" },
  { key: "venv", step: 3, label: "Python 3.12 Umgebung" },
  { key: "torch", step: 4, label: "PyTorch KI-Backend" },
  { key: "deps", step: 5, label: "Spandrel & Videoverarbeitung" },
] as const;

export function RuntimeSetupOverlay({
  appVersion,
  status,
  progress,
  settingUp,
  error,
  onRetry,
}: RuntimeSetupOverlayProps) {
  const isSync = status.needs_sync && !status.needs_setup;
  const accelLabel =
    status.accelerator === "cuda"
      ? "NVIDIA CUDA GPU"
      : status.accelerator === "mps"
        ? "Apple Silicon MPS"
        : "CPU-Modus";

  const activeStep = progress?.step ?? 0;
  const percent = Math.max(0, Math.min(100, progress?.percent ?? 0));

  return (
    <div className="flex flex-1 items-center justify-center bg-background p-6">
      <Card className="w-full max-w-xl border-border/80 bg-card/85 shadow-2xl">
        <CardHeader className="space-y-3">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/15 text-primary ring-1 ring-primary/30">
                <Gem className="h-5 w-5" />
              </div>
              <div>
                <CardTitle className="text-base">
                  {isSync
                    ? "ruby.ai Runtime wird aktualisiert"
                    : "ruby.ai Studio – Einmalige Einrichtung"}
                </CardTitle>
                <CardDescription className="text-xs">
                  {appVersion ? `Version v${appVersion} · ` : ""}
                  {isSync
                    ? "Neue Engine-Abhängigkeiten werden nach dem Update automatisch nachgeladen."
                    : "FFmpeg, Python 3.12 und PyTorch werden vollautomatisch und isoliert eingerichtet."}
                </CardDescription>
              </div>
            </div>
            <Badge
              variant={status.accelerator === "cuda" ? "success" : "secondary"}
              className="shrink-0 gap-1.5 text-[11px]"
            >
              <Cpu className="h-3 w-3" />
              {accelLabel}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Einrichtung fehlgeschlagen</AlertTitle>
              <AlertDescription className="text-xs break-words">{error}</AlertDescription>
            </Alert>
          )}

          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium text-foreground">
                {progress?.message ?? status.reason ?? "Bereite Laufzeitumgebung vor…"}
              </span>
              <span className="tabular-nums text-muted-foreground">
                {Math.round(percent)}%
              </span>
            </div>
            <Progress value={percent} className="h-2.5" />
            {progress?.detail && (
              <p className="truncate font-mono text-[11px] text-muted-foreground">
                {progress.detail}
              </p>
            )}
          </div>

          <div className="space-y-1.5 rounded-lg border bg-background/50 p-3">
            {SETUP_STAGES.map((stage) => {
              const isDone =
                progress?.stage === "ready" ||
                (activeStep > stage.step) ||
                (progress?.stage === "verify" && stage.step === 5);
              const isActive =
                settingUp &&
                !isDone &&
                (activeStep === stage.step || (activeStep === 0 && stage.step === 1));

              const label =
                stage.key === "torch" ? `PyTorch (${accelLabel})` : stage.label;

              return (
                <div
                  key={stage.key}
                  className="flex items-center justify-between text-xs py-0.5"
                >
                  <div className="flex items-center gap-2">
                    {isDone ? (
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                    ) : isActive ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                    ) : (
                      <span className="inline-block h-3.5 w-3.5 rounded-full border border-muted-foreground/40" />
                    )}
                    <span
                      className={
                        isDone || isActive ? "text-foreground" : "text-muted-foreground"
                      }
                    >
                      {label}
                    </span>
                  </div>
                  <span className="text-[11px] text-muted-foreground">
                    {isDone ? "Fertig" : isActive ? "Läuft…" : "Wartend"}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="flex items-center justify-between gap-3 pt-1">
            <span className="truncate font-mono text-[10px] text-muted-foreground">
              Ziel: {status.runtime_dir}
            </span>
            {error && (
              <Button
                type="button"
                size="sm"
                className="shrink-0 gap-1.5 text-xs"
                disabled={settingUp}
                onClick={onRetry}
              >
                <RefreshCw className={`h-3.5 w-3.5 ${settingUp ? "animate-spin" : ""}`} />
                Erneut versuchen
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
