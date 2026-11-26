import { openUrl } from "@tauri-apps/plugin-opener";
import {
  AlertTriangle,
  ArrowUpCircle,
  CheckCircle2,
  DownloadCloud,
  ExternalLink,
  Loader2,
  RefreshCw,
  Wrench,
  XCircle,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { EngineInfo, RuntimeStatus, UpdateState } from "@/lib/types";
import { formatBytes } from "@/lib/utils";

interface ModelsDoctorViewProps {
  appVersion: string | null;
  engineInfo: EngineInfo | null;
  loadingInfo: boolean;
  runtimeStatus: RuntimeStatus | null;
  updateState: UpdateState;
  isJobRunning: boolean;
  onRefreshInfo: () => void;
  onCheckUpdates: () => void;
  onInstallUpdate: () => void;
  onRepairRuntime: () => void;
}

export function ModelsDoctorView({
  appVersion,
  engineInfo,
  loadingInfo,
  runtimeStatus,
  updateState,
  isJobRunning,
  onRefreshInfo,
  onCheckUpdates,
  onInstallUpdate,
  onRepairRuntime,
}: ModelsDoctorViewProps) {
  const versionLabel = appVersion ?? engineInfo?.version ?? "0.2.0";
  const updateBusy =
    updateState.phase === "checking" ||
    updateState.phase === "downloading" ||
    updateState.phase === "installing";
  const downloadPct =
    updateState.totalBytes && updateState.totalBytes > 0
      ? Math.min(100, (updateState.downloadedBytes / updateState.totalBytes) * 100)
      : 0;

  return (
    <ScrollArea className="flex-1">
      <div className="mx-auto max-w-6xl space-y-6 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              System-Diagnose, Updates & Modell-Registry
            </h1>
            <p className="text-xs text-muted-foreground">
              App-Updates (OTA), verwaltete Runtime, GPU-Backend, FFmpeg-Encoder und Open-Weight-Modelle.
            </p>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="gap-2"
            onClick={onRefreshInfo}
            disabled={loadingInfo}
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loadingInfo ? "animate-spin" : ""}`} />
            Neu prüfen
          </Button>
        </div>

        {engineInfo?.cuda_warning && (
          <Alert variant="warning">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>NVIDIA GPU-Treiber prüfen</AlertTitle>
            <AlertDescription className="text-xs">
              {engineInfo.cuda_warning}
            </AlertDescription>
          </Alert>
        )}

        {/* App, Runtime & OTA Updates Card */}
        <Card>
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <CardTitle>App, Laufzeit & OTA-Updates</CardTitle>
                <CardDescription>
                  Signierte Over-the-Air-Updates via GitHub Releases (Velt1/ruby.ai)
                </CardDescription>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline">App v{versionLabel}</Badge>
                {runtimeStatus && (
                  <Badge variant={runtimeStatus.ready ? "success" : "warning"}>
                    {runtimeStatus.mode === "workspace"
                      ? "Dev Workspace"
                      : "Managed Runtime"}
                  </Badge>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 text-xs">
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <div className="space-y-1">
                <span className="text-muted-foreground block">Beschleuniger</span>
                <span className="font-medium uppercase">
                  {runtimeStatus?.accelerator ?? engineInfo?.device.backend ?? "n/a"}
                </span>
              </div>
              <div className="space-y-1">
                <span className="text-muted-foreground block">Update-Status</span>
                <span className="font-medium">
                  {updateState.phase === "checking" && "Suche nach Updates…"}
                  {updateState.phase === "up-to-date" && "Auf dem neuesten Stand"}
                  {updateState.phase === "available" &&
                    `Neue Version v${updateState.available?.version} verfügbar`}
                  {updateState.phase === "downloading" && "Update wird heruntergeladen…"}
                  {updateState.phase === "installing" && "Installiere Update & starte neu…"}
                  {updateState.phase === "error" && "Update-Prüfung fehlgeschlagen"}
                  {updateState.phase === "idle" && "Bereit (automatische Prüfung beim Start)"}
                </span>
              </div>
              <div className="space-y-1">
                <span className="text-muted-foreground block">Laufzeit-Ordner</span>
                <span
                  className="block truncate font-mono text-[11px]"
                  title={runtimeStatus?.runtime_dir ?? ""}
                >
                  {runtimeStatus?.runtime_dir ?? "—"}
                </span>
              </div>
            </div>

            {updateState.phase === "available" && updateState.available?.body && (
              <div className="rounded-md border bg-background/60 p-3 text-[11px] text-muted-foreground whitespace-pre-wrap">
                {updateState.available.body}
              </div>
            )}

            {(updateState.phase === "downloading" || updateState.phase === "installing") && (
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px] text-muted-foreground tabular-nums">
                  <span>
                    {updateState.phase === "installing"
                      ? "Installation läuft…"
                      : "Download läuft…"}
                  </span>
                  <span>
                    {updateState.totalBytes
                      ? `${formatBytes(updateState.downloadedBytes)} / ${formatBytes(updateState.totalBytes)}`
                      : formatBytes(updateState.downloadedBytes)}
                  </span>
                </div>
                <Progress value={downloadPct} className="h-2" />
              </div>
            )}

            {updateState.phase === "error" && updateState.error && (
              <p className="text-[11px] text-destructive break-words">{updateState.error}</p>
            )}

            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8 gap-1.5 text-xs"
                disabled={updateBusy}
                onClick={onCheckUpdates}
              >
                {updateState.phase === "checking" ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <DownloadCloud className="h-3.5 w-3.5" />
                )}
                Nach Updates suchen
              </Button>

              {updateState.phase === "available" && updateState.available && (
                <Button
                  type="button"
                  size="sm"
                  className="h-8 gap-1.5 text-xs"
                  disabled={isJobRunning || updateBusy}
                  onClick={onInstallUpdate}
                  title={
                    isJobRunning
                      ? "Während ein Export/Job läuft nicht verfügbar"
                      : undefined
                  }
                >
                  <ArrowUpCircle className="h-3.5 w-3.5" />
                  v{updateState.available.version} installieren & neu starten
                </Button>
              )}

              {runtimeStatus?.mode === "managed" && (
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  className="h-8 gap-1.5 text-xs"
                  disabled={isJobRunning || updateBusy}
                  onClick={onRepairRuntime}
                >
                  <Wrench className="h-3.5 w-3.5" />
                  Runtime neu synchronisieren
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {!engineInfo ? (
          <Card>
            <CardContent className="p-8 text-center text-sm text-muted-foreground">
              {loadingInfo
                ? "Lade Engine-Diagnose und Modell-Registry…"
                : "Keine Engine-Informationen verfügbar."}
            </CardContent>
          </Card>
        ) : (
          <>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <Card>
                <CardHeader>
                  <CardTitle>Compute & GPU</CardTitle>
                  <CardDescription>{engineInfo.platform}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Backend</span>
                    <span className="font-medium uppercase">{engineInfo.device.backend}</span>
                  </div>
                  <div className="flex justify-between gap-2">
                    <span className="text-muted-foreground">Gerät</span>
                    <span className="truncate font-medium">{engineInfo.device.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">VRAM</span>
                    <span className="tabular-nums font-medium">
                      {engineInfo.device.total_memory_gb != null
                        ? `${engineInfo.device.total_memory_gb} GiB` +
                          (engineInfo.device.free_memory_gb != null
                            ? ` (${engineInfo.device.free_memory_gb} GiB frei)`
                            : "")
                        : "n/a"}
                    </span>
                  </div>
                  {engineInfo.nvidia_driver_version && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">NVIDIA Treiber</span>
                      <span className="tabular-nums">{engineInfo.nvidia_driver_version}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Standard-Präzision</span>
                    <span>{engineInfo.device.default_fp16 ? "fp16" : "fp32"}</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Python & AI-Stack</CardTitle>
                  <CardDescription>ruby.ai v{engineInfo.version}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Python</span>
                    <span className="tabular-nums">{engineInfo.python}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">PyTorch</span>
                    <span className="tabular-nums">{engineInfo.pytorch}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Spandrel</span>
                    <span className="tabular-nums">{engineInfo.spandrel}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">CUDA verfügbar</span>
                    <span>{engineInfo.cuda_available ? "Ja" : "Nein"}</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>FFmpeg & Video-Encoder</CardTitle>
                  <CardDescription>Streaming-Decode & Export</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">ffmpeg</span>
                    <span className="flex items-center gap-1">
                      {engineInfo.ffmpeg_available ? (
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                      ) : (
                        <XCircle className="h-3.5 w-3.5 text-destructive" />
                      )}
                      {engineInfo.ffmpeg_available ? "Erkannt" : "Fehlt"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">ffprobe</span>
                    <span className="flex items-center gap-1">
                      {engineInfo.ffprobe_available ? (
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                      ) : (
                        <XCircle className="h-3.5 w-3.5 text-destructive" />
                      )}
                      {engineInfo.ffprobe_available ? "Erkannt" : "Fehlt"}
                    </span>
                  </div>
                  {engineInfo.nvenc_usable !== null &&
                    engineInfo.nvenc_usable !== undefined && (
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-muted-foreground">NVENC Test</span>
                        <Badge
                          variant={engineInfo.nvenc_usable ? "success" : "warning"}
                          className="text-[10px]"
                          title={engineInfo.nvenc_error ?? undefined}
                        >
                          {engineInfo.nvenc_usable ? "Verwendbar" : "Fallback auf libx264"}
                        </Badge>
                      </div>
                    )}
                  <div className="pt-1">
                    <span className="text-muted-foreground block mb-1.5">Verfügbare Encoder</span>
                    <div className="flex flex-wrap gap-1.5">
                      {engineInfo.encoders.map((enc) => (
                        <Badge key={enc} variant="secondary" className="text-[11px]">
                          {enc}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>
                  Registrierte KI-Modelle ({engineInfo.models.length})
                </CardTitle>
                <CardDescription>
                  Sichere .safetensors-Gewichte – beim ersten Aufruf automatisch gecacht.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b text-muted-foreground">
                        <th className="py-2 pr-3 font-medium">Key</th>
                        <th className="py-2 pr-3 font-medium">Scale</th>
                        <th className="py-2 pr-3 font-medium">Architektur</th>
                        <th className="py-2 pr-3 font-medium">Quelle</th>
                        <th className="py-2 pr-3 font-medium">Autor</th>
                        <th className="py-2 pr-3 font-medium">Lizenz</th>
                        <th className="py-2 pr-3 font-medium">Beschreibung</th>
                        <th className="py-2 text-right font-medium">Link</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/60">
                      {engineInfo.models.map((m) => (
                        <tr key={m.key} className="hover:bg-accent/30">
                          <td className="py-2.5 pr-3 font-mono font-medium text-foreground">
                            {m.key}
                          </td>
                          <td className="py-2.5 pr-3">
                            <Badge variant="outline" className="text-[10px]">
                              {m.scale}×
                            </Badge>
                          </td>
                          <td className="py-2.5 pr-3">{m.architecture}</td>
                          <td className="py-2.5 pr-3 uppercase text-muted-foreground">
                            {m.source_kind}
                          </td>
                          <td className="py-2.5 pr-3">{m.author}</td>
                          <td className="py-2.5 pr-3">
                            <Badge
                              variant={m.commercial_use ? "success" : "warning"}
                              className="text-[10px]"
                            >
                              {m.license}
                            </Badge>
                          </td>
                          <td className="py-2.5 pr-3 text-muted-foreground max-w-md">
                            {m.description}
                          </td>
                          <td className="py-2.5 text-right">
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon"
                              className="h-7 w-7"
                              title={m.source_url}
                              onClick={() => void openUrl(m.source_url)}
                            >
                              <ExternalLink className="h-3.5 w-3.5" />
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </ScrollArea>
  );
}
