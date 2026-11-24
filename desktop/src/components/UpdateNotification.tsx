import { AlertCircle, ArrowUpCircle, Loader2, Sparkles, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import type { UpdateState } from "@/lib/types";
import { formatBytes } from "@/lib/utils";

interface UpdateNotificationProps {
  updateState: UpdateState;
  isJobRunning: boolean;
  onInstall: () => void;
  onRetryCheck: () => void;
  onDismiss: () => void;
}

export function UpdateNotification({
  updateState,
  isJobRunning,
  onInstall,
  onRetryCheck,
  onDismiss,
}: UpdateNotificationProps) {
  const { phase, available, downloadedBytes, totalBytes, error } = updateState;
  const isBusy = phase === "downloading" || phase === "installing";
  const pct =
    phase === "installing"
      ? 100
      : totalBytes && totalBytes > 0
        ? Math.min(100, (downloadedBytes / totalBytes) * 100)
        : 0;

  const title =
    phase === "downloading"
      ? "Update wird heruntergeladen…"
      : phase === "installing"
        ? "Update wird installiert…"
        : phase === "error"
          ? "Update fehlgeschlagen"
          : "Neues Update verfügbar";

  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed bottom-5 right-5 z-50 w-[390px] max-w-[calc(100vw-2.5rem)] rounded-xl border border-primary/45 bg-card/95 p-4 shadow-2xl ring-1 ring-primary/20 backdrop-blur-md"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2.5 min-w-0">
          <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary ring-1 ring-primary/30">
            {isBusy ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : phase === "error" ? (
              <AlertCircle className="h-4 w-4 text-destructive" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
          </div>
          <div className="min-w-0 space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-semibold text-foreground">{title}</span>
              {available && (
                <Badge variant="outline" className="px-1.5 py-0 text-[10px] tabular-nums">
                  v{available.currentVersion} → v{available.version}
                </Badge>
              )}
            </div>
            {phase === "available" && (
              <p className="text-xs text-muted-foreground">
                Eine neue Version von ruby.ai Studio ist verfügbar und kann direkt installiert
                werden.
              </p>
            )}
          </div>
        </div>

        {!isBusy && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-6 w-6 shrink-0 text-muted-foreground hover:text-foreground"
            onClick={onDismiss}
            title="Schließen"
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>

      {phase === "available" && available?.body && (
        <div className="mt-3 max-h-24 overflow-y-auto rounded-md border bg-background/60 p-2.5 text-[11px] text-muted-foreground whitespace-pre-wrap">
          {available.body}
        </div>
      )}

      {phase === "available" && isJobRunning && (
        <p className="mt-2.5 text-[11px] text-amber-400">
          Ein Video-Job läuft noch – danach kannst du direkt aktualisieren.
        </p>
      )}

      {isBusy && (
        <div className="mt-3 space-y-1.5">
          <div className="flex items-center justify-between text-[11px] text-muted-foreground tabular-nums">
            <span>
              {phase === "installing"
                ? "Installer läuft – App startet gleich neu…"
                : totalBytes
                  ? `${formatBytes(downloadedBytes)} / ${formatBytes(totalBytes)}`
                  : formatBytes(downloadedBytes)}
            </span>
            <span>{Math.round(pct)}%</span>
          </div>
          <Progress value={pct} className="h-2" />
        </div>
      )}

      {phase === "error" && error && (
        <p className="mt-2.5 text-[11px] text-destructive break-words">{error}</p>
      )}

      {(phase === "available" || phase === "error") && (
        <div className="mt-3.5 flex items-center justify-end gap-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 px-3 text-xs"
            onClick={onDismiss}
          >
            {phase === "error" ? "Schließen" : "Später"}
          </Button>
          {phase === "available" ? (
            <Button
              type="button"
              size="sm"
              className="h-8 gap-1.5 px-3 text-xs"
              disabled={isJobRunning}
              onClick={onInstall}
            >
              <ArrowUpCircle className="h-3.5 w-3.5" />
              Aktualisieren
            </Button>
          ) : (
            <Button
              type="button"
              size="sm"
              className="h-8 gap-1.5 px-3 text-xs"
              onClick={available ? onInstall : onRetryCheck}
            >
              <ArrowUpCircle className="h-3.5 w-3.5" />
              Erneut versuchen
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
