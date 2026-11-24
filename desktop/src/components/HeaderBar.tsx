import { Activity, ArrowUpCircle, Cpu, Gem, Loader2, RefreshCw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { EngineInfo, UpdateState } from "@/lib/types";

interface HeaderBarProps {
  appVersion: string | null;
  engineInfo: EngineInfo | null;
  loadingInfo: boolean;
  updateState: UpdateState;
  activeView: "studio" | "system";
  onActiveViewChange: (view: "studio" | "system") => void;
  onRefreshInfo: () => void;
  onOpenUpdate: () => void;
}

export function HeaderBar({
  appVersion,
  engineInfo,
  loadingInfo,
  updateState,
  activeView,
  onActiveViewChange,
  onRefreshInfo,
  onOpenUpdate,
}: HeaderBarProps) {
  const hasCuda = Boolean(engineInfo?.cuda_available);
  const hasNvenc = Boolean(engineInfo?.encoders.includes("h264_nvenc"));
  const vramLabel =
    engineInfo?.device.total_memory_gb != null
      ? ` · ${engineInfo.device.total_memory_gb.toFixed(1)} GiB`
      : "";
  const displayVersion = appVersion ?? engineInfo?.version ?? "0.1.0";

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b bg-card/60 px-4 backdrop-blur">
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/15 text-primary ring-1 ring-primary/30">
          <Gem className="h-4 w-4" />
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-semibold tracking-tight">ruby.ai</span>
          <span className="text-xs text-muted-foreground">Social 4K Studio</span>
          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
            v{displayVersion}
          </Badge>
        </div>
      </div>

      <Tabs
        value={activeView}
        onValueChange={(val) => onActiveViewChange(val as "studio" | "system")}
      >
        <TabsList className="h-8">
          <TabsTrigger value="studio" className="text-xs px-4">
            Studio
          </TabsTrigger>
          <TabsTrigger value="system" className="text-xs px-4 gap-1.5">
            <Activity className="h-3.5 w-3.5" />
            Modelle & System
          </TabsTrigger>
        </TabsList>
      </Tabs>

      <div className="flex items-center gap-2">
        {updateState.phase === "available" && updateState.available && (
          <Button
            type="button"
            variant="default"
            size="sm"
            className="h-7 gap-1.5 px-2.5 text-xs"
            onClick={onOpenUpdate}
          >
            <ArrowUpCircle className="h-3.5 w-3.5" />
            Update v{updateState.available.version}
          </Button>
        )}
        {(updateState.phase === "downloading" || updateState.phase === "installing") && (
          <Badge variant="secondary" className="gap-1.5 text-[11px] font-normal">
            <Loader2 className="h-3 w-3 animate-spin" />
            Update…
          </Badge>
        )}

        {engineInfo ? (
          <>
            <Badge variant={hasCuda ? "success" : "secondary"} className="gap-1.5 font-normal">
              <Cpu className="h-3 w-3" />
              <span className="max-w-[200px] truncate">
                {engineInfo.device.name}
                {vramLabel}
              </span>
            </Badge>
            {hasNvenc && (
              <Badge variant="outline" className="text-[11px] font-normal">
                NVENC
              </Badge>
            )}
          </>
        ) : (
          <Badge variant="secondary" className="font-normal">
            {loadingInfo ? "Engine wird geprüft…" : "Engine offline"}
          </Badge>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          onClick={onRefreshInfo}
          disabled={loadingInfo}
          title="Systemstatus aktualisieren"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loadingInfo ? "animate-spin" : ""}`} />
        </Button>
      </div>
    </header>
  );
}
