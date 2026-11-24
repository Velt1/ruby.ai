import { useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  Eye,
  FolderOutput,
  Play,
  SlidersHorizontal,
  Sparkles,
  Square,
  Wand2,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import type {
  EngineInfo,
  PipelineSettings,
  VideoAnalysis,
  VideoProbe,
} from "@/lib/types";
import { cn } from "@/lib/utils";

interface SettingsPanelProps {
  engineInfo: EngineInfo | null;
  probe: VideoProbe | null;
  analysis: VideoAnalysis | null;
  settings: PipelineSettings;
  isRunning: boolean;
  onSelectPreset: (presetKey: string) => void;
  onSelectExportTarget: (targetKey: string) => void;
  onApplyAutoEstimate: () => void;
  onUpdateSettings: (patch: Partial<PipelineSettings>) => void;
  onPickOutput: () => void;
  onPickReference: () => void;
  onStartPreview: () => void;
  onStartExport: () => void;
  onCancel: () => void;
}

interface SliderRowProps {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  displayValue: string;
  onChange: (val: number) => void;
}

function SliderRow({
  label,
  hint,
  value,
  min,
  max,
  step,
  displayValue,
  onChange,
}: SliderRowProps) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <div>
          <Label className="text-xs text-foreground/90">{label}</Label>
          {hint && <p className="text-[10px] text-muted-foreground">{hint}</p>}
        </div>
        <span className="text-xs font-medium tabular-nums text-muted-foreground">
          {displayValue}
        </span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={([v]) => onChange(v)}
      />
    </div>
  );
}

export function SettingsPanel({
  engineInfo,
  probe,
  analysis,
  settings,
  isRunning,
  onSelectPreset,
  onSelectExportTarget,
  onApplyAutoEstimate,
  onUpdateSettings,
  onPickOutput,
  onPickReference,
  onStartPreview,
  onStartExport,
  onCancel,
}: SettingsPanelProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const selectedModel = engineInfo?.models.find((m) => m.key === settings.model);
  const primaryProfiles = (engineInfo?.profiles ?? []).filter((p) => p.primary);
  const mainExportTargets = (engineInfo?.export_targets ?? []).filter(
    (t) => t.key !== "custom"
  );
  const tileSelectValue =
    settings.tile_size === null ? "auto" : String(settings.tile_size);

  return (
    <aside className="flex h-full w-[390px] shrink-0 flex-col border-l bg-card/50">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          <span className="text-sm font-semibold">Look & Social-Export</span>
        </div>
        {analysis && (
          <Button
            type="button"
            variant="secondary"
            size="sm"
            className="h-7 gap-1.5 text-xs"
            onClick={onApplyAutoEstimate}
            title="Automatische Empfehlung anhand der Clip-Analyse anwenden"
          >
            <Wand2 className="h-3 w-3 text-primary" />
            Auto
          </Button>
        )}
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-5 p-4">
          {/* 1. Export target */}
          <div className="space-y-2">
            <Label className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              1. Plattform-Ziel
            </Label>
            <div className="grid grid-cols-2 gap-2">
              {mainExportTargets.map((target) => {
                const active = settings.export_target === target.key;
                return (
                  <button
                    key={target.key}
                    type="button"
                    onClick={() => onSelectExportTarget(target.key)}
                    className={cn(
                      "flex flex-col items-start rounded-lg border px-2.5 py-2 text-left transition-colors",
                      active
                        ? "border-primary bg-primary/15 text-foreground"
                        : "border-border bg-background/50 hover:bg-accent/60"
                    )}
                  >
                    <div className="flex w-full items-center justify-between gap-1">
                      <span className="text-xs font-semibold truncate">{target.label}</span>
                    </div>
                    <span className="mt-0.5 text-[10px] text-muted-foreground">
                      {target.badge}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          <Separator />

          {/* 2. Topaz-style Profiles */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                2. KI-Look
              </Label>
              {analysis && (
                <span className="text-[10px] text-muted-foreground">
                  Empfohlen: <strong className="text-foreground">{analysis.recommended_preset}</strong>
                </span>
              )}
            </div>
            <div className="grid grid-cols-1 gap-2">
              {primaryProfiles.map((profile) => {
                const active = settings.preset === profile.key;
                return (
                  <button
                    key={profile.key}
                    type="button"
                    onClick={() => onSelectPreset(profile.key)}
                    className={cn(
                      "flex flex-col items-start rounded-lg border px-3 py-2 text-left transition-colors",
                      active
                        ? "border-primary bg-primary/15 text-foreground"
                        : "border-border bg-background/50 hover:bg-accent/60"
                    )}
                  >
                    <div className="flex w-full items-center justify-between gap-2">
                      <span className="text-xs font-semibold">{profile.label}</span>
                      <Badge
                        variant={active ? "default" : "outline"}
                        className="text-[10px] px-1.5 py-0"
                      >
                        {profile.badge}
                      </Badge>
                    </div>
                    <span className="mt-0.5 text-[11px] text-muted-foreground leading-snug">
                      {profile.tagline}
                    </span>
                  </button>
                );
              })}
            </div>

            {selectedModel && !selectedModel.commercial_use && (
              <Alert variant="warning">
                <AlertTriangle className="h-3.5 w-3.5" />
                <AlertDescription className="text-[11px]">
                  Das schnelle Turbo-Modell (<code className="font-mono">{selectedModel.key}</code>)
                  steht unter {selectedModel.license} (nur nicht-kommerzielle Nutzung).
                </AlertDescription>
              </Alert>
            )}
          </div>

          <Separator />

          {/* 3. Human-readable 4K/Social Sliders */}
          <div className="space-y-3.5">
            <Label className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              3. Feinjustierung
            </Label>

            <SliderRow
              label="Detail-Restauration"
              hint="Rekonstruiert komprimierte Details per KI"
              value={settings.ai_strength}
              min={0}
              max={1}
              step={0.05}
              displayValue={`${Math.round(settings.ai_strength * 100)}%`}
              onChange={(v) => onUpdateSettings({ ai_strength: v })}
            />
            <SliderRow
              label="Kanten-Schärfe (Luma)"
              hint="Kontrast-adaptives Schärfen ohne Farbsäume"
              value={settings.sharpen}
              min={0}
              max={1}
              step={0.05}
              displayValue={`${Math.round(settings.sharpen * 100)}%`}
              onChange={(v) => onUpdateSettings({ sharpen: v })}
            />
            <SliderRow
              label="4K Clarity / OLED-Pop"
              hint="Mikrokontrast für Knackigkeit auf Handy-Displays"
              value={settings.clarity}
              min={0}
              max={0.6}
              step={0.02}
              displayValue={`${Math.round((settings.clarity / 0.6) * 100)}%`}
              onChange={(v) => onUpdateSettings({ clarity: v })}
            />
            <SliderRow
              label="Rauschen glätten"
              hint="Entfernt Sensorflimmern vor der Kompression"
              value={settings.temporal_denoise}
              min={0}
              max={0.25}
              step={0.01}
              displayValue={`${Math.round((settings.temporal_denoise / 0.25) * 100)}%`}
              onChange={(v) => onUpdateSettings({ temporal_denoise: v })}
            />

            <div className="flex items-center justify-between pt-1">
              <div>
                <Label className="text-xs">Anti-Banding Schutz (Deband + Dither)</Label>
                <p className="text-[10px] text-muted-foreground">
                  Verhindert Treppenstufen in dunklen Flächen
                </p>
              </div>
              <Switch
                checked={settings.deband}
                onCheckedChange={(checked) =>
                  onUpdateSettings({ deband: checked, grain: checked ? Math.max(2, settings.grain) : 0 })
                }
              />
            </div>
          </div>

          <Separator />

          {/* 4. Advanced Technical Controls (Collapsed by Default) */}
          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
            <CollapsibleTrigger asChild>
              <Button
                type="button"
                variant="ghost"
                className="flex w-full items-center justify-between px-1 py-1.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground"
              >
                <span className="flex items-center gap-1.5">
                  <SlidersHorizontal className="h-3.5 w-3.5" />
                  Erweitert (Profi & Engine)
                </span>
                <ChevronDown
                  className={cn(
                    "h-4 w-4 transition-transform",
                    advancedOpen && "rotate-180"
                  )}
                />
              </Button>
            </CollapsibleTrigger>

            <CollapsibleContent className="space-y-3.5 pt-3">
              <div className="space-y-1.5">
                <Label className="text-[11px] text-muted-foreground">
                  Direkte 1×-Modellwahl
                </Label>
                <Select
                  value={settings.model}
                  onValueChange={(val) => onUpdateSettings({ model: val })}
                >
                  <SelectTrigger className="h-8">
                    <SelectValue placeholder="Modell wählen" />
                  </SelectTrigger>
                  <SelectContent>
                    {(engineInfo?.models ?? []).map((m) => (
                      <SelectItem key={m.key} value={m.key}>
                        {m.key} · {m.architecture} ({m.license})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[11px] text-muted-foreground">Start (s)</Label>
                  <Input
                    type="number"
                    min={0}
                    step={0.5}
                    placeholder="0.0"
                    value={settings.start_sec ?? ""}
                    onChange={(e) =>
                      onUpdateSettings({
                        start_sec: e.target.value === "" ? null : Number(e.target.value),
                      })
                    }
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[11px] text-muted-foreground">Dauer (s)</Label>
                  <Input
                    type="number"
                    min={0.5}
                    step={0.5}
                    placeholder="voll"
                    value={settings.duration_sec ?? ""}
                    onChange={(e) =>
                      onUpdateSettings({
                        duration_sec: e.target.value === "" ? null : Number(e.target.value),
                      })
                    }
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[11px] text-muted-foreground">
                    Social-Check Bitrate (kbps)
                  </Label>
                  <Input
                    type="number"
                    min={800}
                    max={25000}
                    step={250}
                    value={settings.bitrate_kbps}
                    onChange={(e) =>
                      onUpdateSettings({
                        bitrate_kbps: Math.max(500, Number(e.target.value) || 3000),
                      })
                    }
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[11px] text-muted-foreground">
                    Maxrate Deckel (kbps)
                  </Label>
                  <Input
                    type="number"
                    min={1000}
                    max={100000}
                    step={1000}
                    placeholder="Auto"
                    value={settings.maxrate_kbps ?? ""}
                    onChange={(e) =>
                      onUpdateSettings({
                        maxrate_kbps: e.target.value === "" ? null : Number(e.target.value),
                      })
                    }
                  />
                </div>
              </div>

              <div className="space-y-1">
                <Label className="text-[11px] text-muted-foreground">
                  Ziel-Datei (optional)
                </Label>
                <div className="flex gap-1.5">
                  <Input
                    readOnly
                    placeholder="Automatisch neben Quelldatei"
                    value={settings.output_path ?? ""}
                    className="truncate"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 shrink-0"
                    onClick={onPickOutput}
                    title="Ausgabepfad wählen"
                  >
                    <FolderOutput className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>

              <div className="space-y-1">
                <Label className="text-[11px] text-muted-foreground">
                  Referenz-Clip für A/B-Vorschau (optional)
                </Label>
                <div className="flex gap-1.5">
                  <Input
                    readOnly
                    placeholder="Kein Referenz-Clip"
                    value={settings.reference_path ?? ""}
                    className="truncate"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-8 px-2.5 text-xs shrink-0"
                    onClick={onPickReference}
                  >
                    Wählen
                  </Button>
                </div>
              </div>

              <SliderRow
                label="CRF / CQ Qualität (kleiner = höher)"
                value={settings.crf}
                min={10}
                max={28}
                step={1}
                displayValue={`${settings.crf}`}
                onChange={(v) => onUpdateSettings({ crf: v })}
              />

              <SliderRow
                label="Anti-Banding Grain"
                value={settings.grain}
                min={0}
                max={8}
                step={1}
                displayValue={`${settings.grain}`}
                onChange={(v) => onUpdateSettings({ grain: v })}
              />

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[11px] text-muted-foreground">Encoder</Label>
                  <Select
                    value={settings.codec}
                    onValueChange={(v) => onUpdateSettings({ codec: v })}
                  >
                    <SelectTrigger className="h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {(engineInfo?.options.codecs ?? ["auto", "libx264", "h264_nvenc"]).map(
                        (c) => (
                          <SelectItem key={c} value={c}>
                            {c}
                          </SelectItem>
                        )
                      )}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1">
                  <Label className="text-[11px] text-muted-foreground">HW-Decode</Label>
                  <Select
                    value={settings.hwaccel}
                    onValueChange={(v) => onUpdateSettings({ hwaccel: v })}
                  >
                    <SelectTrigger className="h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {(engineInfo?.options.hwaccels ?? ["none", "auto", "cuda"]).map((h) => (
                        <SelectItem key={h} value={h}>
                          {h}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div className="space-y-1">
                  <Label className="text-[11px] text-muted-foreground">Tiling</Label>
                  <Select
                    value={tileSelectValue}
                    onValueChange={(v) =>
                      onUpdateSettings({ tile_size: v === "auto" ? null : Number(v) })
                    }
                  >
                    <SelectTrigger className="h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Auto</SelectItem>
                      <SelectItem value="0">Full-Frame</SelectItem>
                      <SelectItem value="768">768px</SelectItem>
                      <SelectItem value="512">512px</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1">
                  <Label className="text-[11px] text-muted-foreground">Frame-Batch</Label>
                  <Select
                    value={String(settings.batch_size)}
                    onValueChange={(v) => onUpdateSettings({ batch_size: Number(v) })}
                  >
                    <SelectTrigger className="h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1</SelectItem>
                      <SelectItem value="2">2</SelectItem>
                      <SelectItem value="4">4</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1">
                  <Label className="text-[11px] text-muted-foreground">NVENC Preset</Label>
                  <Select
                    value={settings.nvenc_preset}
                    onValueChange={(v) => onUpdateSettings({ nvenc_preset: v })}
                  >
                    <SelectTrigger className="h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {(engineInfo?.options.nvenc_presets ?? ["p4", "p5", "p6", "p7"]).map(
                        (p) => (
                          <SelectItem key={p} value={p}>
                            {p}
                          </SelectItem>
                        )
                      )}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2.5 pt-1">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">HDR → SDR Tone-Mapping (BT.709)</Label>
                  <Switch
                    checked={settings.hdr_to_sdr}
                    onCheckedChange={(checked) => onUpdateSettings({ hdr_to_sdr: checked })}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <Label className="text-xs">GPU-Optimierungen (FastGN + NHWC)</Label>
                  <Switch
                    checked={settings.optimize_gpu}
                    onCheckedChange={(checked) =>
                      onUpdateSettings({ optimize_gpu: checked })
                    }
                  />
                </div>
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Async Decode/Encode Queues</Label>
                  <Switch
                    checked={settings.async_io}
                    onCheckedChange={(checked) => onUpdateSettings({ async_io: checked })}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <Label className="text-xs">torch.compile (experimentell)</Label>
                  <Switch
                    checked={settings.compile_model}
                    onCheckedChange={(checked) =>
                      onUpdateSettings({ compile_model: checked })
                    }
                  />
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        </div>
      </ScrollArea>

      {/* Topaz-style Dual Action Footer */}
      <div className="border-t bg-card/80 p-4 space-y-2">
        {isRunning ? (
          <Button
            type="button"
            variant="destructive"
            className="w-full gap-2"
            onClick={onCancel}
          >
            <Square className="h-4 w-4" />
            Laufenden Vorgang abbrechen
          </Button>
        ) : (
          <>
            <Button
              type="button"
              variant="secondary"
              className="w-full gap-2"
              disabled={!probe}
              onClick={onStartPreview}
            >
              <Eye className="h-4 w-4" />
              2s Vorschau (Social-Check)
            </Button>
            <Button
              type="button"
              className="w-full gap-2"
              disabled={!probe}
              onClick={onStartExport}
            >
              <Play className="h-4 w-4" />
              Video exportieren
            </Button>
          </>
        )}
      </div>
    </aside>
  );
}
