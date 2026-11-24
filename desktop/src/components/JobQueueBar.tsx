import { CheckCircle2, FolderOpen, Loader2, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import type { JobRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

interface JobQueueBarProps {
  jobs: JobRecord[];
  activeJobId: string | null;
  onSelectJob: (jobId: string) => void;
  onRevealPath: (path: string) => void;
}

export function JobQueueBar({
  jobs,
  activeJobId,
  onSelectJob,
  onRevealPath,
}: JobQueueBarProps) {
  if (jobs.length === 0) return null;

  return (
    <div className="shrink-0 border-t bg-card/60 px-4 py-2">
      <ScrollArea className="w-full whitespace-nowrap">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mr-1">
            Verlauf
          </span>
          {jobs.map((job) => {
            const isSelected = job.job_id === activeJobId;
            const outTarget =
              job.enhance_result?.output_path ?? job.benchmark_result?.output_dir;
            const modeTag =
              job.mode === "preview" ? "Vorschau" : job.mode === "benchmark" ? "A/B" : "Export";
            return (
              <div
                key={job.job_id}
                onClick={() => onSelectJob(job.job_id)}
                className={cn(
                  "flex cursor-pointer items-center gap-2 rounded-md border px-2.5 py-1 text-xs transition-colors",
                  isSelected
                    ? "border-primary bg-primary/15"
                    : "border-border bg-background/60 hover:bg-accent/60"
                )}
              >
                {job.status === "running" && (
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                )}
                {job.status === "completed" && (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                )}
                {(job.status === "failed" || job.status === "cancelled") && (
                  <XCircle className="h-3.5 w-3.5 text-destructive" />
                )}

                <span className="max-w-[140px] truncate font-medium">{job.filename}</span>
                <Badge variant="outline" className="text-[10px] px-1 py-0">
                  {modeTag} · {job.preset}
                </Badge>
                <span className="tabular-nums text-[11px] text-muted-foreground">
                  {job.status === "running"
                    ? `${Math.round(job.percent)}%`
                    : job.status === "completed"
                      ? `${(
                          job.enhance_result?.fps ??
                          job.benchmark_result?.enhance_result.fps ??
                          job.fps
                        ).toFixed(2)} fps`
                      : job.status}
                </span>
                {outTarget && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-5 w-5"
                    title="Ausgabe im Explorer öffnen"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRevealPath(outTarget);
                    }}
                  >
                    <FolderOpen className="h-3 w-3" />
                  </Button>
                )}
              </div>
            );
          })}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}
