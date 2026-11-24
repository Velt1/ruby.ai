import { Film, FolderOpen, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface VideoDropzoneProps {
  isDragging: boolean;
  onPickVideo: () => void;
}

export function VideoDropzone({ isDragging, onPickVideo }: VideoDropzoneProps) {
  return (
    <div
      onClick={onPickVideo}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onPickVideo();
        }
      }}
      className={cn(
        "flex h-full min-h-[380px] w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center transition-colors",
        isDragging
          ? "border-primary bg-primary/10"
          : "border-border/80 bg-card/30 hover:border-primary/50 hover:bg-card/50"
      )}
    >
      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/15 text-primary ring-1 ring-primary/25">
        {isDragging ? <Upload className="h-6 w-6" /> : <Film className="h-6 w-6" />}
      </div>
      <h2 className="text-base font-semibold tracking-tight">
        {isDragging ? "Video hier ablegen" : "4K-Video laden oder hierher ziehen"}
      </h2>
      <p className="mt-1.5 max-w-md text-xs text-muted-foreground leading-relaxed">
        Unterstützt MP4, MOV (HEVC/H.264), MKV und WebM. Automatische Orientierung,
        1080p-Lanczos-Supersampling und 1×-AI-Restauration für Instagram Reels & TikTok.
      </p>
      <Button
        type="button"
        variant="secondary"
        size="sm"
        className="mt-5 gap-2"
        onClick={(e) => {
          e.stopPropagation();
          onPickVideo();
        }}
      >
        <FolderOpen className="h-3.5 w-3.5" />
        Videodatei auswählen
      </Button>
    </div>
  );
}
