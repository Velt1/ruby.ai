import { getVersion } from "@tauri-apps/api/app";
import { relaunch } from "@tauri-apps/plugin-process";
import { check, type Update } from "@tauri-apps/plugin-updater";
import type { AvailableUpdateInfo, UpdateState } from "./types";

export const INITIAL_UPDATE_STATE: UpdateState = {
  phase: "idle",
  available: null,
  downloadedBytes: 0,
  totalBytes: null,
  error: null,
  lastCheckedAt: null,
};

export interface AppUpdateService {
  getAppVersion(): Promise<string>;
  checkForUpdate(options?: { silent?: boolean }): Promise<UpdateState>;
  downloadAndInstall(onState: (state: UpdateState) => void): Promise<UpdateState>;
}

export class TauriUpdateService implements AppUpdateService {
  private pendingUpdate: Update | null = null;
  private currentState: UpdateState = { ...INITIAL_UPDATE_STATE };

  async getAppVersion(): Promise<string> {
    return getVersion();
  }

  private async replacePendingUpdate(next: Update | null): Promise<void> {
    if (this.pendingUpdate && this.pendingUpdate !== next) {
      try {
        await this.pendingUpdate.close();
      } catch {
        // Ignore cleanup errors on stale Update resource handles.
      }
    }
    this.pendingUpdate = next;
  }

  async checkForUpdate(options?: { silent?: boolean }): Promise<UpdateState> {
    const silent = options?.silent ?? false;
    try {
      const update = await check();
      if (update) {
        await this.replacePendingUpdate(update);
        const available: AvailableUpdateInfo = {
          version: update.version,
          currentVersion: update.currentVersion,
          date: update.date,
          body: update.body,
        };
        this.currentState = {
          phase: "available",
          available,
          downloadedBytes: 0,
          totalBytes: null,
          error: null,
          lastCheckedAt: Date.now(),
        };
        return this.currentState;
      }

      await this.replacePendingUpdate(null);
      this.currentState = {
        phase: silent ? "idle" : "up-to-date",
        available: null,
        downloadedBytes: 0,
        totalBytes: null,
        error: null,
        lastCheckedAt: Date.now(),
      };
      return this.currentState;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (silent) {
        this.currentState = {
          ...this.currentState,
          phase: this.currentState.available ? "available" : "idle",
          lastCheckedAt: Date.now(),
        };
        return this.currentState;
      }
      this.currentState = {
        phase: "error",
        available: this.currentState.available,
        downloadedBytes: 0,
        totalBytes: null,
        error: message,
        lastCheckedAt: Date.now(),
      };
      return this.currentState;
    }
  }

  async downloadAndInstall(onState: (state: UpdateState) => void): Promise<UpdateState> {
    let target = this.pendingUpdate;
    if (!target) {
      const checked = await check();
      if (!checked) {
        this.currentState = {
          phase: "up-to-date",
          available: null,
          downloadedBytes: 0,
          totalBytes: null,
          error: null,
          lastCheckedAt: Date.now(),
        };
        onState(this.currentState);
        return this.currentState;
      }
      await this.replacePendingUpdate(checked);
      target = checked;
    }

    const availableInfo: AvailableUpdateInfo = {
      version: target.version,
      currentVersion: target.currentVersion,
      date: target.date,
      body: target.body,
    };

    let downloaded = 0;
    let total: number | null = null;

    this.currentState = {
      phase: "downloading",
      available: availableInfo,
      downloadedBytes: 0,
      totalBytes: null,
      error: null,
      lastCheckedAt: this.currentState.lastCheckedAt,
    };
    onState(this.currentState);

    try {
      await target.downloadAndInstall((event) => {
        if (event.event === "Started") {
          total = event.data.contentLength ?? null;
          this.currentState = {
            ...this.currentState,
            phase: "downloading",
            downloadedBytes: 0,
            totalBytes: total,
          };
          onState(this.currentState);
        } else if (event.event === "Progress") {
          downloaded += event.data.chunkLength;
          this.currentState = {
            ...this.currentState,
            phase: "downloading",
            downloadedBytes: downloaded,
            totalBytes: total,
          };
          onState(this.currentState);
        } else if (event.event === "Finished") {
          this.currentState = {
            ...this.currentState,
            phase: "installing",
            downloadedBytes: total ?? downloaded,
            totalBytes: total,
          };
          onState(this.currentState);
        }
      });

      this.currentState = {
        ...this.currentState,
        phase: "installing",
      };
      onState(this.currentState);
      await relaunch();
      return this.currentState;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.currentState = {
        phase: "error",
        available: availableInfo,
        downloadedBytes: 0,
        totalBytes: null,
        error: message,
        lastCheckedAt: this.currentState.lastCheckedAt,
      };
      onState(this.currentState);
      return this.currentState;
    }
  }
}

export const appUpdater: AppUpdateService = new TauriUpdateService();
