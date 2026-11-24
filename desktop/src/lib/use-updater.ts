import { useCallback, useEffect, useRef, useState } from "react";
import type { UpdateState } from "./types";
import { appUpdater, INITIAL_UPDATE_STATE, type AppUpdateService } from "./updater";

export interface UseAppUpdaterOptions {
  service?: AppUpdateService;
  enabled?: boolean;
  initialDelayMs?: number;
  pollIntervalMs?: number;
  minFocusCheckIntervalMs?: number;
}

export interface UseAppUpdaterResult {
  appVersion: string | null;
  updateState: UpdateState;
  showNotification: boolean;
  checkForUpdates: (silent?: boolean) => Promise<void>;
  installUpdate: () => Promise<void>;
  dismissNotification: () => void;
  reopenNotification: () => void;
}

const DEFAULT_INITIAL_DELAY_MS = 2_000;
const DEFAULT_POLL_INTERVAL_MS = 5 * 60 * 1_000;
const DEFAULT_FOCUS_THROTTLE_MS = 60 * 1_000;

export function useAppUpdater(options: UseAppUpdaterOptions = {}): UseAppUpdaterResult {
  const {
    service = appUpdater,
    enabled = true,
    initialDelayMs = DEFAULT_INITIAL_DELAY_MS,
    pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
    minFocusCheckIntervalMs = DEFAULT_FOCUS_THROTTLE_MS,
  } = options;

  const [appVersion, setAppVersion] = useState<string | null>(null);
  const [updateState, setUpdateState] = useState<UpdateState>(INITIAL_UPDATE_STATE);
  const [dismissedVersion, setDismissedVersion] = useState<string | null>(null);
  const [notificationForcedOpen, setNotificationForcedOpen] = useState(false);

  const inFlightRef = useRef(false);
  const lastAttemptAtRef = useRef(0);
  const phaseRef = useRef(updateState.phase);
  const dismissedVersionRef = useRef(dismissedVersion);
  phaseRef.current = updateState.phase;
  dismissedVersionRef.current = dismissedVersion;

  useEffect(() => {
    let mounted = true;
    void service
      .getAppVersion()
      .then((ver) => {
        if (mounted) setAppVersion(ver);
      })
      .catch(() => {});
    return () => {
      mounted = false;
    };
  }, [service]);

  const checkForUpdates = useCallback(
    async (silent = false) => {
      if (!enabled && silent) return;
      if (inFlightRef.current) return;
      if (phaseRef.current === "downloading" || phaseRef.current === "installing") {
        return;
      }

      inFlightRef.current = true;
      lastAttemptAtRef.current = Date.now();

      if (!silent) {
        setDismissedVersion(null);
        setNotificationForcedOpen(false);
        setUpdateState((prev) => ({
          ...prev,
          phase: "checking",
          error: null,
        }));
      }

      try {
        const next = await service.checkForUpdate({ silent });
        setUpdateState(next);
        if (
          next.phase === "available" &&
          next.available &&
          next.available.version !== dismissedVersionRef.current
        ) {
          setNotificationForcedOpen(true);
        }
      } finally {
        inFlightRef.current = false;
      }
    },
    [enabled, service]
  );

  const installUpdate = useCallback(async () => {
    setNotificationForcedOpen(true);
    await service.downloadAndInstall((nextState) => {
      setUpdateState(nextState);
    });
  }, [service]);

  const dismissNotification = useCallback(() => {
    const version = updateState.available?.version ?? null;
    if (version) {
      setDismissedVersion(version);
    }
    setNotificationForcedOpen(false);
    setUpdateState((prev) => {
      if (prev.phase === "error" && prev.available) {
        return { ...prev, phase: "available", error: null };
      }
      return prev;
    });
  }, [updateState.available?.version]);

  const reopenNotification = useCallback(() => {
    setDismissedVersion(null);
    setNotificationForcedOpen(true);
  }, []);

  useEffect(() => {
    if (!enabled) return;

    const initialTimer = window.setTimeout(() => {
      void checkForUpdates(true);
    }, initialDelayMs);

    const pollTimer = window.setInterval(() => {
      void checkForUpdates(true);
    }, pollIntervalMs);

    const handleFocusOrVisible = () => {
      if (document.visibilityState !== "visible") return;
      if (Date.now() - lastAttemptAtRef.current < minFocusCheckIntervalMs) return;
      void checkForUpdates(true);
    };

    window.addEventListener("focus", handleFocusOrVisible);
    document.addEventListener("visibilitychange", handleFocusOrVisible);

    return () => {
      window.clearTimeout(initialTimer);
      window.clearInterval(pollTimer);
      window.removeEventListener("focus", handleFocusOrVisible);
      document.removeEventListener("visibilitychange", handleFocusOrVisible);
    };
  }, [checkForUpdates, enabled, initialDelayMs, minFocusCheckIntervalMs, pollIntervalMs]);

  const showNotification = (() => {
    if (updateState.phase === "downloading" || updateState.phase === "installing") {
      return true;
    }
    if (updateState.phase === "available" && updateState.available) {
      return (
        notificationForcedOpen || updateState.available.version !== dismissedVersion
      );
    }
    if (updateState.phase === "error" && updateState.available && notificationForcedOpen) {
      return true;
    }
    return false;
  })();

  return {
    appVersion,
    updateState,
    showNotification,
    checkForUpdates,
    installUpdate,
    dismissNotification,
    reopenNotification,
  };
}
