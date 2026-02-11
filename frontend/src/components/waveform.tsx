"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, { Region } from "wavesurfer.js/dist/plugins/regions.js";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { LoopingWebAudioPlayer } from "@/lib/looping-webaudio-player";

type AudioHealthEventType = "contextlost" | "contextrecovered" | "contextrecoveryfailed";

function formatTimeMs(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00.000";
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toFixed(3).padStart(6, "0")}`;
}

interface WaveformProps {
  audioUrl: string | null;
  loopStart?: number | undefined;
  loopEnd?: number | undefined;
  sampleRate?: number | undefined;
  isLooping: boolean;
  isPlaying: boolean;
  onReady?: (() => void) | undefined;
  onTimeUpdate?: ((time: number) => void) | undefined;
  onLoopChange?: ((loopStartSample: number, loopEndSample: number) => void) | undefined;
  onPlaybackStateChange?: ((playing: boolean) => void) | undefined;
  onAudioHealthEvent?: ((type: AudioHealthEventType) => void) | undefined;
  wavesurferRef?: React.RefObject<WaveSurfer | null> | undefined;
}

export function Waveform({
  audioUrl,
  loopStart,
  loopEnd,
  sampleRate = 44100,
  isLooping,
  isPlaying,
  onReady,
  onTimeUpdate,
  onLoopChange,
  onPlaybackStateChange,
  onAudioHealthEvent,
  wavesurferRef,
}: WaveformProps) {
  const ZOOM_MIN = 0;
  const ZOOM_MAX = 2000;
  const ZOOM_STEP = 50;

  const containerRef = useRef<HTMLDivElement>(null);
  const internalWsRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const loopRegionRef = useRef<Region | null>(null);
  const [isWsReady, setIsWsReady] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [zoomPxPerSec, setZoomPxPerSec] = useState<number>(0);
  const currentLoopToken = `${loopStart ?? "x"}:${loopEnd ?? "x"}:${sampleRate}`;
  const [dragTimes, setDragTimes] = useState<{ start: number; end: number; token: string } | null>(null);
  const dragTimesRef = useRef<{ start: number; end: number; token: string } | null>(null);
  const dragRafRef = useRef<number | null>(null);
  const dragHideTimeoutRef = useRef<number | null>(null);
  const onReadyRef = useRef(onReady);
  const onTimeUpdateRef = useRef(onTimeUpdate);
  const onLoopChangeRef = useRef(onLoopChange);
  const onPlaybackStateChangeRef = useRef(onPlaybackStateChange);
  const onAudioHealthEventRef = useRef(onAudioHealthEvent);

  const clampZoom = useCallback(
    (value: number) => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, Math.round(value))),
    [ZOOM_MAX, ZOOM_MIN],
  );

  const zoomLabel = useMemo(() => {
    if (zoomPxPerSec <= 0) return "맞춤";
    return `${zoomPxPerSec}px/s`;
  }, [zoomPxPerSec]);

  useEffect(() => {
    onReadyRef.current = onReady;
  }, [onReady]);

  useEffect(() => {
    onTimeUpdateRef.current = onTimeUpdate;
  }, [onTimeUpdate]);

  useEffect(() => {
    onLoopChangeRef.current = onLoopChange;
  }, [onLoopChange]);

  useEffect(() => {
    onPlaybackStateChangeRef.current = onPlaybackStateChange;
  }, [onPlaybackStateChange]);

  useEffect(() => {
    onAudioHealthEventRef.current = onAudioHealthEvent;
  }, [onAudioHealthEvent]);

  useEffect(() => {
    if (!containerRef.current || !audioUrl) return;

    const regions = RegionsPlugin.create();
    regionsRef.current = regions;

    const media = new LoopingWebAudioPlayer();

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#94a3b8",
      progressColor: "#3b82f6",
      cursorColor: "#f97316",
      cursorWidth: 2,
      height: 120,
      normalize: true,
      plugins: [regions],
      media: media as unknown as HTMLMediaElement,
    });

    void ws.load(audioUrl).catch((err) => {
      // wavesurfer aborts its internal fetch on cleanup/reload (common in dev/HMR).
      // Treat AbortError as a benign cancellation to avoid Next.js error overlay noise.
      if (err instanceof DOMException && err.name === "AbortError") return;
      if (typeof err === "object" && err && "name" in err && (err as { name?: unknown }).name === "AbortError") return;
      if (typeof err === "object" && err && "message" in err && typeof (err as { message?: unknown }).message === "string") {
        const message = (err as { message: string }).message.toLowerCase();
        if (message.includes("fetch is aborted") || message.includes("aborted")) return;
      }
      console.error("WaveSurfer load error:", err);
      setLoadError("오디오/파형 로딩 실패");
    });

    const handleWsReady = () => {
      setLoadError(null);
      setIsWsReady(true);
      onReadyRef.current?.();
    };

    const handleWsTimeUpdate = (time: number) => {
      onTimeUpdateRef.current?.(time);
    };

    const handleWsError = (err: Error) => {
      console.error("WaveSurfer error:", err);
      setLoadError("오디오/파형 로딩 실패");
      onPlaybackStateChangeRef.current?.(false);
    };

    const handleWsPlay = () => {
      onPlaybackStateChangeRef.current?.(true);
    };

    const handleWsPause = () => {
      onPlaybackStateChangeRef.current?.(false);
    };

    const handleWsFinish = () => {
      onPlaybackStateChangeRef.current?.(false);
    };

    const handleContextLost = () => {
      onAudioHealthEventRef.current?.("contextlost");
      onPlaybackStateChangeRef.current?.(false);
    };

    const handleContextRecovered = () => {
      onAudioHealthEventRef.current?.("contextrecovered");
    };

    const handleContextRecoveryFailed = () => {
      onAudioHealthEventRef.current?.("contextrecoveryfailed");
      onPlaybackStateChangeRef.current?.(false);
    };

    ws.on("ready", handleWsReady);
    ws.on("timeupdate", handleWsTimeUpdate);
    ws.on("error", handleWsError);
    ws.on("play", handleWsPlay);
    ws.on("pause", handleWsPause);
    ws.on("finish", handleWsFinish);
    media.addEventListener("contextlost", handleContextLost as EventListener);
    media.addEventListener("contextrecovered", handleContextRecovered as EventListener);
    media.addEventListener("contextrecoveryfailed", handleContextRecoveryFailed as EventListener);

    internalWsRef.current = ws;
    if (wavesurferRef) {
      wavesurferRef.current = ws;
    }

    return () => {
      setIsWsReady(false);
      setLoadError(null);
      setDragTimes(null);
      dragTimesRef.current = null;
      if (dragRafRef.current !== null) {
        cancelAnimationFrame(dragRafRef.current);
        dragRafRef.current = null;
      }
      if (dragHideTimeoutRef.current !== null) {
        window.clearTimeout(dragHideTimeoutRef.current);
        dragHideTimeoutRef.current = null;
      }
      ws.un("ready", handleWsReady);
      ws.un("timeupdate", handleWsTimeUpdate);
      ws.un("error", handleWsError);
      ws.un("play", handleWsPlay);
      ws.un("pause", handleWsPause);
      ws.un("finish", handleWsFinish);
      media.removeEventListener("contextlost", handleContextLost as EventListener);
      media.removeEventListener("contextrecovered", handleContextRecovered as EventListener);
      media.removeEventListener("contextrecoveryfailed", handleContextRecoveryFailed as EventListener);
      media.pause();
      media.removeAttribute("src");
      ws.destroy();
      media.destroy();
      internalWsRef.current = null;
      if (wavesurferRef) {
        wavesurferRef.current = null;
      }
      if (audioUrl.startsWith("blob:")) {
        URL.revokeObjectURL(audioUrl);
      }
      regionsRef.current = null;
      loopRegionRef.current = null;
    };
  }, [audioUrl, wavesurferRef]);

  useEffect(() => {
    if (!isWsReady) return;
    const ws = internalWsRef.current;
    if (!ws) return;
    ws.zoom(clampZoom(zoomPxPerSec));
  }, [clampZoom, isWsReady, zoomPxPerSec]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      const direction = e.deltaY > 0 ? -1 : 1;
      setZoomPxPerSec((prev) => clampZoom(prev + direction * ZOOM_STEP));
    };

    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [clampZoom]);

  const scheduleDragTimesUpdate = (start: number, end: number, token: string) => {
    dragTimesRef.current = { start, end, token };
    if (dragRafRef.current !== null) return;
    dragRafRef.current = requestAnimationFrame(() => {
      dragRafRef.current = null;
      if (dragTimesRef.current) {
        setDragTimes(dragTimesRef.current);
      }
    });
  };

  // Loop points & looping mode (gapless WebAudio loop)
  useEffect(() => {
    if (!isWsReady || loopStart === undefined || loopEnd === undefined || !sampleRate) return;

    const ws = internalWsRef.current;
    if (!ws) return;

    const rawStartTime = loopStart / sampleRate;
    const rawEndTime = loopEnd / sampleRate;
    if (!Number.isFinite(rawStartTime) || !Number.isFinite(rawEndTime)) return;

    const duration = ws.getDuration();
    const startTime = duration > 0 ? Math.max(0, Math.min(Math.min(rawStartTime, rawEndTime), duration)) : Math.max(0, Math.min(rawStartTime, rawEndTime));
    const endTime = duration > 0 ? Math.max(0, Math.min(Math.max(rawStartTime, rawEndTime), duration)) : Math.max(0, Math.max(rawStartTime, rawEndTime));

    const media = ws.getMediaElement() as unknown as Partial<LoopingWebAudioPlayer>;
    if (typeof media.setLoopPoints === "function") {
      media.setLoopPoints(startTime, endTime);
    }
    if (typeof media.setLooping === "function") {
      media.setLooping(isLooping);
    }
  }, [isWsReady, loopStart, loopEnd, sampleRate, isLooping]);

  // 루프 영역 업데이트 - WaveSurfer가 ready 상태일 때만 실행
  useEffect(() => {
    if (!isWsReady || !regionsRef.current || loopStart === undefined || loopEnd === undefined || !sampleRate) {
      return;
    }

    const token = `${loopStart}:${loopEnd}:${sampleRate}`;

    if (dragHideTimeoutRef.current !== null) {
      window.clearTimeout(dragHideTimeoutRef.current);
      dragHideTimeoutRef.current = null;
    }
    dragTimesRef.current = null;

    // 기존 리전 제거
    if (loopRegionRef.current) {
      loopRegionRef.current.remove();
      loopRegionRef.current = null;
    }

    const rawStartTime = loopStart / sampleRate;
    const rawEndTime = loopEnd / sampleRate;

    if (!Number.isFinite(rawStartTime) || !Number.isFinite(rawEndTime)) return;

    let startTime = Math.min(rawStartTime, rawEndTime);
    let endTime = Math.max(rawStartTime, rawEndTime);

    const duration = internalWsRef.current?.getDuration();
    if (duration && Number.isFinite(duration) && duration > 0) {
      startTime = Math.max(0, Math.min(startTime, duration));
      endTime = Math.max(0, Math.min(endTime, duration));
    }

    // 최소 길이 보장 (너무 짧으면 '구간'이 아닌 마커처럼 보임)
    const minRegionSeconds = 0.02;
    if (endTime - startTime < minRegionSeconds) {
      endTime = duration && duration > 0 ? Math.min(startTime + minRegionSeconds, duration) : startTime + minRegionSeconds;
    }

    loopRegionRef.current = regionsRef.current.addRegion({
      start: startTime,
      end: endTime,
      color: "rgba(34, 197, 94, 0.25)",
      drag: true,
      resize: true,
      resizeStart: true,
      resizeEnd: true,
      minLength: 0.02,
    });

    // Live time display while dragging/resizing
    loopRegionRef.current.on("update", () => {
      const region = loopRegionRef.current;
      if (!region) return;
      scheduleDragTimesUpdate(region.start, region.end, token);
    });

    // Propagate manual edits back to parent (sample-accurate export)
    loopRegionRef.current.on("update-end", () => {
      const region = loopRegionRef.current;
      if (!region) return;
      const sr = sampleRate;
      if (!Number.isFinite(sr) || sr <= 0) return;

      scheduleDragTimesUpdate(region.start, region.end, token);
      if (dragHideTimeoutRef.current !== null) {
        window.clearTimeout(dragHideTimeoutRef.current);
      }
      dragHideTimeoutRef.current = window.setTimeout(() => {
        setDragTimes(null);
        dragTimesRef.current = null;
      }, 700);

      const nextStart = Math.round(region.start * sr);
      const nextEnd = Math.round(region.end * sr);
      onLoopChangeRef.current?.(nextStart, nextEnd);
    });

    // 구간의 시작/끝이 확실하게 보이도록 테두리 라인을 추가
    const regionEl = loopRegionRef.current.element;
    if (regionEl) {
      const handleWidthPx = 22;
      const handleBg = "rgba(34, 197, 94, 0.10)";
      const handleBorder = "3px solid rgba(34, 197, 94, 0.95)";

      regionEl.style.borderLeft = handleBorder;
      regionEl.style.borderRight = handleBorder;
      regionEl.style.boxSizing = "border-box";

      const leftHandle = regionEl.querySelector<HTMLElement>('[part*="region-handle-left"]');
      const rightHandle = regionEl.querySelector<HTMLElement>('[part*="region-handle-right"]');

      if (leftHandle) {
        leftHandle.style.width = `${handleWidthPx}px`;
        leftHandle.style.borderLeft = handleBorder;
        leftHandle.style.backgroundColor = handleBg;
        leftHandle.style.touchAction = "none";
      }

      if (rightHandle) {
        rightHandle.style.width = `${handleWidthPx}px`;
        rightHandle.style.borderRight = handleBorder;
        rightHandle.style.backgroundColor = handleBg;
        rightHandle.style.touchAction = "none";
      }
    }
  }, [isWsReady, loopStart, loopEnd, sampleRate]);

  useEffect(() => {
    if (!internalWsRef.current) return;

    if (isPlaying && !internalWsRef.current.isPlaying()) {
      const ws = internalWsRef.current;
      const recoverAndPlay = async () => {
        try {
          await ws.play();
        } catch (error) {
          console.warn("WaveSurfer play failed, retrying with media recovery", error);
          const media = ws.getMediaElement() as unknown as Partial<LoopingWebAudioPlayer>;
          if (typeof media.play === "function") {
            try {
              await media.play();
            } catch (mediaError) {
              console.warn("Direct media.play recovery failed:", mediaError);
            }
          }
          try {
            await ws.play();
          } catch (retryError) {
            console.error("WaveSurfer retry play failed:", retryError);
            onAudioHealthEventRef.current?.("contextrecoveryfailed");
            onPlaybackStateChangeRef.current?.(false);
          }
        }
      };
      void recoverAndPlay();
    } else if (!isPlaying && internalWsRef.current.isPlaying()) {
      internalWsRef.current.pause();
    }
  }, [isPlaying]);

  if (!audioUrl) {
    return (
      <Card className="h-[150px] flex items-center justify-center bg-muted/30">
        <p className="text-muted-foreground">파일을 업로드하면 파형이 표시됩니다</p>
      </Card>
    );
  }

  const liveDragTimes = dragTimes?.token === currentLoopToken ? dragTimes : null;
  const labelStartSeconds =
    liveDragTimes?.start ?? (loopStart !== undefined && sampleRate ? loopStart / sampleRate : undefined);
  const labelEndSeconds =
    liveDragTimes?.end ?? (loopEnd !== undefined && sampleRate ? loopEnd / sampleRate : undefined);
  const labelDurationSeconds =
    labelStartSeconds !== undefined && labelEndSeconds !== undefined ? Math.max(0, labelEndSeconds - labelStartSeconds) : undefined;
  const labelTitle =
    labelStartSeconds !== undefined && labelEndSeconds !== undefined && sampleRate
      ? `start=${Math.round(labelStartSeconds * sampleRate)} / end=${Math.round(labelEndSeconds * sampleRate)} samples`
      : undefined;

  return (
    <Card className="relative p-4">
      {loadError && (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-md bg-background/80">
          <p className="text-sm text-muted-foreground">{loadError}</p>
        </div>
      )}
      {labelStartSeconds !== undefined && labelEndSeconds !== undefined && (
        <div className="pointer-events-none absolute right-3 top-3 z-10" title={labelTitle}>
          <div
            className={`rounded-md border px-2 py-1 text-xs font-mono backdrop-blur ${
              dragTimes ? "bg-primary/10 text-foreground" : "bg-background/70 text-muted-foreground"
            }`}
          >
            {formatTimeMs(labelStartSeconds)} &rarr; {formatTimeMs(labelEndSeconds)}{" "}
            {labelDurationSeconds !== undefined && `(${formatTimeMs(labelDurationSeconds)})`}
          </div>
        </div>
      )}

      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">확대</span>
        <Button
          type="button"
          variant="outline"
          size="icon-xs"
          onClick={() => setZoomPxPerSec((prev) => clampZoom(prev - ZOOM_STEP))}
          title="축소"
        >
          -
        </Button>
        <input
          type="range"
          min={ZOOM_MIN}
          max={ZOOM_MAX}
          step={10}
          value={zoomPxPerSec}
          onChange={(e) => setZoomPxPerSec(clampZoom(Number(e.target.value)))}
          className="h-2 w-40 cursor-pointer accent-primary"
          aria-label="파형 확대/축소"
        />
        <Button
          type="button"
          variant="outline"
          size="icon-xs"
          onClick={() => setZoomPxPerSec((prev) => clampZoom(prev + ZOOM_STEP))}
          title="확대"
        >
          +
        </Button>
        <Button type="button" variant="ghost" size="xs" onClick={() => setZoomPxPerSec(0)} title="맞춤(전체 보기)">
          맞춤
        </Button>
        <span className="text-xs font-mono text-muted-foreground">{zoomLabel}</span>
        <span className="text-xs text-muted-foreground">Ctrl/⌘ + 스크롤로도 확대/축소</span>
      </div>
      <div ref={containerRef} className="w-full" />
    </Card>
  );
}
