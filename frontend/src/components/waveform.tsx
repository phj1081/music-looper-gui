"use client";

import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, { Region } from "wavesurfer.js/dist/plugins/regions.js";
import { Card } from "@/components/ui/card";
import { LoopingWebAudioPlayer } from "@/lib/looping-webaudio-player";

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
  wavesurferRef,
}: WaveformProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const internalWsRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const loopRegionRef = useRef<Region | null>(null);
  const [isWsReady, setIsWsReady] = useState(false);
  const currentLoopToken = `${loopStart ?? "x"}:${loopEnd ?? "x"}:${sampleRate}`;
  const [dragTimes, setDragTimes] = useState<{ start: number; end: number; token: string } | null>(null);
  const dragTimesRef = useRef<{ start: number; end: number; token: string } | null>(null);
  const dragRafRef = useRef<number | null>(null);
  const dragHideTimeoutRef = useRef<number | null>(null);
  const onReadyRef = useRef(onReady);
  const onTimeUpdateRef = useRef(onTimeUpdate);
  const onLoopChangeRef = useRef(onLoopChange);

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
    });

    ws.on("ready", () => {
      setIsWsReady(true);
      onReadyRef.current?.();
    });

    ws.on("timeupdate", (time) => {
      onTimeUpdateRef.current?.(time);
    });

    internalWsRef.current = ws;
    if (wavesurferRef) {
      wavesurferRef.current = ws;
    }

    return () => {
      setIsWsReady(false);
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
      media.pause();
      media.removeAttribute("src");
      ws.destroy();
      media.destroy();
      internalWsRef.current = null;
      if (wavesurferRef) {
        wavesurferRef.current = null;
      }
      regionsRef.current = null;
      loopRegionRef.current = null;
    };
  }, [audioUrl, wavesurferRef]);

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
      regionEl.style.borderLeft = "2px solid rgba(34, 197, 94, 0.95)";
      regionEl.style.borderRight = "2px solid rgba(34, 197, 94, 0.95)";
    }
  }, [isWsReady, loopStart, loopEnd, sampleRate]);

  useEffect(() => {
    if (!internalWsRef.current) return;

    if (isPlaying && !internalWsRef.current.isPlaying()) {
      internalWsRef.current.play();
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
      <div ref={containerRef} className="w-full" />
    </Card>
  );
}
