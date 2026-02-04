"use client";

import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, { Region } from "wavesurfer.js/dist/plugins/regions.js";
import { Card } from "@/components/ui/card";

interface WaveformProps {
  audioUrl: string | null;
  loopStart?: number;
  loopEnd?: number;
  sampleRate?: number;
  isPlaying: boolean;
  onReady?: () => void;
  onTimeUpdate?: (time: number) => void;
  wavesurferRef?: React.RefObject<WaveSurfer | null>;
}

export function Waveform({
  audioUrl,
  loopStart,
  loopEnd,
  sampleRate = 44100,
  isPlaying,
  onReady,
  onTimeUpdate,
  wavesurferRef,
}: WaveformProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const internalWsRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const loopRegionRef = useRef<Region | null>(null);
  const [isWsReady, setIsWsReady] = useState(false);

  useEffect(() => {
    if (!containerRef.current || !audioUrl) return;

    const regions = RegionsPlugin.create();
    regionsRef.current = regions;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#94a3b8",
      progressColor: "#3b82f6",
      cursorColor: "#f97316",
      cursorWidth: 2,
      height: 120,
      normalize: true,
      plugins: [regions],
    });

    ws.load(audioUrl);

    ws.on("ready", () => {
      setIsWsReady(true);
      onReady?.();
    });

    ws.on("timeupdate", (time) => {
      onTimeUpdate?.(time);
    });

    internalWsRef.current = ws;
    if (wavesurferRef) {
      wavesurferRef.current = ws;
    }

    return () => {
      setIsWsReady(false);
      ws.destroy();
      internalWsRef.current = null;
      if (wavesurferRef) {
        wavesurferRef.current = null;
      }
      regionsRef.current = null;
      loopRegionRef.current = null;
    };
  }, [audioUrl, onReady, onTimeUpdate, wavesurferRef]);

  // 루프 영역 업데이트 - WaveSurfer가 ready 상태일 때만 실행
  useEffect(() => {
    if (!isWsReady || !regionsRef.current || loopStart === undefined || loopEnd === undefined || !sampleRate) {
      return;
    }

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
      drag: false,
      resize: false,
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

  return (
    <Card className="p-4">
      <div ref={containerRef} className="w-full" />
    </Card>
  );
}
