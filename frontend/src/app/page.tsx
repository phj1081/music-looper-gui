"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import WaveSurfer from "wavesurfer.js";
import { Music, Loader2, FolderOpen } from "lucide-react";
import { Waveform } from "@/components/waveform";
import { LoopTable } from "@/components/loop-table";
import { PlayerControls } from "@/components/player-controls";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  selectFile,
  analyzeFile,
  getAudioUrl,
  onProgress,
  getPreloadStatus,
  type LoopPoint,
  type AnalyzeResponse,
  type ProgressResponse,
} from "@/lib/api";
import { ExportMenu } from "@/components/export-menu";
import { Progress } from "@/components/ui/progress";
import { LoopEditor } from "@/components/loop-editor";

type Status = "idle" | "selecting" | "analyzing" | "ready" | "error";
type AudioHealthEventType = "contextlost" | "contextrecovered" | "contextrecoveryfailed";
type LoopingMedia = {
  setLoopPoints?: (startSeconds: number, endSeconds: number) => void;
  setLooping?: (enabled: boolean) => void;
  play?: () => Promise<void>;
};

type SortMode = "score_desc" | "length_desc" | "length_asc";

const stageMessages: Record<string, string> = {
  idle: "대기 중",
  starting: "분석 시작...",
  loading_audio: "오디오 파일 로딩 중...",
  loading_model: "모델 로딩 중...",
  preparing_chunks: "오디오 분할 중...",
  extracting_embeddings: "임베딩 추출 중...",
  finding_patterns: "패턴 분석 중...",
  computing_recurrence: "반복 행렬 계산 중...",
  enhancing_paths: "경로 강화 중...",
  detecting_beats: "박자 탐지 중...",
  checking_structure_model: "구조 모델 확인 중...",
  downloading_structure_model: "구조 모델 다운로드 중...",
  loading_structure_model: "구조 모델 로딩 중...",
  analyzing_structure: "구조 분석 중 (allin1)...",
  refining_seam: "이음새 미세조정 중...",
  completed: "완료",
  error: "오류 발생",
};

const stageProgressRanges: Record<string, { start: number; end: number }> = {
  idle: { start: 0, end: 0 },
  starting: { start: 0, end: 2 },
  loading_audio: { start: 2, end: 5 },
  loading_model: { start: 5, end: 12 },
  preparing_chunks: { start: 12, end: 20 },
  extracting_embeddings: { start: 20, end: 55 },
  computing_recurrence: { start: 55, end: 65 },
  enhancing_paths: { start: 65, end: 72 },
  finding_patterns: { start: 72, end: 80 },
  detecting_beats: { start: 80, end: 88 },
  checking_structure_model: { start: 88, end: 89 },
  downloading_structure_model: { start: 89, end: 92 },
  loading_structure_model: { start: 92, end: 93 },
  analyzing_structure: { start: 93, end: 96 },
  structure_complete: { start: 96, end: 96 },
  refining_seam: { start: 96, end: 99 },
  completed: { start: 100, end: 100 },
  error: { start: 0, end: 0 },
};

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(Math.max(value, 0), 1);
}

function calculateOverallProgressPercent(progress: ProgressResponse): number {
  const range = stageProgressRanges[progress.stage];
  const ratio =
    progress.total > 0 ? clamp01(progress.current / progress.total) : 0;

  if (!range) {
    return Math.round(ratio * 100);
  }

  if (range.start === range.end) {
    return range.start;
  }

  return Math.round(range.start + (range.end - range.start) * ratio);
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "0:00";
  }
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function formatTimeMs(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "0:00.000";
  }
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toFixed(3).padStart(6, "0")}`;
}

function formatSamples(samples: number, sampleRate: number): string {
  if (
    !Number.isFinite(samples) ||
    !Number.isFinite(sampleRate) ||
    sampleRate <= 0
  )
    return "0:00.000";
  return formatTimeMs(samples / sampleRate);
}

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [statusMessage, setStatusMessage] = useState("파일을 선택하세요");
  const [filename, setFilename] = useState<string | null>(null);
  const [audioDataUrl, setAudioDataUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalyzeResponse | null>(
    null
  );
  const [selectedLoop, setSelectedLoop] = useState<LoopPoint | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(true);
  const [isReady, setIsReady] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isExportMenuOpen, setIsExportMenuOpen] = useState(false);
  const [sortMode, setSortMode] = useState<SortMode>("score_desc");
  const [, setProgress] = useState<ProgressResponse>({
    current: 0,
    total: 0,
    stage: "idle",
  });
  const [overallProgressPercent, setOverallProgressPercent] = useState(0);
  const [preloadStatus, setPreloadStatus] = useState<
    "idle" | "loading" | "ready" | "error"
  >("idle");
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const analyzeFileRef = useRef<((path: string, name: string) => void) | null>(
    null
  );
  const currentFilePathRef = useRef<string | null>(null);

  // Tauri is always ready (no PyWebView polling needed)
  useEffect(() => {
    setIsReady(true);
  }, []);

  // Poll model preload status
  useEffect(() => {
    if (preloadStatus !== "idle" && preloadStatus !== "loading") return;

    let cancelled = false;
    const poll = async () => {
      try {
        const res = await getPreloadStatus();
        if (!cancelled) setPreloadStatus(res.status);
      } catch {
        // Server not ready yet, keep polling
      }
    };

    poll();
    const id = setInterval(poll, 1500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [preloadStatus]);

  // Setup Tauri drag-drop handler
  useEffect(() => {
    let cleanup: (() => void) | undefined;

    async function setupDragDrop() {
      try {
        const { getCurrentWebview } = await import("@tauri-apps/api/webview");
        const unlisten = await getCurrentWebview().onDragDropEvent((event) => {
          if (event.payload.type === "drop") {
            const paths = event.payload.paths;
            if (paths.length > 0) {
              const filePath = paths[0]!;
              const ext = filePath.split(".").pop()?.toLowerCase();
              if (
                ext &&
                ["mp3", "wav", "flac", "ogg", "m4a"].includes(ext)
              ) {
                const fileName =
                  filePath.split("/").pop() ??
                  filePath.split("\\").pop() ??
                  filePath;
                analyzeFileRef.current?.(filePath, fileName);
              }
            }
          }
        });
        cleanup = unlisten;
      } catch {
        // Not in Tauri environment (dev browser)
      }
    }

    setupDragDrop();
    return () => cleanup?.();
  }, []);

  const displayLoops = useMemo(() => {
    const loops = analysisResult?.loops ? [...analysisResult.loops] : [];
    const length = (loop: LoopPoint) =>
      Math.abs(loop.end_sample - loop.start_sample);

    loops.sort((a, b) => {
      if (sortMode === "length_desc") {
        const d = length(b) - length(a);
        if (d !== 0) return d;
        return (b.score ?? 0) - (a.score ?? 0);
      }
      if (sortMode === "length_asc") {
        const d = length(a) - length(b);
        if (d !== 0) return d;
        return (b.score ?? 0) - (a.score ?? 0);
      }
      // score_desc
      const d = (b.score ?? 0) - (a.score ?? 0);
      if (d !== 0) return d;
      const sr = analysisResult?.sample_rate ?? 0;
      if (sr > 0) {
        const ld = length(b) - length(a);
        if (ld !== 0) return ld;
      }
      return a.index - b.index;
    });

    return loops;
  }, [analysisResult, sortMode]);

  const handleAnalyzeFile = useCallback(async (filePath: string, name: string) => {
    setStatus("analyzing");
    setStatusMessage("분석 시작...");
    setFilename(name);
    setAudioDataUrl(null);
    setAnalysisResult(null);
    setSelectedLoop(null);
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setProgress({ current: 0, total: 0, stage: "starting" });
    setOverallProgressPercent(0);
    currentFilePathRef.current = filePath;

    // Subscribe to push-based progress events
    let unlistenProgress: (() => void) | undefined;
    let unlistenComplete: (() => void) | undefined;

    try {
      unlistenProgress = await onProgress((prog) => {
        setProgress(prog);
        setStatusMessage(stageMessages[prog.stage] || prog.stage);
        setOverallProgressPercent((prev) => {
          const next = calculateOverallProgressPercent(prog);
          return Math.max(prev, next);
        });

        if (prog.stage === "error") {
          setStatus("error");
          setStatusMessage(prog.error || "오류가 발생했습니다");
        }
      });

      // invoke analyze (blocks until sidecar completes)
      const analysis = await analyzeFile(filePath);

      // Cleanup progress listener
      unlistenProgress?.();
      unlistenProgress = undefined;

      setAnalysisResult(analysis);
      setOverallProgressPercent(100);

      // Load audio via temp file + convertFileSrc
      let audioLoaded = false;
      try {
        const audioUrl = await getAudioUrl();
        if (audioUrl) {
          setAudioDataUrl(audioUrl);
          audioLoaded = true;
        }
      } catch (err) {
        console.error("Failed to load audio:", err);
        setAudioDataUrl(null);
      }

      const firstLoop = analysis.loops[0];
      if (firstLoop) {
        setSelectedLoop(firstLoop);
        setStatusMessage(
          `${analysis.loops.length}개의 루프 포인트 발견${audioLoaded ? "" : " (오디오 로딩 실패)"}`
        );
      } else {
        setStatusMessage(
          `루프 포인트를 찾을 수 없습니다${audioLoaded ? "" : " (오디오 로딩 실패)"}`
        );
      }
      setStatus("ready");
    } catch (error) {
      setStatus("error");
      const msg = error instanceof Error ? error.message : String(error);
      console.error("Analyze error:", error);
      setStatusMessage(msg || "오류가 발생했습니다");
    } finally {
      unlistenProgress?.();
      unlistenComplete?.();
    }
  }, []);

  // Keep ref updated for drop handler
  useEffect(() => {
    analyzeFileRef.current = handleAnalyzeFile;
  }, [handleAnalyzeFile]);

  // Compatibility bridge for legacy PyWebView drop integration.
  // backend/app.py may call window.onFileDropped(...) before React hydrates.
  useEffect(() => {
    const legacyWindow = window as Window & {
      onFileDropped?: (filePath: string, filename: string) => void;
      __musicLooperPendingDrops?: Array<{ path?: string; filename?: string }>;
    };

    const isSupportedAudio = (filePath: string) => {
      const ext = filePath.split(".").pop()?.toLowerCase();
      return Boolean(ext && ["mp3", "wav", "flac", "ogg", "m4a"].includes(ext));
    };

    const processDroppedFile = (filePath: string, filename?: string) => {
      if (!filePath || !isSupportedAudio(filePath)) return;
      const resolvedName =
        filename ||
        filePath.split("/").pop() ||
        filePath.split("\\").pop() ||
        filePath;
      handleAnalyzeFile(filePath, resolvedName);
    };

    const onLegacyDropped = (event: Event) => {
      const detail = (
        event as CustomEvent<{ path?: string; filename?: string }>
      ).detail;
      processDroppedFile(detail?.path ?? "", detail?.filename);
    };

    legacyWindow.onFileDropped = processDroppedFile;
    window.addEventListener("music-looper-file-drop", onLegacyDropped);

    const pendingDrops = Array.isArray(legacyWindow.__musicLooperPendingDrops)
      ? [...legacyWindow.__musicLooperPendingDrops]
      : [];
    legacyWindow.__musicLooperPendingDrops = [];
    for (const dropped of pendingDrops) {
      processDroppedFile(dropped?.path ?? "", dropped?.filename);
    }

    return () => {
      if (legacyWindow.onFileDropped === processDroppedFile) {
        delete legacyWindow.onFileDropped;
      }
      window.removeEventListener("music-looper-file-drop", onLegacyDropped);
    };
  }, [handleAnalyzeFile]);

  const handleLoopChange = useCallback(
    (nextStartSample: number, nextEndSample: number) => {
      if (!analysisResult || !selectedLoop) return;
      const sr = analysisResult.sample_rate;

      const start = Math.max(0, Math.min(nextStartSample, nextEndSample));
      const end = Math.max(
        start,
        Math.max(nextStartSample, nextEndSample)
      );

      const updated: LoopPoint = {
        ...selectedLoop,
        start_sample: start,
        end_sample: end,
        start_time: formatSamples(start, sr),
        end_time: formatSamples(end, sr),
        duration: formatSamples(end - start, sr),
      };

      setSelectedLoop(updated);
      setAnalysisResult((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          loops: prev.loops.map((loop) =>
            loop.index === updated.index ? updated : loop
          ),
        };
      });
    },
    [analysisResult, selectedLoop]
  );

  const handleSelectFile = useCallback(async () => {
    setStatus("selecting");
    setStatusMessage("파일 선택 중...");

    try {
      const result = await selectFile();
      if (!result) {
        setStatus("idle");
        setStatusMessage("파일을 선택하세요");
        return;
      }

      await handleAnalyzeFile(result.path, result.filename);
    } catch (error) {
      setStatus("error");
      const msg = error instanceof Error ? error.message : String(error);
      console.error("Analyze error:", error);
      setStatusMessage(msg || "오류가 발생했습니다");
    }
  }, [handleAnalyzeFile]);

  const handleLoopSelect = useCallback(
    async (loop: LoopPoint) => {
      setSelectedLoop(loop);

      if (wavesurferRef.current && analysisResult) {
        const sampleRate = analysisResult.sample_rate;
        const loopStartTime =
          Math.min(loop.start_sample, loop.end_sample) / sampleRate;
        const loopEndTime =
          Math.max(loop.start_sample, loop.end_sample) / sampleRate;

        const media = wavesurferRef.current.getMediaElement() as unknown as LoopingMedia;
        media.setLoopPoints?.(loopStartTime, loopEndTime);
        media.setLooping?.(isLooping);

        const previewStart = Math.max(0, loopEndTime - 3);
        wavesurferRef.current.setTime(previewStart);
        try {
          await wavesurferRef.current.play();
          setIsPlaying(true);
        } catch (error) {
          console.warn("WaveSurfer play failed during loop select, retrying:", error);
          setStatusMessage("오디오 엔진 복구 후 재시도 중...");
          try {
            await media.play?.();
            await wavesurferRef.current.play();
            setIsPlaying(true);
            setStatusMessage("오디오 엔진 복구 완료");
          } catch (retryError) {
            console.error("Loop select playback retry failed:", retryError);
            setIsPlaying(false);
            setStatusMessage("재생 복구 실패: 재생 버튼을 다시 눌러주세요");
          }
        }
      }
    },
    [analysisResult, isLooping]
  );

  const handlePlayPause = useCallback(async () => {
    if (!wavesurferRef.current || !selectedLoop || !analysisResult) return;

    const sampleRate = analysisResult.sample_rate;
    const isActuallyPlaying = wavesurferRef.current.isPlaying();

    if (isPlaying || isActuallyPlaying) {
      wavesurferRef.current.pause();
      setIsPlaying(false);
    } else {
      const startTime =
        Math.min(selectedLoop.start_sample, selectedLoop.end_sample) /
        sampleRate;
      const endTime =
        Math.max(selectedLoop.start_sample, selectedLoop.end_sample) /
        sampleRate;

      const media = wavesurferRef.current.getMediaElement() as unknown as LoopingMedia;
      media.setLoopPoints?.(startTime, endTime);
      media.setLooping?.(isLooping);

      const previewTime = Math.max(0, endTime - 3);

      wavesurferRef.current.setTime(previewTime);
      try {
        await wavesurferRef.current.play();
        setIsPlaying(true);
      } catch (error) {
        console.warn("WaveSurfer play failed, trying media recovery:", error);
        setStatusMessage("오디오 엔진 복구 후 재시도 중...");
        try {
          await media.play?.();
          await wavesurferRef.current.play();
          setIsPlaying(true);
          setStatusMessage("오디오 엔진 복구 완료");
        } catch (retryError) {
          console.error("Playback recovery failed:", retryError);
          setIsPlaying(false);
          setStatusMessage("재생 실패: 재생 버튼을 다시 눌러주세요");
        }
      }
    }
  }, [isPlaying, selectedLoop, analysisResult, isLooping]);

  const handlePlaybackStateChange = useCallback((playing: boolean) => {
    setIsPlaying(playing);
  }, []);

  const handleAudioHealthEvent = useCallback(
    (type: AudioHealthEventType) => {
      if (status !== "ready") return;
      if (type === "contextlost") {
        setStatusMessage("오디오 엔진이 일시 중단되었습니다. 자동 복구 중...");
        return;
      }
      if (type === "contextrecovered") {
        setStatusMessage("오디오 엔진 복구 완료");
        return;
      }
      setStatusMessage("오디오 엔진 복구 실패: 재생 버튼을 다시 눌러주세요");
    },
    [status]
  );

  const handleWaveformReady = useCallback(() => {
    if (wavesurferRef.current) {
      setDuration(wavesurferRef.current.getDuration());
    }
    if (selectedLoop && analysisResult && wavesurferRef.current) {
      const sampleRate = analysisResult.sample_rate;
      const startTime =
        Math.min(selectedLoop.start_sample, selectedLoop.end_sample) /
        sampleRate;
      const endTime =
        Math.max(selectedLoop.start_sample, selectedLoop.end_sample) /
        sampleRate;

      const media = wavesurferRef.current.getMediaElement() as unknown as LoopingMedia;
      media.setLoopPoints?.(startTime, endTime);
      media.setLooping?.(isLooping);

      const previewStart = Math.max(0, endTime - 3);
      wavesurferRef.current.setTime(previewStart);
    }
  }, [selectedLoop, analysisResult, isLooping]);

  const handleTimeUpdate = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  const handleStatusChange = useCallback((message: string) => {
    setStatusMessage(message);
  }, []);

  const handleReset = useCallback(() => {
    setStatus("idle");
    setStatusMessage("파일을 선택하세요");
    setFilename(null);
    setAudioDataUrl(null);
    setAnalysisResult(null);
    setSelectedLoop(null);
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setProgress({ current: 0, total: 0, stage: "idle" });
    setOverallProgressPercent(0);
    currentFilePathRef.current = null;
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      if (status !== "ready") return;

      switch (e.key.toLowerCase()) {
        case " ":
          e.preventDefault();
          handlePlayPause();
          break;
        case "l":
          setIsLooping((prev) => !prev);
          break;
        case "e":
          setIsExportMenuOpen(true);
          break;
        case "1":
        case "2":
        case "3":
        case "4":
        case "5":
        case "6":
        case "7":
        case "8":
        case "9": {
          const index = parseInt(e.key, 10) - 1;
          const loop = displayLoops[index];
          if (loop) {
            handleLoopSelect(loop);
          }
          break;
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [status, handlePlayPause, handleLoopSelect, displayLoops]);

  if (!isReady) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto max-w-4xl px-4 py-8">
        <header className="mb-8 text-center">
          <div className="inline-flex items-center gap-3 mb-2">
            <Music className="h-8 w-8 text-primary" />
            <h1 className="text-3xl font-bold tracking-tight">Music Looper</h1>
          </div>
          <p className="text-muted-foreground">
            오디오 파일의 완벽한 루프 포인트를 자동으로 찾아줍니다
          </p>
          {preloadStatus === "loading" && (
            <p className="mt-1 text-xs text-muted-foreground animate-pulse">
              모델 준비 중...
            </p>
          )}
        </header>

        <main className="space-y-6">
          {status === "idle" && (
            <Card
              className="border-2 border-dashed p-8 text-center cursor-pointer transition-colors border-muted-foreground/25 hover:border-primary/50"
              onClick={handleSelectFile}
            >
              <FolderOpen className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-lg font-medium mb-2">
                파일을 끌어다 놓거나 클릭하세요
              </p>
              <p className="text-sm text-muted-foreground">
                지원 형식: MP3, WAV, FLAC, OGG, M4A
              </p>
            </Card>
          )}

          {(status === "selecting" || status === "analyzing") && (
            <div className="flex flex-col items-center justify-center py-16 gap-6">
              <Loader2 className="h-12 w-12 animate-spin text-primary" />
              <div className="w-full max-w-md space-y-2">
                <p className="text-lg text-muted-foreground text-center">
                  {statusMessage}
                </p>
                {status === "analyzing" && (
                  <div className="space-y-1">
                    <Progress
                      value={overallProgressPercent}
                      max={100}
                      className="h-2"
                    />
                    <p className="text-xs font-medium text-center">
                      {overallProgressPercent}%
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {status === "error" && (
            <Card className="p-6 space-y-4">
              <div className="space-y-1">
                <h2 className="text-lg font-semibold">{filename}</h2>
                <p className="text-sm text-destructive">{statusMessage}</p>
              </div>
              <div className="flex justify-end">
                <Button variant="outline" onClick={handleReset}>
                  다른 파일 선택
                </Button>
              </div>
            </Card>
          )}

          {status === "ready" && analysisResult && (
            <>
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className="text-lg font-semibold">{filename}</h2>
                  <p className="text-sm text-muted-foreground">
                    {statusMessage}
                  </p>
                  {analysisResult.enhancements && (
                    <div className="mt-2 flex flex-wrap gap-2">
                      <Badge
                        variant={
                          analysisResult.enhancements.beat_alignment.effective
                            ? "secondary"
                            : "outline"
                        }
                        className={
                          !analysisResult.enhancements.beat_alignment.enabled
                            ? "opacity-60"
                            : ""
                        }
                        title="madmom 기반 비트/마디 정렬"
                      >
                        비트정렬:{" "}
                        {analysisResult.enhancements.beat_alignment.enabled
                          ? analysisResult.enhancements.beat_alignment.effective
                            ? "ON"
                            : "ON(미적용)"
                          : "OFF"}
                      </Badge>
                      <Badge
                        variant={
                          analysisResult.enhancements.structure.effective
                            ? "secondary"
                            : "outline"
                        }
                        className={
                          !analysisResult.enhancements.structure.enabled
                            ? "opacity-60"
                            : ""
                        }
                        title="allin1 기반 구조(섹션) 분석 보정"
                      >
                        구조분석:{" "}
                        {analysisResult.enhancements.structure.enabled
                          ? analysisResult.enhancements.structure.effective
                            ? "ON"
                            : "ON(미적용)"
                          : "OFF"}
                      </Badge>
                    </div>
                  )}
                </div>
                <Button variant="outline" onClick={handleReset}>
                  다른 파일 선택
                </Button>
              </div>

              <div className="space-y-1">
                <Waveform
                  audioUrl={audioDataUrl}
                  loopStart={selectedLoop?.start_sample}
                  loopEnd={selectedLoop?.end_sample}
                  sampleRate={analysisResult.sample_rate}
                  isLooping={isLooping}
                  isPlaying={isPlaying}
                  onReady={handleWaveformReady}
                  onTimeUpdate={handleTimeUpdate}
                  onLoopChange={handleLoopChange}
                  onPlaybackStateChange={handlePlaybackStateChange}
                  onAudioHealthEvent={handleAudioHealthEvent}
                  wavesurferRef={wavesurferRef}
                />
                <div className="flex justify-end px-1">
                  <span className="text-sm font-mono text-muted-foreground">
                    {formatTime(currentTime)} / {formatTime(duration)}
                  </span>
                </div>
              </div>

              <div className="flex flex-col gap-3">
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-muted-foreground">정렬</span>
                    <Select
                      value={sortMode}
                      onValueChange={(v) => setSortMode(v as SortMode)}
                    >
                      <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="정렬 선택" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="score_desc">
                          정확도(점수) ↓
                        </SelectItem>
                        <SelectItem value="length_desc">길이 ↓</SelectItem>
                        <SelectItem value="length_asc">길이 ↑</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    초록 구간 드래그/리사이즈로 수동 편집 가능
                  </span>
                </div>

                <LoopTable
                  loops={displayLoops}
                  selectedIndex={selectedLoop?.index ?? null}
                  onSelect={handleLoopSelect}
                />
              </div>

              <LoopEditor
                key={
                  selectedLoop
                    ? `${selectedLoop.index}:${selectedLoop.start_sample}:${selectedLoop.end_sample}`
                    : "loop-editor-empty"
                }
                loop={selectedLoop}
                sampleRate={analysisResult.sample_rate}
                durationSeconds={analysisResult.duration}
                onChange={handleLoopChange}
              />

              <div className="flex items-center justify-between">
                <PlayerControls
                  isPlaying={isPlaying}
                  isLooping={isLooping}
                  disabled={!selectedLoop}
                  onPlayPause={handlePlayPause}
                  onToggleLoop={() => setIsLooping(!isLooping)}
                />
                <ExportMenu
                  loopStart={selectedLoop?.start_sample ?? 0}
                  loopEnd={selectedLoop?.end_sample ?? 0}
                  disabled={!selectedLoop}
                  open={isExportMenuOpen}
                  onOpenChange={setIsExportMenuOpen}
                  onStatusChange={handleStatusChange}
                />
              </div>
            </>
          )}
        </main>

        <footer className="mt-12 text-center text-sm text-muted-foreground">
          <a
            href="https://github.com/phj1081/music-looper-gui"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-foreground transition-colors"
          >
            Made by phj1081
          </a>
        </footer>
      </div>
    </div>
  );
}
