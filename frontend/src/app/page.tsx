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
  analyzeFileAsync,
  getProgress,
  getAnalysisResult,
  getAudioBase64,
  type LoopPoint,
  type AnalyzeResponse,
  type ProgressResponse,
} from "@/lib/api";
import { ExportMenu } from "@/components/export-menu";
import { Progress } from "@/components/ui/progress";
import { LoopEditor } from "@/components/loop-editor";

type Status = "idle" | "selecting" | "analyzing" | "ready" | "error";
type LoopingMedia = {
  setLoopPoints?: (startSeconds: number, endSeconds: number) => void;
  setLooping?: (enabled: boolean) => void;
};

type SortMode = "score_desc" | "length_desc" | "length_asc";

const stageMessages: Record<string, string> = {
  idle: "대기 중",
  starting: "분석 시작...",
  loading_model: "모델 로딩 중...",
  preparing_chunks: "오디오 분할 중...",
  extracting_embeddings: "임베딩 추출 중...",
  finding_patterns: "패턴 분석 중...",
  computing_recurrence: "반복 행렬 계산 중...",
  enhancing_paths: "경로 강화 중...",
  detecting_beats: "박자 탐지 중...",
  analyzing_structure: "구조 분석 중 (allin1)...",
  refining_seam: "이음새 미세조정 중...",
  completed: "완료",
  error: "오류 발생",
};

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
  if (!Number.isFinite(samples) || !Number.isFinite(sampleRate) || sampleRate <= 0) return "0:00.000";
  return formatTimeMs(samples / sampleRate);
}

// Global handler for file drop from PyWebView
declare global {
  interface Window {
    onFileDropped?: ((filePath: string, filename: string) => void) | undefined;
  }
}

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [statusMessage, setStatusMessage] = useState("파일을 선택하세요");
  const [filename, setFilename] = useState<string | null>(null);
  const [audioDataUrl, setAudioDataUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalyzeResponse | null>(null);
  const [selectedLoop, setSelectedLoop] = useState<LoopPoint | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(true);
  const [isReady, setIsReady] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isExportMenuOpen, setIsExportMenuOpen] = useState(false);
  const [sortMode, setSortMode] = useState<SortMode>("score_desc");
  const [progress, setProgress] = useState<ProgressResponse>({
    current: 0,
    total: 0,
    stage: "idle",
  });
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const analyzeFileRef = useRef<((path: string, name: string) => void) | null>(null);
  // Store current file path for re-analysis
  const currentFilePathRef = useRef<string | null>(null);

  const displayLoops = useMemo(() => {
    const loops = analysisResult?.loops ? [...analysisResult.loops] : [];
    const sr = analysisResult?.sample_rate ?? 0;
    const length = (loop: LoopPoint) => Math.abs(loop.end_sample - loop.start_sample);

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
      // Tie-breaker: prefer longer loop slightly
      if (sr > 0) {
        const ld = length(b) - length(a);
        if (ld !== 0) return ld;
      }
      return a.index - b.index;
    });

    return loops;
  }, [analysisResult, sortMode]);

  // Wait for PyWebView to be ready and setup drop handler
  useEffect(() => {
    let attempts = 0;
    const maxAttempts = 30; // 3 seconds max wait

    const checkPyWebView = () => {
      if (window.pywebview) {
        setIsReady(true);
      } else if (attempts < maxAttempts) {
        attempts++;
        setTimeout(checkPyWebView, 100);
      } else {
        // Timeout: allow UI to render for development/testing
        // (API calls will still fail without pywebview)
        console.warn("PyWebView not detected. Running in browser-only mode.");
        setIsReady(true);
      }
    };
    checkPyWebView();

    // Setup global drop handler
    window.onFileDropped = (filePath: string, filename: string) => {
      if (analyzeFileRef.current) {
        analyzeFileRef.current(filePath, filename);
      }
    };

    return () => {
      window.onFileDropped = undefined;
    };
  }, []);

  const handleAnalyzeFile = useCallback(async (filePath: string, name: string) => {
    setStatus("analyzing");
    setStatusMessage("분석 시작...");
    setFilename(name);
    setProgress({ current: 0, total: 0, stage: "starting" });
    currentFilePathRef.current = filePath;

    try {
      // Start async analysis (backend auto-detects available enhancements)
      await analyzeFileAsync(filePath);

      // Poll for progress
      const pollProgress = async () => {
        try {
          const prog = await getProgress();
          setProgress(prog);
          setStatusMessage(stageMessages[prog.stage] || prog.stage);

          if (prog.stage === "completed") {
            // Get the result
            const analysis = await getAnalysisResult();
            if (analysis) {
              setAnalysisResult(analysis);

              const base64 = await getAudioBase64();
              if (base64) {
                setAudioDataUrl(`data:audio/wav;base64,${base64}`);
              }

              const firstLoop = analysis.loops[0];
              if (firstLoop) {
                setSelectedLoop(firstLoop);
                setStatusMessage(`${analysis.loops.length}개의 루프 포인트 발견`);
              } else {
                setStatusMessage("루프 포인트를 찾을 수 없습니다");
              }
              setStatus("ready");
            }
          } else if (prog.stage === "error") {
            setStatus("error");
            setStatusMessage(prog.error || "오류가 발생했습니다");
          } else {
            // Continue polling
            setTimeout(pollProgress, 500);
          }
        } catch (error) {
          setStatus("error");
          setStatusMessage(error instanceof Error ? error.message : "오류가 발생했습니다");
        }
      };

      // Start polling
      setTimeout(pollProgress, 200);
    } catch (error) {
      setStatus("error");
      setStatusMessage(error instanceof Error ? error.message : "오류가 발생했습니다");
    }
  }, []);

  // Keep ref updated for drop handler
  useEffect(() => {
    analyzeFileRef.current = handleAnalyzeFile;
  }, [handleAnalyzeFile]);

  const handleLoopChange = useCallback((nextStartSample: number, nextEndSample: number) => {
    if (!analysisResult || !selectedLoop) return;
    const sr = analysisResult.sample_rate;

    const start = Math.max(0, Math.min(nextStartSample, nextEndSample));
    const end = Math.max(start, Math.max(nextStartSample, nextEndSample));

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
        loops: prev.loops.map((loop) => (loop.index === updated.index ? updated : loop)),
      };
    });
  }, [analysisResult, selectedLoop]);

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
      setStatusMessage(error instanceof Error ? error.message : "오류가 발생했습니다");
    }
  }, [handleAnalyzeFile]);

  const handleLoopSelect = useCallback(async (loop: LoopPoint) => {
    setSelectedLoop(loop);

    // Auto-play from 3 seconds before loop end to hear the transition
    if (wavesurferRef.current && analysisResult) {
      const sampleRate = analysisResult.sample_rate;
      const loopStartTime = Math.min(loop.start_sample, loop.end_sample) / sampleRate;
      const loopEndTime = Math.max(loop.start_sample, loop.end_sample) / sampleRate;

      const media = wavesurferRef.current.getMediaElement() as unknown as LoopingMedia;
      media.setLoopPoints?.(loopStartTime, loopEndTime);
      media.setLooping?.(isLooping);

      const previewStart = Math.max(0, loopEndTime - 3);
      wavesurferRef.current.setTime(previewStart);
      wavesurferRef.current.play();
      setIsPlaying(true);
    }
  }, [analysisResult, isLooping]);

  const handlePlayPause = useCallback(async () => {
    if (!wavesurferRef.current || !selectedLoop || !analysisResult) return;

    const sampleRate = analysisResult.sample_rate;

    if (isPlaying) {
      wavesurferRef.current.pause();
      setIsPlaying(false);
    } else {
      const startTime = Math.min(selectedLoop.start_sample, selectedLoop.end_sample) / sampleRate;
      const endTime = Math.max(selectedLoop.start_sample, selectedLoop.end_sample) / sampleRate;

      const media = wavesurferRef.current.getMediaElement() as unknown as LoopingMedia;
      media.setLoopPoints?.(startTime, endTime);
      media.setLooping?.(isLooping);

      const previewTime = Math.max(0, endTime - 3);

      wavesurferRef.current.setTime(previewTime);
      wavesurferRef.current.play();
      setIsPlaying(true);
    }
  }, [isPlaying, selectedLoop, analysisResult, isLooping]);

  const handleWaveformReady = useCallback(() => {
    if (wavesurferRef.current) {
      setDuration(wavesurferRef.current.getDuration());
    }
    if (selectedLoop && analysisResult && wavesurferRef.current) {
      const sampleRate = analysisResult.sample_rate;
      const startTime = Math.min(selectedLoop.start_sample, selectedLoop.end_sample) / sampleRate;
      const endTime = Math.max(selectedLoop.start_sample, selectedLoop.end_sample) / sampleRate;

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
    currentFilePathRef.current = null;
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore shortcuts when typing in input fields
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Only enable shortcuts when audio is ready
      if (status !== "ready") return;

      switch (e.key.toLowerCase()) {
        case " ": // Space - Play/Pause
          e.preventDefault();
          handlePlayPause();
          break;
        case "l": // L - Toggle loop
          setIsLooping((prev) => !prev);
          break;
        case "e": // E - Open export menu
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
          // 1-9 - Select loop by index
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
        </header>

        <main className="space-y-6">
          {status === "idle" && (
            <Card
              className="border-2 border-dashed p-8 text-center cursor-pointer transition-colors border-muted-foreground/25 hover:border-primary/50"
              onClick={handleSelectFile}
            >
              <FolderOpen className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-lg font-medium mb-2">파일을 끌어다 놓거나 클릭하세요</p>
              <p className="text-sm text-muted-foreground">
                지원 형식: MP3, WAV, FLAC, OGG, M4A
              </p>
            </Card>
          )}

          {(status === "selecting" || status === "analyzing") && (
            <div className="flex flex-col items-center justify-center py-16 gap-6">
              <Loader2 className="h-12 w-12 animate-spin text-primary" />
              <div className="w-full max-w-md space-y-2">
                <p className="text-lg text-muted-foreground text-center">{statusMessage}</p>
                {status === "analyzing" && progress.total > 0 && (
                  <div className="space-y-1">
                    <Progress value={progress.current} max={progress.total} className="h-2" />
                    <p className="text-xs text-muted-foreground text-center">
                      {progress.current} / {progress.total}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {(status === "ready" || status === "error") && (
            <>
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className="text-lg font-semibold">{filename}</h2>
                  <p className="text-sm text-muted-foreground">{statusMessage}</p>
                  {analysisResult?.enhancements && (
                    <div className="mt-2 flex flex-wrap gap-2">
                      <Badge
                        variant={analysisResult.enhancements.beat_alignment.effective ? "secondary" : "outline"}
                        className={!analysisResult.enhancements.beat_alignment.enabled ? "opacity-60" : ""}
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
                        variant={analysisResult.enhancements.structure.effective ? "secondary" : "outline"}
                        className={!analysisResult.enhancements.structure.enabled ? "opacity-60" : ""}
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
                  sampleRate={analysisResult?.sample_rate}
                  isLooping={isLooping}
                  isPlaying={isPlaying}
                  onReady={handleWaveformReady}
                  onTimeUpdate={handleTimeUpdate}
                  onLoopChange={handleLoopChange}
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
                    <Select value={sortMode} onValueChange={(v) => setSortMode(v as SortMode)}>
                      <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="정렬 선택" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="score_desc">정확도(점수) ↓</SelectItem>
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
                sampleRate={analysisResult?.sample_rate}
                durationSeconds={analysisResult?.duration}
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
