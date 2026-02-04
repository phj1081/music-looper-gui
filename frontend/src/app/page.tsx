"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import { Music, Loader2, FolderOpen } from "lucide-react";
import { Waveform } from "@/components/waveform";
import { LoopTable } from "@/components/loop-table";
import { PlayerControls } from "@/components/player-controls";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  selectFile,
  analyzeFile,
  getAudioBase64,
  type LoopPoint,
  type AnalyzeResponse,
} from "@/lib/api";
import { ExportMenu } from "@/components/export-menu";

type Status = "idle" | "selecting" | "analyzing" | "ready" | "error";

// Global handler for file drop from PyWebView
declare global {
  interface Window {
    onFileDropped?: (filePath: string, filename: string) => void;
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
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const analyzeFileRef = useRef<((path: string, name: string) => void) | null>(null);

  // Wait for PyWebView to be ready and setup drop handler
  useEffect(() => {
    const checkPyWebView = () => {
      if (window.pywebview) {
        setIsReady(true);
      } else {
        setTimeout(checkPyWebView, 100);
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
    setStatusMessage("분석 중...");
    setFilename(name);

    try {
      const analysis = await analyzeFile(filePath);
      setAnalysisResult(analysis);

      const base64 = await getAudioBase64();
      if (base64) {
        setAudioDataUrl(`data:audio/wav;base64,${base64}`);
      }

      if (analysis.loops.length > 0) {
        setSelectedLoop(analysis.loops[0]);
        setStatusMessage(`${analysis.loops.length}개의 루프 포인트 발견`);
      } else {
        setStatusMessage("루프 포인트를 찾을 수 없습니다");
      }
      setStatus("ready");
    } catch (error) {
      setStatus("error");
      setStatusMessage(error instanceof Error ? error.message : "오류가 발생했습니다");
    }
  }, []);

  // Keep ref updated for drop handler
  useEffect(() => {
    analyzeFileRef.current = handleAnalyzeFile;
  }, [handleAnalyzeFile]);

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
    setIsPlaying(false);
    setSelectedLoop(loop);

    // Auto-play from 3 seconds before loop end to hear the transition
    if (wavesurferRef.current && analysisResult) {
      const loopStartTime = loop.start_sample / analysisResult.sample_rate;
      const loopEndTime = loop.end_sample / analysisResult.sample_rate;
      const previewStart = Math.max(loopStartTime, loopEndTime - 3);
      wavesurferRef.current.setTime(previewStart);
      await new Promise(resolve => setTimeout(resolve, 50));
      wavesurferRef.current.play();
      setIsPlaying(true);
    }
  }, [analysisResult]);

  const handlePlayPause = useCallback(async () => {
    if (!wavesurferRef.current || !selectedLoop || !analysisResult) return;

    if (isPlaying) {
      wavesurferRef.current.pause();
      setIsPlaying(false);
    } else {
      const startTime = selectedLoop.start_sample / analysisResult.sample_rate;
      const endTime = selectedLoop.end_sample / analysisResult.sample_rate;
      const previewSeconds = 3;
      const previewTime = Math.max(startTime, endTime - previewSeconds);

      wavesurferRef.current.setTime(previewTime);
      // 50ms 대기 (시킹 완료 및 버퍼 준비)
      await new Promise(resolve => setTimeout(resolve, 50));
      wavesurferRef.current.play();
      setIsPlaying(true);
    }
  }, [isPlaying, selectedLoop, analysisResult]);

  const handleTimeUpdate = useCallback(
    (currentTime: number) => {
      if (!isLooping || !isPlaying || !selectedLoop || !analysisResult || !wavesurferRef.current)
        return;

      const endTime = selectedLoop.end_sample / analysisResult.sample_rate;
      const startTime = selectedLoop.start_sample / analysisResult.sample_rate;

      if (currentTime >= endTime) {
        wavesurferRef.current.setTime(startTime);
      }
    },
    [isLooping, isPlaying, selectedLoop, analysisResult]
  );

  const handleWaveformReady = useCallback(() => {
    if (selectedLoop && analysisResult && wavesurferRef.current) {
      const startTime = selectedLoop.start_sample / analysisResult.sample_rate;
      wavesurferRef.current.setTime(startTime);
    }
  }, [selectedLoop, analysisResult]);

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
  }, []);

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
            <div className="flex flex-col items-center justify-center py-16 gap-4">
              <Loader2 className="h-12 w-12 animate-spin text-primary" />
              <p className="text-lg text-muted-foreground">{statusMessage}</p>
            </div>
          )}

          {(status === "ready" || status === "error") && (
            <>
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold">{filename}</h2>
                  <p className="text-sm text-muted-foreground">{statusMessage}</p>
                </div>
                <Button variant="outline" onClick={handleReset}>
                  다른 파일 선택
                </Button>
              </div>

              <Waveform
                audioUrl={audioDataUrl}
                loopStart={selectedLoop?.start_sample}
                loopEnd={selectedLoop?.end_sample}
                sampleRate={analysisResult?.sample_rate}
                isPlaying={isPlaying}
                onReady={handleWaveformReady}
                onTimeUpdate={handleTimeUpdate}
                wavesurferRef={wavesurferRef}
              />

              <LoopTable
                loops={analysisResult?.loops || []}
                selectedIndex={selectedLoop?.index ?? null}
                onSelect={handleLoopSelect}
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
