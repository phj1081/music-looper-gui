"use client";

import { useCallback, useMemo, useState } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import type { LoopPoint } from "@/lib/api";

function formatSecondsMMSSms(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00.000";
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toFixed(3).padStart(6, "0")}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export interface LoopEditorProps {
  loop: LoopPoint | null;
  sampleRate?: number | undefined;
  durationSeconds?: number | undefined;
  onChange: (loopStartSample: number, loopEndSample: number) => void;
}

export function LoopEditor({ loop, sampleRate, durationSeconds, onChange }: LoopEditorProps) {
  const [startSampleText, setStartSampleText] = useState(() => (loop ? String(loop.start_sample) : ""));
  const [endSampleText, setEndSampleText] = useState(() => (loop ? String(loop.end_sample) : ""));

  const sr = sampleRate ?? 0;
  const startSeconds = useMemo(() => (loop && sr > 0 ? loop.start_sample / sr : 0), [loop, sr]);
  const endSeconds = useMemo(() => (loop && sr > 0 ? loop.end_sample / sr : 0), [loop, sr]);

  const apply = useCallback(() => {
    if (!loop) return;
    if (!Number.isFinite(sr) || sr <= 0) return;

    const parsedStart = Number.parseInt(startSampleText, 10);
    const parsedEnd = Number.parseInt(endSampleText, 10);
    if (!Number.isFinite(parsedStart) || !Number.isFinite(parsedEnd)) return;

    let nextStart = Math.min(parsedStart, parsedEnd);
    let nextEnd = Math.max(parsedStart, parsedEnd);

    const maxSamples =
      Number.isFinite(durationSeconds) && (durationSeconds ?? 0) > 0
        ? Math.max(0, Math.floor((durationSeconds as number) * sr))
        : Number.POSITIVE_INFINITY;

    nextStart = Math.round(clamp(nextStart, 0, maxSamples));
    nextEnd = Math.round(clamp(nextEnd, 0, maxSamples));

    // Enforce a small minimum region length (20ms) so the region remains usable
    const minSamples = Math.max(1, Math.round(0.02 * sr));
    if (nextEnd - nextStart < minSamples) {
      nextEnd = Math.min(maxSamples, nextStart + minSamples);
    }

    onChange(nextStart, nextEnd);
  }, [durationSeconds, endSampleText, loop, onChange, sr, startSampleText]);

  if (!loop) return null;

  return (
    <Card className="p-4">
      <div className="flex flex-col gap-3">
        <div>
          <p className="text-sm font-medium">루프 구간 편집</p>
          <p className="text-xs text-muted-foreground">
            파형의 초록 구간을 드래그/리사이즈하거나, 샘플 값을 직접 입력할 수 있어요.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="space-y-1">
            <Label htmlFor="loop-start-sample">시작 (sample)</Label>
            <Input
              id="loop-start-sample"
              type="number"
              inputMode="numeric"
              value={startSampleText}
              onChange={(e) => setStartSampleText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") apply();
              }}
            />
            <p className="text-xs font-mono text-muted-foreground">{formatSecondsMMSSms(startSeconds)}</p>
          </div>

          <div className="space-y-1">
            <Label htmlFor="loop-end-sample">끝 (sample)</Label>
            <Input
              id="loop-end-sample"
              type="number"
              inputMode="numeric"
              value={endSampleText}
              onChange={(e) => setEndSampleText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") apply();
              }}
            />
            <p className="text-xs font-mono text-muted-foreground">{formatSecondsMMSSms(endSeconds)}</p>
          </div>
        </div>

        <div className="flex justify-end">
          <Button variant="secondary" onClick={apply}>
            적용
          </Button>
        </div>
      </div>
    </Card>
  );
}
