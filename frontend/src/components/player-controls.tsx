"use client";

import { Play, Pause, Repeat } from "lucide-react";
import { Button } from "@/components/ui/button";

interface PlayerControlsProps {
  isPlaying: boolean;
  isLooping: boolean;
  disabled: boolean;
  onPlayPause: () => void;
  onToggleLoop: () => void;
}

export function PlayerControls({
  isPlaying,
  isLooping,
  disabled,
  onPlayPause,
  onToggleLoop,
}: PlayerControlsProps) {
  return (
    <div className="flex items-center gap-2">
      <Button
        variant={isPlaying ? "secondary" : "default"}
        size="lg"
        onClick={onPlayPause}
        disabled={disabled}
      >
        {isPlaying ? (
          <>
            <Pause className="mr-2 h-5 w-5" />
            정지
          </>
        ) : (
          <>
            <Play className="mr-2 h-5 w-5" />
            미리듣기
          </>
        )}
      </Button>

      <Button
        variant={isLooping ? "default" : "outline"}
        size="lg"
        onClick={onToggleLoop}
        disabled={disabled}
        title="루프 반복"
      >
        <Repeat className={`h-5 w-5 ${isLooping ? "" : "opacity-50"}`} />
      </Button>
    </div>
  );
}
