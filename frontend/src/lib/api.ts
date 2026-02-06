// Tauri API bridge — HTTP fetch + SSE (replaces JSON-RPC invoke)
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";

export type ExportFormat = "ogg" | "wav";
export type InfoFormat = "json" | "txt";

export interface ProgressResponse {
  current: number;
  total: number;
  stage: string;
  error?: string;
}

export interface EnhancementStatus {
  enabled: boolean;
  effective: boolean;
}

export interface EnhancementsInfo {
  beat_alignment: EnhancementStatus;
  structure: EnhancementStatus;
  seam_refinement?: string;
}

export interface LoopPoint {
  index: number;
  start_sample: number;
  end_sample: number;
  start_time: string;
  end_time: string;
  duration: string;
  score: number;
  similarity_score?: number;
  // allin1 structure analysis fields (auto-populated if allin1 installed)
  start_segment?: string; // 'intro', 'verse', 'chorus', 'bridge', 'outro'
  end_segment?: string;
  is_downbeat_aligned?: boolean;
  structure_boost?: number;
}

export interface AnalyzeResponse {
  duration: number;
  sample_rate: number;
  enhancements?: EnhancementsInfo;
  loops: LoopPoint[];
}

interface AnalyzeResult {
  success: boolean;
  error?: string;
  duration?: number;
  sample_rate?: number;
  enhancements?: EnhancementsInfo;
  loops?: LoopPoint[];
}

const AUDIO_EXTENSIONS = ["mp3", "wav", "flac", "ogg", "m4a"];

// ── Server port management ──────────────────────────────────────────

let serverPort: number | null = null;

async function getBaseUrl(): Promise<string> {
  if (!serverPort) {
    serverPort = await invoke<number>("get_server_port");
  }
  return `http://127.0.0.1:${serverPort}`;
}

// ── File Selection (Frontend dialog via Tauri plugin) ──────────────

export async function selectFile(): Promise<{
  filename: string;
  path: string;
} | null> {
  const path = await open({
    filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
    multiple: false,
    directory: false,
  });

  if (!path) return null;

  const filename = path.split("/").pop() ?? path.split("\\").pop() ?? path;
  return { filename, path };
}

// ── Analysis (fetch → Python HTTP server) ───────────────────────────

export async function analyzeFile(filePath: string): Promise<AnalyzeResponse> {
  const baseUrl = await getBaseUrl();
  const response = await fetch(`${baseUrl}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_path: filePath }),
  });

  const result = (await response.json()) as AnalyzeResult;

  if (!result.success) {
    throw new Error(result.error || "Analysis failed");
  }

  return {
    duration: result.duration!,
    sample_rate: result.sample_rate!,
    enhancements: result.enhancements,
    loops: result.loops!,
  };
}

// ── Progress & Events (SSE) ─────────────────────────────────────────

export function onProgress(
  callback: (progress: ProgressResponse) => void
): Promise<() => void> {
  return (async () => {
    const baseUrl = await getBaseUrl();
    const eventSource = new EventSource(`${baseUrl}/progress`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as ProgressResponse;
        callback(data);
      } catch {
        // ignore malformed data
      }
    };

    return () => {
      eventSource.close();
    };
  })();
}

// ── Audio (direct HTTP serving) ─────────────────────────────────────

export async function getAudioUrl(): Promise<string | null> {
  const baseUrl = await getBaseUrl();
  // Return the URL directly — the browser/wavesurfer can fetch it.
  return `${baseUrl}/audio`;
}

// ── Waveform ───────────────────────────────────────────────────────

export async function getWaveformData(points = 1000): Promise<number[]> {
  const baseUrl = await getBaseUrl();
  const response = await fetch(`${baseUrl}/waveform?points=${points}`);
  const data = await response.json();
  if (Array.isArray(data)) return data;
  return [];
}

// ── Export (Frontend dialog → fetch with output path) ───────────────

export async function exportLoop(
  loopStart: number,
  loopEnd: number
): Promise<boolean> {
  const outputPath = await save({
    defaultPath: "loop_export.wav",
    filters: [{ name: "WAV Files", extensions: ["wav"] }],
  });
  if (!outputPath) return false;

  const baseUrl = await getBaseUrl();
  const response = await fetch(`${baseUrl}/export/loop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      loop_start: loopStart,
      loop_end: loopEnd,
      output_path: outputPath,
    }),
  });
  const result = await response.json();
  return typeof result === "boolean" ? result : !!result;
}

export async function exportWithLoopTags(
  loopStart: number,
  loopEnd: number,
  format: ExportFormat
): Promise<boolean> {
  const ext = format === "ogg" ? "ogg" : "wav";
  const outputPath = await save({
    defaultPath: `loop_tagged.${ext}`,
    filters: [{ name: `${ext.toUpperCase()} Files`, extensions: [ext] }],
  });
  if (!outputPath) return false;

  const baseUrl = await getBaseUrl();
  const response = await fetch(`${baseUrl}/export/loop-tags`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      loop_start: loopStart,
      loop_end: loopEnd,
      format,
      output_path: outputPath,
    }),
  });
  const result = await response.json();
  return typeof result === "boolean" ? result : !!result;
}

export async function exportSplitSections(
  loopStart: number,
  loopEnd: number
): Promise<boolean> {
  const outputDir = await open({
    directory: true,
    multiple: false,
  });
  if (!outputDir) return false;

  const baseUrl = await getBaseUrl();
  const response = await fetch(`${baseUrl}/export/split`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      loop_start: loopStart,
      loop_end: loopEnd,
      output_dir: outputDir,
    }),
  });
  const result = await response.json();
  return typeof result === "boolean" ? result : !!result;
}

export async function exportExtended(
  loopStart: number,
  loopEnd: number,
  loopCount: number
): Promise<boolean> {
  const outputPath = await save({
    defaultPath: "extended_loop.wav",
    filters: [{ name: "WAV Files", extensions: ["wav"] }],
  });
  if (!outputPath) return false;

  const baseUrl = await getBaseUrl();
  const response = await fetch(`${baseUrl}/export/extended`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      loop_start: loopStart,
      loop_end: loopEnd,
      loop_count: loopCount,
      output_path: outputPath,
    }),
  });
  const result = await response.json();
  return typeof result === "boolean" ? result : !!result;
}

export async function exportLoopInfo(
  loopStart: number,
  loopEnd: number,
  format: InfoFormat
): Promise<boolean> {
  const ext = format === "json" ? "json" : "txt";
  const outputPath = await save({
    defaultPath: `loop_info.${ext}`,
    filters: [{ name: `${ext.toUpperCase()} Files`, extensions: [ext] }],
  });
  if (!outputPath) return false;

  const baseUrl = await getBaseUrl();
  const response = await fetch(`${baseUrl}/export/info`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      loop_start: loopStart,
      loop_end: loopEnd,
      format,
      output_path: outputPath,
    }),
  });
  const result = await response.json();
  return typeof result === "boolean" ? result : !!result;
}
