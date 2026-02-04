// PyWebView API types
declare global {
  interface Window {
    pywebview?: {
      api: {
        select_file: () => Promise<{ filename: string; path: string } | null>;
        analyze: (filePath: string, method?: string) => Promise<AnalyzeResult>;
        get_audio_base64: () => Promise<string | null>;
        get_waveform: (points?: number) => Promise<number[] | null>;
        export_loop: (loopStart: number, loopEnd: number) => Promise<boolean>;
        get_available_methods: () => Promise<AnalysisMethod[]>;
      };
    };
  }
}

export type AnalysisMethodId = "pymusiclooper" | "ssm";

export interface AnalysisMethod {
  id: AnalysisMethodId;
  name: string;
  description: string;
}

export interface LoopPoint {
  index: number;
  start_sample: number;
  end_sample: number;
  start_time: string;
  end_time: string;
  duration: string;
  score: number;
  method?: string;
  ssm_score?: number;
  correlation_score?: number;
}

interface AnalyzeResult {
  success: boolean;
  error?: string;
  duration?: number;
  sample_rate?: number;
  loops?: LoopPoint[];
  method?: string;
}

export interface AnalyzeResponse {
  duration: number;
  sample_rate: number;
  loops: LoopPoint[];
  method: string;
}

function getPyWebView() {
  return window.pywebview?.api;
}

export async function selectFile(): Promise<{ filename: string; path: string } | null> {
  const api = getPyWebView();
  if (!api) throw new Error("PyWebView not available");
  return api.select_file();
}

export async function analyzeFile(
  filePath: string,
  method: AnalysisMethodId = "pymusiclooper"
): Promise<AnalyzeResponse> {
  const api = getPyWebView();
  if (!api) throw new Error("PyWebView not available");

  const result = await api.analyze(filePath, method);

  if (!result.success) {
    throw new Error(result.error || "Analysis failed");
  }

  return {
    duration: result.duration!,
    sample_rate: result.sample_rate!,
    loops: result.loops!,
    method: result.method || method,
  };
}

export async function getAvailableMethods(): Promise<AnalysisMethod[]> {
  const api = getPyWebView();
  if (!api) throw new Error("PyWebView not available");
  return api.get_available_methods();
}

export async function getAudioBase64(): Promise<string | null> {
  const api = getPyWebView();
  if (!api) throw new Error("PyWebView not available");
  return api.get_audio_base64();
}

export async function getWaveformData(points = 1000): Promise<number[]> {
  const api = getPyWebView();
  if (!api) throw new Error("PyWebView not available");

  const data = await api.get_waveform(points);
  return data || [];
}

export async function exportLoop(loopStart: number, loopEnd: number): Promise<boolean> {
  const api = getPyWebView();
  if (!api) throw new Error("PyWebView not available");
  return api.export_loop(loopStart, loopEnd);
}
