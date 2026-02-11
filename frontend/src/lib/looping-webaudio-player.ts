const DEFAULT_FADE_SECONDS = 0.004;
type AudioHealthEventName = "contextlost" | "contextrecovered" | "contextrecoveryfailed";

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

/**
 * A WebAudio-backed "media element" that supports gapless looping between loop points.
 *
 * It intentionally implements only the subset of the HTMLMediaElement API that
 * wavesurfer.js uses.
 */
export class LoopingWebAudioPlayer extends EventTarget {
  private audioContext: AudioContext;
  private gainNode: GainNode;

  private buffer: AudioBuffer | null = null;
  private bufferNode: AudioBufferSourceNode | null = null;

  private _src = "";
  private _durationOverride: number | undefined;
  private _volume = 1;
  private _muted = false;
  private _playbackRate = 1;

  private loadId = 0;
  private loadAbortController: AbortController | null = null;

  private playRequested = false;
  private pendingPlayPromise: Promise<void> | null = null;
  private resolvePendingPlay: (() => void) | null = null;
  private resumePromise: Promise<void> | null = null;
  private contextRecoveryPromise: Promise<boolean> | null = null;
  private lastContextState: string;
  private readonly contextStateChangeHandler: () => void;

  // Playback state
  paused = true;
  ended = false;
  seeking = false;
  autoplay = false;
  controls = false;
  crossOrigin: string | null = null;

  private positionSeconds = 0;
  private startedAtContextSeconds = 0;
  private positionAtStartSeconds = 0;

  // Loop config
  private loopEnabled = false;
  private loopStartSeconds = 0;
  private loopEndSeconds = 0;

  constructor(audioContext: AudioContext = new AudioContext()) {
    super();
    this.audioContext = audioContext;
    this.gainNode = this.audioContext.createGain();
    this.gainNode.gain.value = 1;
    this.gainNode.connect(this.audioContext.destination);
    this.lastContextState = this.getContextState();
    this.contextStateChangeHandler = () => {
      this.handleContextStateChange();
    };
    this.audioContext.addEventListener("statechange", this.contextStateChangeHandler);
  }

  private getContextState(): string {
    return this.audioContext.state as string;
  }

  private isRecoverableContextState(state: string): boolean {
    return state === "suspended" || state === "interrupted";
  }

  private emitContextEvent(type: AudioHealthEventName, reason: string): void {
    this.dispatchEvent(
      new CustomEvent(type, {
        detail: {
          reason,
          state: this.getContextState(),
        },
      }),
    );
  }

  private handleContextStateChange(): void {
    const currentState = this.getContextState();
    const previousState = this.lastContextState;
    this.lastContextState = currentState;
    if (currentState === previousState) return;

    if (this.isRecoverableContextState(currentState) && !this.paused) {
      this.emitContextEvent("contextlost", "statechange");
      void this.recoverContextIfNeeded("statechange");
      return;
    }

    if (
      currentState === "running" &&
      this.isRecoverableContextState(previousState) &&
      !this.paused &&
      this.buffer &&
      !this.bufferNode
    ) {
      const offset = clamp(this.currentTime, 0, this.duration || Infinity);
      this.positionSeconds = offset;
      this.positionAtStartSeconds = offset;
      this.startedAtContextSeconds = this.audioContext.currentTime;
      this.startSource(offset);
    }
  }

  private async recoverContextIfNeeded(reason: string): Promise<boolean> {
    const state = this.getContextState();
    if (state === "running") return true;
    if (state === "closed") {
      this.emitContextEvent("contextrecoveryfailed", reason);
      return false;
    }
    if (!this.isRecoverableContextState(state)) return false;

    if (!this.contextRecoveryPromise) {
      this.contextRecoveryPromise = (async () => {
        try {
          await this.audioContext.resume();
        } catch (error) {
          console.warn("LoopingWebAudioPlayer: audio context resume failed", error);
        }

        const recovered = this.getContextState() === "running";
        this.emitContextEvent(recovered ? "contextrecovered" : "contextrecoveryfailed", reason);
        return recovered;
      })();
    }

    try {
      return await this.contextRecoveryPromise;
    } finally {
      this.contextRecoveryPromise = null;
    }
  }

  private requestResume(): void {
    if (!this.isRecoverableContextState(this.getContextState())) return;
    if (this.resumePromise) return;

    this.resumePromise = this.audioContext.resume().catch(() => {
      // If resume fails (e.g. missing user gesture), allow future attempts
      this.resumePromise = null;
    });
  }

  private async ensureResumed(): Promise<void> {
    if (this.resumePromise) {
      try {
        await this.resumePromise;
      } catch {
        // Ignore
      }
      return;
    }

    if (this.isRecoverableContextState(this.getContextState())) {
      try {
        await this.audioContext.resume();
      } catch {
        // Ignore
      }
    }
  }

  // ---- Source loading --------------------------------------------------------

  get src(): string {
    return this._src;
  }

  set src(value: string) {
    this._src = value;
    this._durationOverride = undefined;
    this.ended = false;

    const currentLoadId = ++this.loadId;
    this.loadAbortController?.abort();
    this.loadAbortController = null;

    if (!value) {
      this.stopInternal();
      this.buffer = null;
      this.positionSeconds = 0;
      this.playRequested = false;
      this.resolvePendingPlay?.();
      this.resolvePendingPlay = null;
      this.pendingPlayPromise = null;
      this.dispatchEvent(new Event("emptied"));
      return;
    }

    const controller = new AbortController();
    this.loadAbortController = controller;

    fetch(value, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to fetch ${value}: ${response.status} (${response.statusText})`);
        }
        return response.arrayBuffer();
      })
      .then((arrayBuffer) => {
        if (this.loadId !== currentLoadId) return null;
        return this.audioContext.decodeAudioData(arrayBuffer);
      })
      .then((audioBuffer) => {
        if (!audioBuffer) return;
        if (this.loadId !== currentLoadId) return;
        if (this.loadAbortController === controller) {
          this.loadAbortController = null;
        }

        this.buffer = audioBuffer;
        this.dispatchEvent(new Event("loadedmetadata"));
        this.dispatchEvent(new Event("durationchange"));
        this.dispatchEvent(new Event("canplay"));

        if (this.playRequested || this.autoplay) {
          void this.play();
        }
      })
      .catch((err) => {
        if (this.loadAbortController === controller) {
          this.loadAbortController = null;
        }
        if (err instanceof DOMException && err.name === "AbortError") return;
        if (typeof err === "object" && err && "name" in err && (err as { name?: unknown }).name === "AbortError") return;
        console.error("LoopingWebAudioPlayer load error:", err);
        this.dispatchEvent(new Event("error"));
      });
  }

  get currentSrc(): string {
    return this._src;
  }

  async load(): Promise<void> {
    return;
  }

  // ---- Timing ---------------------------------------------------------------

  get duration(): number {
    return this._durationOverride ?? this.buffer?.duration ?? 0;
  }

  set duration(value: number) {
    this._durationOverride = value;
    this.dispatchEvent(new Event("durationchange"));
  }

  private getRawPositionSeconds(nowContextSeconds = this.audioContext.currentTime): number {
    if (this.paused) return this.positionSeconds;
    const elapsed = nowContextSeconds - this.startedAtContextSeconds;
    return this.positionAtStartSeconds + elapsed * this._playbackRate;
  }

  private getLoopIsValid(): boolean {
    return (
      this.loopEnabled &&
      isFiniteNumber(this.loopStartSeconds) &&
      isFiniteNumber(this.loopEndSeconds) &&
      this.loopEndSeconds > this.loopStartSeconds
    );
  }

  private wrapPositionSeconds(rawSeconds: number): number {
    const duration = this.duration;
    if (!isFiniteNumber(duration) || duration <= 0) return 0;

    if (!this.getLoopIsValid()) return clamp(rawSeconds, 0, duration);

    const safeRaw = Math.max(0, rawSeconds);

    // No wrapping until we reach loop end the first time
    if (safeRaw < this.loopEndSeconds) return clamp(safeRaw, 0, duration);

    const loopLength = this.loopEndSeconds - this.loopStartSeconds;
    if (loopLength <= 0) return clamp(this.loopStartSeconds, 0, duration);

    const afterEnd = safeRaw - this.loopEndSeconds;
    const wrapped = this.loopStartSeconds + (afterEnd % loopLength);
    return clamp(wrapped, 0, duration);
  }

  get currentTime(): number {
    return this.wrapPositionSeconds(this.getRawPositionSeconds());
  }

  set currentTime(value: number) {
    const wasPlaying = !this.paused;

    this.seeking = true;
    this.dispatchEvent(new Event("seeking"));

    const duration = this.duration;
    const safe = isFiniteNumber(duration) && duration > 0 ? clamp(value, 0, duration) : Math.max(0, value);

    // In loop mode, keep seeks inside the loop range to avoid undefined WebAudio behavior
    let next = safe;
    if (this.getLoopIsValid() && next >= this.loopEndSeconds) {
      next = clamp(this.loopStartSeconds, 0, duration);
    }

    this.positionSeconds = next;
    this.positionAtStartSeconds = next;
    this.startedAtContextSeconds = this.audioContext.currentTime;

    if (wasPlaying) {
      this.restartAt(next);
    }

    this.seeking = false;
    this.dispatchEvent(new Event("seeked"));
    this.dispatchEvent(new Event("timeupdate"));
  }

  // ---- Volume / mute / rate -------------------------------------------------

  get volume(): number {
    return this._volume;
  }

  set volume(value: number) {
    this._volume = clamp(value, 0, 1);
    if (!this._muted) {
      this.gainNode.gain.setValueAtTime(this._volume, this.audioContext.currentTime);
    }
    this.dispatchEvent(new Event("volumechange"));
  }

  get muted(): boolean {
    return this._muted;
  }

  set muted(value: boolean) {
    this._muted = value;
    this.gainNode.gain.setValueAtTime(this._muted ? 0 : this._volume, this.audioContext.currentTime);
    this.dispatchEvent(new Event("volumechange"));
  }

  get playbackRate(): number {
    return this._playbackRate;
  }

  set playbackRate(value: number) {
    const next = value > 0 ? value : 1;
    const now = this.audioContext.currentTime;

    if (!this.paused) {
      const pos = this.currentTime;
      this.positionSeconds = pos;
      this.positionAtStartSeconds = pos;
      this.startedAtContextSeconds = now;
    }

    this._playbackRate = next;
    if (this.bufferNode) {
      this.bufferNode.playbackRate.setValueAtTime(next, now);
    }
    this.dispatchEvent(new Event("ratechange"));
  }

  // ---- Loop control ---------------------------------------------------------

  setLoopPoints(startSeconds: number, endSeconds: number): void {
    const start = Math.min(startSeconds, endSeconds);
    const end = Math.max(startSeconds, endSeconds);

    const duration = this.duration;
    const maxTime = isFiniteNumber(duration) && duration > 0 ? duration : Infinity;
    this.loopStartSeconds = clamp(start, 0, maxTime);
    this.loopEndSeconds = clamp(end, 0, maxTime);

    if (this.bufferNode && this.getLoopIsValid()) {
      this.bufferNode.loopStart = this.loopStartSeconds;
      this.bufferNode.loopEnd = this.loopEndSeconds;
    }

    // Re-anchor time so toggling loop doesn't jump due to large raw time.
    if (!this.paused) {
      const now = this.audioContext.currentTime;
      const pos = this.currentTime;
      this.positionSeconds = pos;
      this.positionAtStartSeconds = pos;
      this.startedAtContextSeconds = now;
    }
  }

  setLooping(enabled: boolean): void {
    if (this.loopEnabled === enabled) return;

    const now = this.audioContext.currentTime;
    const pos = this.currentTime;

    this.loopEnabled = enabled;

    // Keep progress stable when changing loop mode
    this.positionSeconds = pos;
    this.positionAtStartSeconds = pos;
    this.startedAtContextSeconds = now;

    if (this.bufferNode) {
      if (this.getLoopIsValid()) {
        this.bufferNode.loop = true;
        this.bufferNode.loopStart = this.loopStartSeconds;
        this.bufferNode.loopEnd = this.loopEndSeconds;
      } else {
        this.bufferNode.loop = false;
      }
    }

    if (this.getLoopIsValid() && pos >= this.loopEndSeconds) {
      this.currentTime = this.loopStartSeconds;
    }
  }

  // ---- Playback -------------------------------------------------------------

  async play(): Promise<void> {
    if (!this.paused) return;

    // Ensure we request resume on the *user gesture* call path even if the buffer isn't ready yet.
    this.requestResume();

    if (!this.buffer) {
      this.playRequested = true;
      if (!this.pendingPlayPromise) {
        this.pendingPlayPromise = new Promise((resolve) => {
          this.resolvePendingPlay = resolve;
        });
      }
      return this.pendingPlayPromise;
    }

    await this.recoverContextIfNeeded("play");
    await this.ensureResumed();
    if (this.getContextState() !== "running") {
      this.emitContextEvent("contextrecoveryfailed", "play");
      throw new Error(`Audio context is ${this.getContextState()}`);
    }

    this.playRequested = false;

    this.ended = false;
    this.paused = false;

    const now = this.audioContext.currentTime;
    const offset = clamp(this.positionSeconds, 0, this.duration || Infinity);

    this.positionAtStartSeconds = offset;
    this.startedAtContextSeconds = now;

    let started = this.startSource(offset);
    if (!started) {
      const recovered = await this.recoverContextIfNeeded("play-start");
      if (recovered) {
        started = this.startSource(offset);
      }
    }
    if (!started) {
      this.paused = true;
      this.emitContextEvent("contextrecoveryfailed", "play-start");
      throw new Error("Failed to start audio playback");
    }
    this.dispatchEvent(new Event("play"));

    this.resolvePendingPlay?.();
    this.resolvePendingPlay = null;
    this.pendingPlayPromise = null;
  }

  pause(): void {
    if (this.paused) {
      // Cancel any queued playback (e.g. play() called before audio finished decoding)
      this.playRequested = false;
      this.resolvePendingPlay?.();
      this.resolvePendingPlay = null;
      this.pendingPlayPromise = null;
      return;
    }

    // Snapshot wrapped position at pause time
    this.positionSeconds = this.currentTime;
    this.paused = true;

    this.stopInternal();
    this.dispatchEvent(new Event("pause"));
  }

  private startSource(offsetSeconds: number): boolean {
    if (!this.buffer) return false;
    if (this.getContextState() !== "running") return false;

    // Make sure any previous node is gone
    this.stopInternal();

    let source: AudioBufferSourceNode;
    try {
      source = this.audioContext.createBufferSource();
      source.buffer = this.buffer;
      source.playbackRate.setValueAtTime(this._playbackRate, this.audioContext.currentTime);
      source.connect(this.gainNode);
    } catch (error) {
      console.error("LoopingWebAudioPlayer startSource creation failed:", error);
      this.dispatchEvent(new Event("error"));
      return false;
    }

    if (this.getLoopIsValid()) {
      source.loop = true;
      source.loopStart = this.loopStartSeconds;
      source.loopEnd = this.loopEndSeconds;
    }

    // Light fade-in to prevent clicks on start/seek/pause
    const now = this.audioContext.currentTime;
    if (!this._muted) {
      this.gainNode.gain.cancelScheduledValues(now);
      this.gainNode.gain.setValueAtTime(0, now);
      this.gainNode.gain.linearRampToValueAtTime(this._volume, now + DEFAULT_FADE_SECONDS);
    }

    source.onended = () => {
      // Ignore if it was replaced/stopped
      if (this.bufferNode !== source) return;

      // Natural end (loop disabled) – keep consistent with HTMLMediaElement
      if (!this.getLoopIsValid() && this.currentTime >= this.duration) {
        this.positionSeconds = this.duration;
        this.paused = true;
        this.ended = true;
        this.bufferNode = null;
        this.dispatchEvent(new Event("ended"));
        this.dispatchEvent(new Event("pause"));
      }
    };

    this.bufferNode = source;
    try {
      source.start(now, offsetSeconds);
      return true;
    } catch (error) {
      console.error("LoopingWebAudioPlayer source.start failed:", error);
      this.bufferNode = null;
      this.dispatchEvent(new Event("error"));
      return false;
    }
  }

  private restartAt(offsetSeconds: number): void {
    if (this.paused) return;
    this.positionSeconds = offsetSeconds;
    this.positionAtStartSeconds = offsetSeconds;
    this.startedAtContextSeconds = this.audioContext.currentTime;
    if (!this.startSource(offsetSeconds)) {
      void this.recoverContextIfNeeded("restart");
    }
  }

  private stopInternal(): void {
    const node = this.bufferNode;
    if (!node) return;

    // Prevent onended from firing natural-end handlers for intentional stops
    node.onended = null;

    const now = this.audioContext.currentTime;
    if (!this._muted) {
      try {
        const currentGain = this.gainNode.gain.value;
        this.gainNode.gain.cancelScheduledValues(now);
        this.gainNode.gain.setValueAtTime(currentGain, now);
        this.gainNode.gain.linearRampToValueAtTime(0, now + DEFAULT_FADE_SECONDS);
      } catch {
        // Safari can throw if context is closed; ignore
      }
    }

    try {
      node.stop(now + DEFAULT_FADE_SECONDS + 0.001);
    } catch {
      // Ignore if already stopped
    }

    try {
      node.disconnect();
    } catch {
      // Ignore
    }

    this.bufferNode = null;
  }

  // ---- Compatibility helpers ------------------------------------------------

  canPlayType(mimeType: string): string {
    return /^(audio|video)\//.test(mimeType) ? "maybe" : "";
  }

  removeAttribute(attrName: string): void {
    switch (attrName) {
      case "src":
        this.src = "";
        break;
      case "playbackRate":
        this.playbackRate = 1;
        break;
      case "currentTime":
        this.currentTime = 0;
        break;
      case "duration":
        this.duration = 0;
        break;
      case "volume":
        this.volume = 1;
        break;
      case "muted":
        this.muted = false;
        break;
    }
  }

  async setSinkId(deviceId: string): Promise<void> {
    // Not supported in WebAudio
    void deviceId;
    return;
  }

  destroy(): void {
    this.stopInternal();
    this.buffer = null;
    this._src = "";
    this.loadAbortController?.abort();
    this.loadAbortController = null;
    this.audioContext.removeEventListener("statechange", this.contextStateChangeHandler);
    void this.audioContext.close().catch(() => {
      // Ignore
    });
  }
}
