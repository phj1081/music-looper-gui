"use client";

import { useEffect } from "react";

function isAbortError(reason: unknown): boolean {
  if (!reason) return false;

  if (reason instanceof DOMException && reason.name === "AbortError") return true;

  if (typeof reason === "object") {
    const maybeError = reason as { name?: unknown; message?: unknown };
    if (maybeError.name === "AbortError") return true;
    if (typeof maybeError.message === "string" && maybeError.message.includes("Fetch is aborted")) return true;
    if (typeof maybeError.message === "string" && maybeError.message.toLowerCase().includes("aborted")) return true;
  }

  if (typeof reason === "string" && reason.toLowerCase().includes("aborted")) return true;

  return false;
}

export function DevAbortErrorSuppressor() {
  useEffect(() => {
    if (process.env.NODE_ENV !== "development") return;

    const onUnhandledRejection = (event: PromiseRejectionEvent) => {
      if (!isAbortError(event.reason)) return;
      event.preventDefault();
      event.stopImmediatePropagation();
    };

    const onError = (event: ErrorEvent) => {
      // Some libraries surface aborts as normal ErrorEvents, not unhandled rejections.
      if (!isAbortError(event.error) && !isAbortError(event.message)) return;
      event.preventDefault();
      event.stopImmediatePropagation();
    };

    window.addEventListener("unhandledrejection", onUnhandledRejection, { capture: true });
    window.addEventListener("error", onError, { capture: true });
    return () => {
      window.removeEventListener("unhandledrejection", onUnhandledRejection, { capture: true });
      window.removeEventListener("error", onError, { capture: true });
    };
  }, []);

  return null;
}
