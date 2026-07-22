import { useSyncExternalStore } from "react";

let size = { width: 0, height: 0 };

function subscribe(callback: () => void) {
  window.addEventListener("resize", callback);
  return () => window.removeEventListener("resize", callback);
}

function getSnapshot() {
  if (typeof window === "undefined") {
    return { width: 1200, height: 800 };
  }
  if (size.width !== window.innerWidth || size.height !== window.innerHeight) {
    size = { width: window.innerWidth, height: window.innerHeight };
  }
  return size;
}

function getServerSnapshot() {
  return { width: 1200, height: 800 };
}

export function useWindowSize() {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
