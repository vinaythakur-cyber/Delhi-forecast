"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError } from "./api";

export interface ApiState<T> {
  data: T | null;
  error: string | null;
  status: number | null; // HTTP status of the last failure (503 means "still initialising")
  loading: boolean;
}

/** Fetch on mount and whenever `deps` change; optionally poll. Keeps showing the old data while refreshing. */
export function useApi<T>(fetcher: () => Promise<T>, deps: unknown[], refreshMs?: number): ApiState<T> {
  const [state, setState] = useState<ApiState<T>>({ data: null, error: null, status: null, loading: true });
  const fetchRef = useRef(fetcher);
  fetchRef.current = fetcher;

  useEffect(() => {
    let alive = true;
    const run = async () => {
      try {
        const data = await fetchRef.current();
        if (alive) setState({ data, error: null, status: null, loading: false });
      } catch (e) {
        if (alive) setState((s) => ({ data: s.data, error: e instanceof Error ? e.message : String(e), status: e instanceof ApiError ? e.status : null, loading: false }));
      }
    };
    setState((s) => ({ ...s, loading: true }));
    run();
    const id = refreshMs ? setInterval(run, refreshMs) : undefined;
    return () => {
      alive = false;
      if (id) clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
