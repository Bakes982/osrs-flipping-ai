import { useState, useEffect, useCallback, useRef, useContext } from 'react';
import { ErrorContext } from '../components/ErrorPanel';

/**
 * Generic data-fetching hook with auto-refresh, error handling, and
 * in-flight request cancellation.
 *
 * @param {() => Promise<any>} fetchFn   – async function that returns data
 * @param {any[]}              deps      – re-fetch when these change (like useEffect)
 * @param {number|null}        interval  – auto-refresh interval in ms (null = no auto-refresh)
 * @param {string}             label     – label for ErrorPanel (optional)
 */
export function useApi(fetchFn, deps = [], interval = null, label = '') {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [errorObj, setErrorObj] = useState(null);

  const mountedRef       = useRef(true);
  const initialLoadDone  = useRef(false);
  // AbortController for the currently in-flight request
  const abortCtrlRef     = useRef(null);
  const errorCtx         = useContext(ErrorContext);

  const load = useCallback(async () => {
    // Cancel any previously in-flight request before starting a new one.
    if (abortCtrlRef.current) {
      abortCtrlRef.current.abort();
    }
    const ctrl = new AbortController();
    abortCtrlRef.current = ctrl;

    try {
      // Only show the full loading spinner on the very first fetch.
      // Subsequent background refreshes update data silently.
      if (!initialLoadDone.current) {
        setLoading(true);
      }
      const result = await fetchFn();
      if (mountedRef.current && !ctrl.signal.aborted) {
        setData(result);
        setErrorObj(null);
        initialLoadDone.current = true;
      }
    } catch (e) {
      // Ignore aborts – they are expected on dep changes and unmount.
      if (e?.name === 'AbortError') return;
      if (mountedRef.current) {
        const err = e instanceof Error ? e : new Error(String(e));
        setErrorObj(err);
        if (errorCtx) errorCtx.addError(err, label);
      }
    } finally {
      if (mountedRef.current && !ctrl.signal.aborted) {
        setLoading(false);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    mountedRef.current      = true;
    initialLoadDone.current = false;
    load();

    let timer;
    if (interval) {
      timer = setInterval(load, interval);
    }

    return () => {
      mountedRef.current = false;
      // Abort any in-flight request when the component unmounts or deps change.
      if (abortCtrlRef.current) {
        abortCtrlRef.current.abort();
        abortCtrlRef.current = null;
      }
      if (timer) clearInterval(timer);
    };
  }, [load, interval]);

  // `error` is a string for backward compatibility; `errorObj` has the full Error
  return { data, loading, error: errorObj?.message || null, errorObj, reload: load };
}
