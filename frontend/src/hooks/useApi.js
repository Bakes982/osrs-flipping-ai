import { useState, useEffect, useCallback, useRef, useContext } from 'react';
import { ErrorContext } from '../components/ErrorPanel';

export function useApi(fetchFn, deps = [], interval = null, label = '') {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [errorObj, setErrorObj] = useState(null);
  const mountedRef = useRef(true);
  const initialLoadDone = useRef(false);
  const errorCtx = useContext(ErrorContext);

  const load = useCallback(async () => {
    try {
      // Only show loading spinner on the very first fetch.
      // Subsequent refreshes update data silently in the background.
      if (!initialLoadDone.current) {
        setLoading(true);
      }
      const result = await fetchFn();
      if (mountedRef.current) {
        setData(result);
        setErrorObj(null);
        initialLoadDone.current = true;
      }
    } catch (e) {
      if (mountedRef.current) {
        const err = e instanceof Error ? e : new Error(String(e));
        setErrorObj(err);
        if (errorCtx) errorCtx.addError(err, label);
      }
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, deps);

  useEffect(() => {
    mountedRef.current = true;
    initialLoadDone.current = false;
    load();

    let timer;
    if (interval) {
      timer = setInterval(load, interval);
    }

    return () => {
      mountedRef.current = false;
      if (timer) clearInterval(timer);
    };
  }, [load, interval]);

  // `error` is a string for backward compatibility; `errorObj` has the full Error
  return { data, loading, error: errorObj?.message || null, errorObj, reload: load };
}
