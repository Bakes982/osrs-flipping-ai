import { useState, useEffect, useCallback, useRef } from 'react';

export function useApi(fetchFn, deps = [], interval = null) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const mountedRef = useRef(true);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const result = await fetchFn();
      if (mountedRef.current) {
        setData(result);
        setError(null);
      }
    } catch (e) {
      if (mountedRef.current) setError(e.message);
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, deps);

  useEffect(() => {
    mountedRef.current = true;
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

  return { data, loading, error, reload: load };
}
