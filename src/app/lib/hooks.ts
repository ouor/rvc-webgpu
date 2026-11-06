import { useEffect, useState } from 'react';

/** performance.now() refreshed every second while `active`; depends on the flag only, so frequent re-renders don't reset the timer. */
export function useNow(active: boolean) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (!active) return;
    setNow(performance.now()); // eslint-disable-line react-hooks/set-state-in-effect -- start from a fresh clock
    const id = setInterval(() => setNow(performance.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}
