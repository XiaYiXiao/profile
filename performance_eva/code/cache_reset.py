"""Reset an idle benchmark service before taking the load metrics baseline."""
from datetime import datetime, timezone
import json
import time
import urllib.request

from server_metrics import parse_metrics


def reset_prefix_cache(url, targets, report, *, api_key=None, timeout=60., idle_seconds=6.):
    """Fail closed on unavailable metrics, busy workers or an unsuccessful reset.

    All DP gauge series must be idle for at least one normal vLLM stats interval.
    The service must be reserved for this benchmark: this is not an admission lock.
    """
    headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
    report.update(status='waiting_for_idle', started_at=datetime.now(timezone.utc).isoformat())
    deadline = time.monotonic() + timeout
    idle_since = None
    try:
        while True:
            busy = False
            observations = {}
            for role, target in targets.items():
                request = urllib.request.Request(target, headers=headers)
                with urllib.request.urlopen(request, timeout=min(10., max(.1, deadline - time.monotonic()))) as response:
                    exposition = response.read().decode()
                    rows = parse_metrics(exposition)
                names = (('eva_pd_inflight',) if role == 'proxy' else
                         ('vllm:num_requests_running', 'vllm:num_requests_waiting'))
                selected = [row for row in rows if row['name'] in names]
                if any(not any(row['name'] == name for row in selected) for name in names):
                    # Image vLLM's multiprocess "mostrecent" gauges have no
                    # samples until the first scheduler update. Accept this
                    # pristine state only if the gauge declarations exist and
                    # every exposed workload counter is still zero.
                    counters = [r for r in rows if r['kind'] == 'counter']
                    pristine = (role != 'proxy' and not selected and
                                all(f'# TYPE {name} gauge' in exposition for name in names) and
                                any(r['name'] == 'vllm:prefix_cache_queries_total' for r in counters) and
                                all(r['value'] == 0 for r in counters))
                    if not pristine:
                        raise RuntimeError(f'{role}: missing idle gauges; refusing to reset')
                    report.setdefault('initial_gauges_unset', {})[role] = True
                observations[role] = selected
                busy |= any(row['value'] != 0 for row in selected)
            report['idle_gauges'] = observations
            now = time.monotonic()
            idle_since = None if busy else (idle_since if idle_since is not None else now)
            if idle_since is not None and now - idle_since >= idle_seconds:
                break
            if now >= deadline:
                raise TimeoutError('Service did not become idle before prefix cache reset')
            time.sleep(min(1., deadline - now))
        report['status'] = 'resetting'
        endpoint = url.rstrip('/') + '/reset_prefix_cache?reset_running_requests=false&reset_external=false'
        report['url'] = endpoint
        request = urllib.request.Request(endpoint, data=b'', headers=headers, method='POST')
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        report['response'] = result
        if not isinstance(result, dict) or result.get('success') is not True:
            raise RuntimeError(f'Prefix cache reset was not successful: {result!r}')
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise RuntimeError(f'Prefix cache reset failed; load was not started: {exc}') from exc
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
