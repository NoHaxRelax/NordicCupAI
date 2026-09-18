#!/usr/bin/env python3
"""Bounded drone API client. Deliberately has no final evaluation command."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # Never forward the team key to a redirect destination.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['status', 'verify', 'validate', 'queue', 'result'])
    p.add_argument('--key-file', type=Path, default=Path('/private/tmp/nordic-ai-cup-secrets/api-key'))
    p.add_argument('--url-file', type=Path)
    p.add_argument('--uuid')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    key = a.key_file.read_text().strip()
    if not key or '\n' in key:
        p.error('Expected one nonempty key in key file')
    paths = {'status': '/status', 'verify': '/verify', 'validate': '/validate/queue'}
    if a.action in ('queue', 'result'):
        if not a.uuid or any(c not in '0123456789abcdefABCDEF-' for c in a.uuid):
            p.error('A queue UUID is required')
        path = '/validate/queue/' + a.uuid + ('/attempt' if a.action == 'result' else '')
    else:
        path = paths[a.action]
    body = None
    if a.action in ('verify', 'validate'):
        if not a.url_file:
            p.error('--url-file is required')
        body = json.dumps({'url': a.url_file.read_text().strip()}).encode()
    request = Request('https://cases.nordicaicup.com/api/v1/usecases/drone-flyby' + path,
                      data=body, headers={'x-token': key, 'Content-Type': 'application/json'})
    try:
        with build_opener(NoRedirect).open(request, timeout=45) as r:
            raw = r.read()
    except HTTPError as e:
        message = e.read().decode('utf-8', errors='replace').replace(key, '[redacted]')
        if a.url_file:
            message = message.replace(a.url_file.read_text().strip(), '[redacted endpoint]')
        try:
            detail = json.loads(message)
            if isinstance(detail, dict):
                detail.pop('sample', None)  # Verification embeds a large base64 PNG even on error.
        except json.JSONDecodeError:
            detail = message[:4000]
        error = {'http_status': e.code, 'detail': detail}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.with_suffix('.http-error.json').write_text(json.dumps(error, indent=2)+'\n')
        print(json.dumps(error))
        raise SystemExit(1)
    # Endpoint capability URLs can occur in results; redact before persisting.
    data = json.loads(raw)
    def redact(x):
        if isinstance(x, dict):
            return {k: ('[redacted endpoint]' if k in ('url', 'service_url') else redact(v)) for k,v in x.items()}
        if isinstance(x, list):
            return [redact(v) for v in x]
        if isinstance(x, str):
            x = x.replace(key, '[redacted]')
            if a.url_file:
                x = x.replace(a.url_file.read_text().strip(), '[redacted endpoint]')
        return x
    data = redact(data)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, indent=2) + '\n')
    def summary(x):
        if isinstance(x, dict):
            return {k: ('[image payload saved locally]' if k == 'image' else summary(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [summary(v) for v in x]
        return x
    print(json.dumps(summary(data), indent=2))


if __name__ == '__main__':
    main()
