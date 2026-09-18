"""Package a portable snapshot; --all includes the current complete replay library."""
import argparse
import base64
import gzip
import json
from pathlib import Path
from catalog import ReplayCatalog

ROOT = Path(__file__).resolve().parent


def build(output, include_all=False):
    catalog = ReplayCatalog()
    entries = catalog.refresh()['recordings']
    if not include_all:
        entries = [entry for entry in entries if entry.get('renderer') == 'native'][:10]
    sources = [(entry, catalog.resolve(entry['file'])) for entry in entries]
    html = (ROOT/'index.html').read_text()
    css = (ROOT/'style.css').read_text()
    js = (ROOT/'app.js').read_text().replace('</script', '<\\/script')
    html = html.replace('<link rel="stylesheet" href="style.css">', '<style>'+css+'</style>')
    html = html.replace('<script src="app.js" defer></script>', '')
    before, after = html.split('</body>', 1)
    output = Path(output)
    # Stream bundles to avoid holding the entire all-runs export in memory.
    with output.open('w') as stream:
        stream.write(before+'<script>window.SURVIVAL_BUNDLES=[')
        for index, (entry, path) in enumerate(sources):
            if path is None or not path.is_relative_to(ROOT.parent):
                raise ValueError('Recording is missing or outside the survival workspace')
            payload = path.read_bytes()
            if path.suffix != '.gz':
                payload = gzip.compress(payload)
            bundle = dict(title=entry['title'], description=entry.get('description', ''),
                          group=entry.get('group', 'Bundled demonstrations'), source=entry.get('source', entry['file']),
                          data=base64.b64encode(payload).decode('ascii'))
            if index:
                stream.write(',')
            for chunk in json.JSONEncoder(separators=(',', ':'), ensure_ascii=True).iterencode(bundle):
                stream.write(chunk.replace('<', '\\u003c'))
        stream.write('];</script>\n<script>'+js+'</script>\n</body>'+after)
    print(f'{output} ({output.stat().st_size/1048576:.1f} MB; {len(sources)} recordings; offline snapshot)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'Survival Lab.html')
    parser.add_argument('--all', action='store_true', help='Include every discovered replay, including state-only recordings (can create a very large file). Default: the 10 newest organizer-rendered recordings.')
    args = parser.parse_args()
    build(args.output, args.all)
