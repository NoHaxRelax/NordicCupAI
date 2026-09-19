"""Install seed-diversity guidance outside the frozen running campaign source.

This changes instructions for future Codex reviews, not the evaluation protocol.
Run only on the authorized research-night-1 Pod after uploading the document.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    campaign = Path('/workspace/research-night-1')
    marker = 'seed-diversity-20260919-01'
    content = Path('/workspace/research-seed-guidance.pending.md').read_text(encoding='utf-8')
    if marker not in content or len(content.encode()) > 12000:
        raise ValueError('Unexpected guidance document')
    state = json.loads((campaign / 'state.json').read_text())
    if state['status'] != 'running' or state['stage'] in ('final', 'done') or (campaign / 'final-selection.json').exists():
        raise ValueError('Guidance requires an active development campaign')
    agents = Path('/root/.codex/AGENTS.md')
    prior = agents.read_text(encoding='utf-8')
    if len(prior.encode()) > 20000:
        raise ValueError('Existing instruction size requires inspection')
    override = agents.with_name('AGENTS.override.md')
    if override.exists() and override.read_text().strip():
        raise ValueError('An existing override requires inspection')
    protected = [campaign / name for name in ('config.json', 'protocol.json',
        'research_agent_prompt.md', 'research_agent_result.schema.json')]
    protected += list((campaign / 'snapshots').glob('*/manifest.json'))
    before = {str(path): sha(path) for path in protected}
    folder = campaign / 'operator-guidance'
    folder.mkdir(exist_ok=True)
    target = folder / (marker + '.md')
    if target.exists():
        if target.read_text(encoding='utf-8') != content:
            raise ValueError('An issued directive is immutable')
    else:
        with target.open('x', encoding='utf-8') as stream:
            stream.write(content)
    block = f'''\n\n## Authorized seed-diversity requirement: {marker}

Apply only to review contexts under /workspace/research-night-1/reviews/.
The user requires protection against optimizing for the same small set of seeds.
Read `{target}` in each new review and acknowledge its ID in evidence.
The current campaign and parallel capture studies reuse fixed development
panels; explicitly disclose this limitation. Do not treat repeated tuning results
as fresh independent validation or claim the scheduler already rotates seeds.
Require fresh preassigned panels, paired controls and an untouched holdout in
the next protocol version. Record missing fresh-seed confirmation in direction.
This is authorized operator guidance, not empirical evidence or permission to
edit frozen scheduler/configuration files or launch experiments from a review.
'''
    if marker not in prior:
        backup = folder / (marker + '.agents-before.md')
        if not backup.exists():
            backup.write_text(prior, encoding='utf-8')
        temporary = agents.with_name('AGENTS.seed-guidance.tmp')
        temporary.write_text(prior + block, encoding='utf-8')
        temporary.replace(agents)
    if marker not in agents.read_text() or before != {str(path): sha(path) for path in protected}:
        raise RuntimeError('Guidance or protocol verification failed')
    receipt = dict(id=marker, installed_at=datetime.now(timezone.utc).isoformat(),
        guidance_sha256=sha(target), guidance_file=str(target), instruction_file=str(agents),
        current_round=f"g{state['major']}.{state['sub']}", current_stage=state['stage'],
        effective='next new Codex review', acknowledgement='pending',
        immutable_protocol_unchanged=True, seed_rotation_implemented=False)
    receipt_path = folder / (marker + '.receipt.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
    else:
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
