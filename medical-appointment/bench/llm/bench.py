"""Offline answering bench: 390 training questions against an OpenAI-compatible
chat server (vLLM, Ollama, ...) on cached transcripts, no audio, no HTTP api.py.

For every conversation: load transcripts/<stem>.<asr>.json, build the same
sentence units model.py serves with, then send its ten questions concurrently
(--workers) to POST <url>/chat/completions with the prompt variant from
bench/llm/prompts.py. Answers are scored with utils.temporal_iou / gold_evidence
through local_evaluator.Statistics, so the report block is the one
local_evaluator.py prints, plus yes-rate, error count, token and latency lines.
Per-conversation wall time is measured under the given concurrency and compared
with the 60 s serving budget (LLM only: ASR time is not included here).

Output: bench/results/llm/<model short>.<variant>.<asr>[<suffix>].json with
config, summary and one record per question (answer, span, tIoU, latency, raw JSON),
written once the run finishes. After every conversation the same document is
checkpointed to <out stem>.partial.json (summary.partial = true; written to a .tmp
and os.replace'd, so it is never half-written), and the checkpoint is removed when
the final file lands. A crash, a bkill (SIGINT/SIGTERM) or Ctrl-C therefore keeps
what was done, while a resubmitted bench/hpc/llm_bench.lsf (which skips an existing
<out>) redoes the unfinished run instead of mistaking the checkpoint for a result.
A transcript that fails to load is skipped and listed under summary.skipped; a
question whose prompt, request or JSON fails is recorded as unanswered. Neither
aborts the run.

Install (Linux venv):
    pip install requests pydantic            # pydantic: local_evaluator imports dtos.py
    (vLLM itself: see bench/llm/serve_vllm.sh)

Examples:
    # vLLM on the cluster: --no-think auto resolves to vllm (thinking off through the
    # chat template), schema-constrained JSON
    python bench/llm/bench.py --asr large-v3 --variant units --model Qwen/Qwen3.6-27B
    # Ollama on the laptop, small smoke run: --no-think auto resolves to ollama (port 11434)
    python bench/llm/bench.py --asr large-v3 --variant words --url http://localhost:11434/v1 \\
        --model qwen3:4b --workers 4 --limit 2
    # print the first prompt and exit (no server needed)
    python bench/llm/bench.py --asr large-v3 --variant units-claim --print-prompt

Verified API surfaces (2026-09-17):
  - response_format {"type":"json_schema","json_schema":{"name":..,"schema":..}}:
        https://docs.vllm.ai/en/latest/features/structured_outputs/
    Ollama 0.32.1 /v1/chat/completions also accepted this shape and it constrained output
    when probed live on qwen3:4b, but this is NOT confirmed by docs.ollama.com/api/openai-
    compatibility (that page documents only a bare response_format / reasoning_effort, no
    json_schema) -- treat the Ollama half of this claim as unverified, live-probe-only.
  - vLLM thinking off per request: body["chat_template_kwargs"] = {"enable_thinking": false}
    (the openai client's extra_body is merged into the JSON body; with requests we set it directly):
        https://docs.vllm.ai/en/latest/features/reasoning_outputs/
  - Ollama thinking off: "reasoning_effort": "none" on /v1/chat/completions
        https://docs.ollama.com/api/openai-compatibility  (probed live on qwen3:4b: no reasoning,
        content is the JSON). There is no "/no_think" mechanism in Ollama's OpenAI layer;
        the soft switch "/no_think" in the prompt is a Qwen3 chat-template convention that
        Ollama does not document, so it is not used.
  - Ollama json_schema is NOT documented at docs.ollama.com/api/openai-compatibility (that
    page only lists a bare response_format / reasoning_effort, no json_schema shape); the
    line above claiming it is "also accepted by Ollama" rests solely on an unrepeatable
    live probe, not on the cited doc. Treat it as unverified for Ollama (tonight's cluster
    run is vLLM only, where json_schema IS documented). --json-mode's 400 -> json_object
    fallback already covers a server that rejects it.

Caveats:
  - --no-think defaults to auto: 'ollama' when --url has port 11434, else 'vllm'. Qwen3.5/3.6/3.8
    "operate in thinking mode by default" (HF cards); with thinking on and --max-tokens 200 the
    content is empty and every question scores 0. After the first conversation the bench
    therefore warns loudly when the mean completion length reaches 0.9 * max_tokens or most
    answers stop with finish_reason "length". Use --no-think none only on purpose (and with a
    large --max-tokens). serve_vllm.sh additionally sets the server-side default
    enable_thinking=false for Qwen3.x, so a stray 'none' is still safe there.
  - Ollama's OpenAI layer cannot set num_ctx; the model's default context applies
    (a consultation prompt is ~1.5k tokens, fine). Use the native /api/chat (model.py)
    if a transcript ever exceeds it.
  - --json-mode object relies on the JSON shape in the system prompt; the schema is
    appended to the system prompt in that mode. If the server 400s on json_schema the
    bench switches to object mode once and says so in the report.
  - A request that fails (transport, non-2xx, unparsable JSON) is recorded as
    unanswered (counted wrong, span missing), unlike model.py which guesses yes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics as st
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

HERE = Path(__file__).resolve().parent            # bench/llm
CASE = HERE.parent.parent                         # medical-appointment
for p in (str(CASE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import model                                      # noqa: E402  serving code
from model import Word, make_units                # noqa: E402
from utils import gold_evidence, group_questions_by_conversation, temporal_iou  # noqa: E402
from local_evaluator import (                     # noqa: E402  same report block
    REQUEST_TIMEOUT_SECONDS, UNANSWERED, YES, Statistics)
from prompts import VARIANTS                      # noqa: E402  bench/llm/prompts.py

TRANSCRIPTS = CASE / 'transcripts'
RESULTS = CASE / 'bench' / 'results' / 'llm'
Span = Tuple[float, float]
TRUNCATION_FRACTION = 0.9      # mean completion >= this * max_tokens after conversation 1 -> warning


# --------------------------------------------------------------------------- #
# Transcripts
# --------------------------------------------------------------------------- #

def load_words(path: Path) -> Tuple[List[Word], float, dict]:
    """Word stream + duration from a cached transcript. Mirrors
    model._cached_transcript (segment boundaries marked with \\x00) so the
    units are the ones the serving path would build. Raises on a malformed
    file (missing duration, word without start/end); main() catches that per
    conversation and skips the file."""
    d = json.loads(path.read_text(encoding='utf-8'))
    words: List[Word] = []
    for s in d['segments']:
        for w in s.get('words', []):
            words.append(Word(w['w'], float(w['start']), float(w['end'])))
        if s.get('words'):
            words[-1].w += '\x00'
    return words, float(d['duration']), d


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.S)
_THINK = re.compile(r'<think>.*?</think>', re.S)


def parse_json(content: str) -> dict:
    s = _THINK.sub('', content or '').strip()
    s = _FENCE.sub('', s).strip()
    try:
        out = json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find('{'), s.rfind('}')
        if a < 0 or b <= a:
            raise
        out = json.loads(s[a:b + 1])
    if not isinstance(out, dict):
        raise ValueError(f'model returned {type(out).__name__}, not an object')
    return out


class Client:
    def __init__(self, url: str, model_name: str, timeout: float, no_think: str,
                 json_mode: str, max_tokens: int, temperature: float, workers: int):
        self.url = url.rstrip('/')
        self.model = model_name
        self.timeout = timeout
        self.no_think = no_think
        self.json_mode = json_mode
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fell_back = False
        self._lock = threading.Lock()
        self.session = requests.Session()
        self.session.mount('http://', HTTPAdapter(pool_maxsize=max(10, workers), pool_connections=4))
        self.session.mount('https://', HTTPAdapter(pool_maxsize=max(10, workers), pool_connections=4))

    def body(self, system: str, user: str, schema: dict, mode: str) -> dict:
        if mode == 'object':
            system = system + '\nThe JSON must match this schema exactly:\n' + json.dumps(schema)
        b: dict = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': user}],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'stream': False,
            # per-token log-probabilities, so run_question can read P(yes) at the
            # answer token (Ollama and vLLM both honour these OpenAI fields)
            'logprobs': True,
            'top_logprobs': 6,
        }
        if mode == 'schema':
            # https://docs.vllm.ai/en/latest/features/structured_outputs/  (OpenAI json_schema form)
            b['response_format'] = {'type': 'json_schema',
                                    'json_schema': {'name': 'answer', 'schema': schema}}
        else:
            b['response_format'] = {'type': 'json_object'}
        if self.no_think == 'vllm':
            # https://docs.vllm.ai/en/latest/features/reasoning_outputs/  extra_body -> top-level key
            b['chat_template_kwargs'] = {'enable_thinking': False}
        elif self.no_think == 'ollama':
            # https://docs.ollama.com/api/openai-compatibility  reasoning_effort: "none"
            b['reasoning_effort'] = 'none'
        return b

    def chat(self, system: str, user: str, schema: dict) -> Tuple[str, dict]:
        """Returns (content, full response JSON). Falls back from json_schema to
        json_object once, for the whole run, if the server rejects the schema."""
        mode = self.json_mode
        r = self.session.post(f'{self.url}/chat/completions',
                              json=self.body(system, user, schema, mode), timeout=self.timeout)
        if r.status_code == 400 and mode == 'schema':
            with self._lock:
                if not self.fell_back:
                    self.fell_back = True
                    self.json_mode = 'object'
                    print(f'  server rejected response_format json_schema ({r.text[:160]!r}); '
                          'falling back to json_object + schema in the prompt', file=sys.stderr)
            r = self.session.post(f'{self.url}/chat/completions',
                                  json=self.body(system, user, schema, 'object'), timeout=self.timeout)
        r.raise_for_status()
        resp = r.json()
        return resp['choices'][0]['message'].get('content') or '', resp


# --------------------------------------------------------------------------- #
# One question
# --------------------------------------------------------------------------- #

def p_yes_from_logprobs(resp: dict) -> Optional[float]:
    """P(yes) at the answer token of a JSON reply: find the token that carries
    the "answer" key, then the first following token that is yes or no, and
    normalise the probabilities of yes and no among the sampled token and its
    top alternatives. None when the server sent no log-probabilities."""
    import math
    try:
        toks = ((resp.get('choices') or [{}])[0].get('logprobs') or {}).get('content') or []
    except AttributeError:
        return None
    seen_key = False
    for t in toks:
        s = (t.get('token') or '')
        if not seen_key:
            if 'answer' in s.lower():
                seen_key = True
            continue
        k = s.strip().strip('"').strip(':').strip().lower()
        if k in ('yes', 'no'):
            probs = {k: math.exp(t.get('logprob', -99.0))}
            for alt in t.get('top_logprobs') or []:
                a = (alt.get('token') or '').strip().strip('"').lower()
                if a in ('yes', 'no') and a not in probs:
                    probs[a] = math.exp(alt.get('logprob', -99.0))
            py, pn = probs.get('yes', 0.0), probs.get('no', 0.0)
            return py / (py + pn) if (py + pn) > 0 else None
    return None


def run_question(client: Client, variant, row: dict, units, words, duration: float) -> dict:
    rec: dict = {'question_id': row['question_id'], 'transcript_id': row['transcript_id'],
                 'question': row['question'], 'question_type': row['question_type'],
                 'label': int(row['label']),
                 'raw_content': None, 'usage': None, 'finish_reason': None, 'raw': None}
    gold = gold_evidence(row)
    rec['gold'] = list(gold) if gold else None
    t0 = time.time()
    try:
        # The prompt build is inside the try as well: a variant that trips on one
        # question must record an error for that question, not abort the run.
        p = variant(row['question'], units)
        content, resp = client.chat(p.system, p.user, p.schema)
        rec['raw_content'] = content
        rec['usage'] = resp.get('usage')
        rec['finish_reason'] = (resp.get('choices') or [{}])[0].get('finish_reason')
        out = parse_json(content)
        rec['raw'] = out
        rec['p_yes'] = p_yes_from_logprobs(resp)
        yes, span = p.postprocess(out, units, words, duration)
        rec['answer'] = bool(yes)
        rec['span'] = list(span) if span else None
        rec['error'] = None
    except Exception as exc:                        # prompt, transport, 4xx/5xx, bad JSON
        rec.update({'answer': None, 'span': None, 'error': f'{type(exc).__name__}: {exc}'})
    rec['latency_ms'] = (time.time() - t0) * 1000
    pred = UNANSWERED if rec['answer'] is None else int(rec['answer'])
    rec['prediction'] = pred
    rec['correct'] = pred == rec['label']
    rec['tiou'] = temporal_iou(gold, tuple(rec['span'])) if (gold and rec['span']) else (0.0 if gold else None)
    return rec


def run_conversation_joint(client: Client, variant, rows: List[dict], units, words, duration: float) -> List[dict]:
    """Joint variants: one request carries all questions of the conversation; the
    answer is split per question and post-processed like the 'units' variant."""
    recs = []
    for row in rows:
        recs.append({'question_id': row['question_id'], 'transcript_id': row['transcript_id'],
                     'question': row['question'], 'question_type': row['question_type'],
                     'label': int(row['label']), 'raw_content': None, 'usage': None,
                     'finish_reason': None, 'raw': None, 'p_yes': None})
    t0 = time.time()
    try:
        p = variant.build_all([r['question'] for r in rows], units)
        content, resp = client.chat(p.system, p.user, p.schema)
        out = parse_json(content)
        per = variant.split(out, len(rows))
        for rec, item in zip(recs, per):
            rec['raw_content'] = content if rec is recs[0] else None
            rec['usage'] = resp.get('usage') if rec is recs[0] else None
            rec['finish_reason'] = (resp.get('choices') or [{}])[0].get('finish_reason')
            rec['raw'] = item
            yes, span = p.postprocess(item, units, words, duration)
            rec['answer'] = bool(yes)
            rec['span'] = list(span) if span else None
            rec['error'] = None
    except Exception as exc:
        for rec in recs:
            rec.update({'answer': None, 'span': None, 'error': f'{type(exc).__name__}: {exc}'})
    wall = (time.time() - t0) * 1000
    for rec, row in zip(recs, rows):
        gold = gold_evidence(row)
        rec['gold'] = list(gold) if gold else None
        rec['latency_ms'] = wall / len(rows)
        pred = UNANSWERED if rec['answer'] is None else int(rec['answer'])
        rec['prediction'] = pred
        rec['correct'] = pred == rec['label']
        rec['tiou'] = temporal_iou(gold, tuple(rec['span'])) if (gold and rec['span']) else (0.0 if gold else None)
    return recs


def truncation_warning(recs: List[dict], max_tokens: int, no_think: str) -> Optional[str]:
    """Thinking models that were not switched off spend the whole completion
    budget on <think> and return no content. Detect that on the first
    conversation instead of discovering 390 zeros in the report."""
    ctok = [r['usage']['completion_tokens'] for r in recs
            if r.get('usage') and 'completion_tokens' in r['usage']]
    if not ctok:
        return None
    cut = sum(1 for r in recs if r.get('finish_reason') == 'length')
    mean = st.mean(ctok)
    if mean < TRUNCATION_FRACTION * max_tokens and cut <= len(recs) // 2:
        return None
    return (f'mean completion {mean:.0f} tokens of max_tokens {max_tokens} and {cut}/{len(recs)} '
            f'answers stopped at the limit after the first conversation: outputs are truncated '
            f'(thinking still on? no-think is {no_think!r}; use --no-think vllm|ollama, or raise --max-tokens)')


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def model_short(name: str) -> str:
    return re.sub(r'[^a-z0-9.+-]+', '-', name.split('/')[-1].lower()).strip('-')


def pct(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def result_paths(a: argparse.Namespace) -> Tuple[Path, Path]:
    """(final result file, checkpoint file). The checkpoint carries the same
    document with summary.partial = true and lives next to the final file as
    <stem>.partial.json, so a job that resumes by checking for <out> (bench/hpc/
    llm_bench.lsf) does not mistake an interrupted run for a finished one."""
    out = Path(a.out) if a.out else RESULTS / f'{model_short(a.model)}.{a.variant}.{a.asr}{a.suffix}.json'
    return out, out.with_name(out.stem + '.partial' + out.suffix)


def write_results(a: argparse.Namespace, client: Client, stats: Statistics, records: List[dict],
                  walls: List[float], skipped: List[str], warnings: List[str], total_wall: float,
                  partial: bool) -> Tuple[Path, str]:
    """Build the report (local_evaluator block + bench lines) and write the
    results JSON atomically (<target>.tmp, then os.replace). Called after every
    conversation with partial=True (checkpoint file) and once at the end with
    partial=False (final file; the checkpoint is then removed)."""
    n = stats.total or 1
    yes_rate = sum(1 for r in records if r['prediction'] == YES) / n
    yes_by_type: Dict[str, str] = {}
    for qt in ('positive', 'hard_negative', 'off_topic'):
        rs = [r for r in records if r['question_type'] == qt]
        if rs:
            yes_by_type[qt] = f'{sum(1 for r in rs if r["prediction"] == YES) / len(rs):.3f}'
    lat = [r['latency_ms'] for r in records]
    ptok = [r['usage']['prompt_tokens'] for r in records if r.get('usage') and 'prompt_tokens' in r['usage']]
    ctok = [r['usage']['completion_tokens'] for r in records if r.get('usage') and 'completion_tokens' in r['usage']]
    cut = sum(1 for r in records if r.get('finish_reason') == 'length')
    extra = ['', 'Bench' + ('  (PARTIAL: the run did not finish)' if partial else ''),
             f'  {"model":<24} {a.model}  variant {a.variant}  asr {a.asr}',
             f'  {"yes-rate":<24} {yes_rate:.3f}  by type {yes_by_type}',
             f'  {"span policy":<24} spans on every question: mean tIoU {stats.mean_tiou:.3f}, score {stats.final_score:.3f}'
             + (f'  |  nulls on no: mean tIoU {stats.nulls_on_no.mean_tiou:.3f}, score {stats.nulls_on_no.final_score:.3f}'
                if hasattr(stats, 'nulls_on_no') else ''),
             f'  {"request errors":<24} {sum(1 for r in records if r["error"])}',
             f'  {"json mode":<24} {a.json_mode}' + ('  (fell back to json_object)' if client.fell_back else ''),
             f'  {"no-think":<24} {a.no_think}',
             f'  {"offsets":<24} start {model.START_OFFSET:+.2f}s end {model.END_OFFSET:+.2f}s  pause split {model.PAUSE_SPLIT}s']
    if lat:
        extra.append(f'  {"per question latency":<24} {st.mean(lat):8.0f} ms mean, {pct(lat, .5):8.0f} p50, '
                     f'{pct(lat, .95):8.0f} p95')
    if walls:
        extra.append(f'  {"per conversation wall":<24} {st.mean(walls):6.2f} s mean, {pct(walls, .5):6.2f} p50, '
                     f'{max(walls):6.2f} worst  (LLM only, workers={a.workers}, budget {REQUEST_TIMEOUT_SECONDS} s)')
    if ptok and ctok:
        extra.append(f'  {"tokens":<24} prompt {st.mean(ptok):.0f} mean, completion {st.mean(ctok):.1f} mean '
                     f'(max {max(ctok)}; {cut} stopped at max_tokens {a.max_tokens})')
    for w in warnings:
        extra.append(f'  {"WARNING":<24} {w}')
    if skipped:
        extra.append(f'  {"skipped":<24} {len(skipped)} conversation(s), see summary.skipped')
    extra.append(f'  {"total wall":<24} {total_wall:.1f} s for {stats.conversations} conversations')
    report = stats.report() + '\n' + '\n'.join(extra)

    out, checkpoint = result_paths(a)
    target = checkpoint if partial else out
    target.parent.mkdir(parents=True, exist_ok=True)
    by_type = {k: {'correct': v[0], 'total': v[1], 'accuracy': v[0] / v[1]}
               for k, v in stats.by_type.items() if v[1]}
    summary = {
        'partial': partial,
        'questions': stats.total, 'correct': stats.correct, 'unanswered': stats.errors,
        'conversations': stats.conversations, 'accuracy': stats.accuracy,
        'accuracy_by_type': by_type, 'mean_tiou': stats.mean_tiou, 'n_annotated': len(stats.tious),
        'missing_spans': stats.missing_spans, 'tiou_answered_yes': stats.mean_tiou_answered_yes,
        'n_answered_yes_annotated': len(stats.tious_answered_yes), 'score': stats.final_score,
        'nulls_on_no': ({'mean_tiou': stats.nulls_on_no.mean_tiou, 'score': stats.nulls_on_no.final_score,
                         'missing_spans': stats.nulls_on_no.missing_spans}
                        if hasattr(stats, 'nulls_on_no') else None),
        'yes_rate': yes_rate, 'yes_rate_by_type': {k: float(v) for k, v in yes_by_type.items()},
        'latency_ms': {'mean': st.mean(lat) if lat else None, 'p50': pct(lat, .5), 'p95': pct(lat, .95)},
        'conversation_wall_s': {'mean': st.mean(walls) if walls else None, 'p50': pct(walls, .5),
                                'worst': max(walls) if walls else None, 'budget': REQUEST_TIMEOUT_SECONDS},
        'tokens': {'prompt_mean': st.mean(ptok) if ptok else None,
                   'completion_mean': st.mean(ctok) if ctok else None,
                   'stopped_at_max_tokens': cut},
        'total_wall_s': total_wall, 'skipped': skipped, 'warnings': warnings,
        'json_fallback': client.fell_back,
    }
    config = {'model': a.model, 'variant': a.variant, 'asr': a.asr, 'url': a.url, 'workers': a.workers,
              'limit': a.limit, 'no_think': a.no_think, 'json_mode': a.json_mode,
              'max_tokens': a.max_tokens, 'temperature': a.temperature, 'timeout': a.timeout,
              'start_offset': model.START_OFFSET, 'end_offset': model.END_OFFSET,
              'pause_split': model.PAUSE_SPLIT, 'date': time.strftime('%Y-%m-%d %H:%M:%S')}
    tmp = target.with_name(target.name + '.tmp')
    tmp.write_text(json.dumps({'config': config, 'summary': summary, 'report': report,
                               'questions': records}, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, target)                         # atomic on POSIX and on NTFS
    if not partial and checkpoint.exists():
        checkpoint.unlink()                         # superseded by the final file
    return target, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--asr', required=True, help='transcript tag, e.g. large-v3 (bench/README.md)')
    ap.add_argument('--variant', default='units', choices=sorted(VARIANTS))
    ap.add_argument('--url', default='http://localhost:8000/v1', help='OpenAI-compatible base URL')
    ap.add_argument('--model', default='', help='model name as the server knows it (default: first of /models)')
    ap.add_argument('--workers', type=int, default=10, help='concurrent questions per conversation')
    ap.add_argument('--limit', type=int, default=0, help='only the first N conversations')
    ap.add_argument('--no-think', default='auto', choices=['auto', 'vllm', 'ollama', 'none'],
                    help='how to switch thinking off; auto = ollama when --url has port 11434, else vllm')
    ap.add_argument('--json-mode', default='schema', choices=['schema', 'object'])
    ap.add_argument('--max-tokens', type=int, default=200)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--timeout', type=float, default=120.0, help='seconds per request')
    ap.add_argument('--out', default='', help='result path (default bench/results/llm/<model>.<variant>.<asr>.json)')
    ap.add_argument('--suffix', default='', help='appended to the default result name, e.g. .off0')
    ap.add_argument('--verbose', action='store_true', help='one line per question')
    ap.add_argument('--print-prompt', action='store_true', help='print the first prompt and exit')
    a = ap.parse_args()

    no_think_note = ''
    if a.no_think == 'auto':
        a.no_think = 'ollama' if ':11434' in a.url else 'vllm'
        no_think_note = ' (auto, from --url)'

    variant = VARIANTS[a.variant]
    conversations = group_questions_by_conversation()
    if a.limit:
        conversations = conversations[:a.limit]

    if a.print_prompt:
        fn, rows = conversations[0]
        words, duration, _ = load_words(TRANSCRIPTS / f'{Path(fn).stem}.{a.asr}.json')
        if hasattr(variant, 'set_conversation'):
            variant.set_conversation(Path(fn).stem, a.asr)
        p = variant(rows[0]['question'], make_units(words))
        print('--- SYSTEM ---\n' + p.system + '\n--- USER ---\n' + p.user +
              '\n--- SCHEMA ---\n' + json.dumps(p.schema, indent=1))
        return 0

    if not a.model:
        r = requests.get(f'{a.url.rstrip("/")}/models', timeout=10)
        r.raise_for_status()
        a.model = r.json()['data'][0]['id']
        print(f'model: {a.model} (first of {a.url}/models)')
    print(f'no-think: {a.no_think}{no_think_note}; max_tokens {a.max_tokens}; json mode {a.json_mode}')

    client = Client(a.url, a.model, a.timeout, a.no_think, a.json_mode, a.max_tokens,
                    a.temperature, a.workers)
    stats = Statistics()
    # Second scoring of the same run under the other span policy: a question
    # answered no returns no span. The primary `stats` keeps every located span
    # (model.SPAN_ON_NO=1 behaviour); the difference is what a paired
    # validation run would show if the live scorer credits spans on no.
    stats.nulls_on_no = Statistics()
    records: List[dict] = []
    walls: List[float] = []
    skipped: List[str] = []
    warnings: List[str] = []
    t_run = time.time()
    completed = False
    truncation_checked = False

    # bkill sends SIGINT, then SIGTERM, then SIGKILL. SIGINT is a KeyboardInterrupt
    # already; turn SIGTERM into one too so the finally below writes the results.
    def _terminate(signum, _frame):
        raise KeyboardInterrupt(f'signal {signum}')
    try:
        signal.signal(signal.SIGTERM, _terminate)
    except (ValueError, OSError, AttributeError):    # not the main thread / no SIGTERM here
        pass

    try:
        for fn, rows in conversations:
            tf = TRANSCRIPTS / f'{Path(fn).stem}.{a.asr}.json'
            if not tf.exists():
                skipped.append(f'{fn}: no transcript {tf.name}')
                continue
            try:
                words, duration, _ = load_words(tf)
                units = make_units(words) if words else []
            except Exception as exc:                # one malformed transcript must not abort the run
                skipped.append(f'{fn}: {tf.name}: {type(exc).__name__}: {exc}')
                print(f'  skipped {skipped[-1]}', file=sys.stderr, flush=True)
                continue
            if not words:
                skipped.append(f'{fn}: transcript has no word timestamps')
                continue
            if hasattr(variant, 'set_conversation'):      # few-shot variants: hold this conversation out
                variant.set_conversation(Path(fn).stem, a.asr)
            t0 = time.time()
            if getattr(variant, 'joint', False):
                recs = run_conversation_joint(client, variant, rows, units, words, duration)
            else:
                ex = ThreadPoolExecutor(max_workers=max(1, a.workers))
                try:
                    recs = list(ex.map(lambda r: run_question(client, variant, r, units, words, duration), rows))
                finally:
                    ex.shutdown(wait=False, cancel_futures=True)   # on interrupt: do not wait for in-flight requests
            wall = time.time() - t0
            walls.append(wall)
            failed = any(r['error'] for r in recs)
            stats.record_request(len(rows), wall * 1000, failed=failed)
            for r in recs:
                gold = tuple(r['gold']) if r['gold'] else None
                span = tuple(r['span']) if r['span'] else None
                stats.record(r['question_type'], r['label'], r['prediction'], gold, span)
                stats.nulls_on_no.record(r['question_type'], r['label'], r['prediction'], gold,
                                         span if r['prediction'] == 1 else None)
                records.append(r)
                if a.verbose:
                    mark = 'ok  ' if r['correct'] else 'WRONG'
                    said = {True: 'yes', False: 'no'}.get(r['answer'], '-')
                    ev = f' tIoU {r["tiou"]:.3f}' if gold else ''
                    err = f'  {r["error"][:80]}' if r['error'] else ''
                    print(f'  {mark} {r["question_id"]:<24} {r["question_type"]:<14} said {said:<3} '
                          f'wanted {"yes" if r["label"] else "no":<3}{ev}{err}')
            n_yes = sum(1 for r in recs if r['answer'])
            n_err = sum(1 for r in recs if r['error'])
            print(f'{fn}: {len(units)} units, {wall:5.1f}s wall for {len(rows)} questions '
                  f'(workers {a.workers}), yes={n_yes}/{len(rows)}'
                  + (f', errors={n_err}' if n_err else ''), flush=True)
            if not truncation_checked and any(r.get('usage') for r in recs):
                truncation_checked = True
                msg = truncation_warning(recs, a.max_tokens, a.no_think)
                if msg:
                    warnings.append(msg)
                    print(f'\n!!! WARNING: {msg}\n', file=sys.stderr, flush=True)
            # Checkpoint: <stem>.partial.json always holds every conversation done so far.
            write_results(a, client, stats, records, walls, skipped, warnings,
                          time.time() - t_run, partial=True)
        completed = True
    except KeyboardInterrupt as exc:
        print(f'\ninterrupted ({exc}); writing the partial results', file=sys.stderr, flush=True)
        return 130
    finally:
        total_wall = time.time() - t_run
        for s in skipped:
            print(f'  skipped {s}', file=sys.stderr)
        out, report = write_results(a, client, stats, records, walls, skipped, warnings,
                                    total_wall, partial=not completed)
        print(report)
        print(f'\nwrote {out}' + ('' if completed else '  (partial: the run did not finish)'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
