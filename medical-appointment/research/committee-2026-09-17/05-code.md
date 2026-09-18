# Committee 05: code quality, correctness, reproducibility

Reviewer lens: a strong engineer joining tomorrow morning who has to run this without Elias
and without the agent. Written 2026-09-18 early hours, 67 commits into 2026-09-17, branch
`medical-appointment`. Read-only review: no bench was run, no network call was made, no
cluster or pod was touched. Every Python file listed in the brief byte-compiles.

## Verdict

The engineering is unusually disciplined for a hackathon: one transcript contract, one scoring
path through `utils.py` and `local_evaluator.py`, atomic writes with checkpoints everywhere,
failure-tolerant job scripts, and docstrings that cite the doc page they were verified
against. The problems are not sloppiness, they are the two classic ones: a configuration
coupling that silently corrupted the measurements, and a growing set of tools that only run on
Elias's laptop. The first is concrete and costs score. Every cluster bench run scored
large-v3-turbo transcripts with **large-v3's edge offsets**, because `bench.py` takes the
transcript tag from `--asr` but the offsets from `model.ASR_MODEL`, which nothing sets. I
re-scored the raw model outputs offline: the 0.780 headline is **0.793** under turbo's own
fitted rule, and joint-demo is 0.789 rather than 0.775. That is free score, recoverable
without a GPU, and it also means the ASR comparison in findings log 34 reading (4) is
confounded and the +0.010 joint gain in entry 32 mixes two edge rules. The second problem is
that a teammate cloning this repo tomorrow can reproduce none of today's three headline
numbers: `bench/results/` is gitignored (all 63 result files, including the 0.780, exist only
on one laptop), `request_dump/` is gitignored and is a hard import dependency of four tools,
the pod's three serving scripts live only on a stopped pod's volume, `model.py`'s defaults are
not the served configuration, and the handoff tells the next person to run a flag
(`bench.py --summary`) that does not exist. Highest-value next actions, in order: the offline
rescore (finding 1), commit the result summaries and a `serve.sh` (findings 15 and 16), fix
`Joint.split` before any joint variant is served (finding 3), and copy the pod scripts into
the repo (finding 17).

---

## A. Measurement integrity

### 1. `bench.py` scores every transcript tag with `model.py`'s default offsets

`bench/llm/bench.py:458` (`--asr` chooses the transcript file), `model.py:66` and
`model.py:77-83` (offsets are keyed on the `ASR_MODEL` environment variable, defaulting to
`large-v3`), `bench/llm/bench.py:445` (the run records the offsets it happened to use).

**What is wrong.** `--asr large-v3-turbo` loads turbo transcripts but leaves
`model.START_OFFSET = -0.14` and `model.END_OFFSET = +0.12`, which were fitted for large-v3.
Turbo's own fitted rule is `-0.20 / -0.02`. `prompts.py:34-36` documents the coupling as a
caveat, but nothing in `bench/hpc/llm_bench.lsf` exports `ASR_MODEL`, so no cluster run ever
got it right. Counting the recorded `config` blocks in `bench/results/llm/`:

| transcripts | offsets recorded | runs |
|---|---|---:|
| large-v3-turbo | -0.14 / +0.12 (large-v3's rule) | 19 |
| large-v3-turbo | -0.20 / -0.02 (turbo's own rule) | 6 |
| parakeet-tdt-0.6b-v2 | -0.14 / +0.12 (large-v3's rule) | 15 |
| large-v3 | -0.14 / +0.12 (correct) | 18 |

The six correct turbo runs are the early local qwen3:4b ones, where `ASR_MODEL` was exported
by hand. Everything from about 19:30 onward, including the whole cluster sweep, used the wrong
rule.

**Consequence.** Three separate damages. (a) Score left on the table: re-running the recorded
raw outputs through `model.anchor_ids` and `model.span_from_ids` with `ASR_MODEL=large-v3-turbo`
(the control run with `large-v3` reproduces the logged numbers to four decimals) gives
`qwen3.8-27b units-fewshot` 0.9897 / 0.6613 / **0.7927** against the logged 0.7801, and
`units-joint-demo` 0.9949 / 0.6514 / **0.7888** against 0.7748. About +0.013, two noise
floors, and the ranking of the top two variants does not change. (b) The cross-ASR comparison
is confounded: every tag was scored with large-v3's rule, which flatters large-v3, so "turbo
beats large-v3 by about 0.01 and Parakeet by 0.03-0.04" understates turbo and overstates
large-v3. (c) Within-variant comparisons in entries 30, 32 and 33 mix rules: the `units`
baseline quoted as 0.700 is `units-v2` under turbo's rule (0.7002) while `units-fewshot` 0.699
and `units-joint` 0.710 are under large-v3's rule, so the claimed +0.010 joint gain against a
0.006 noise floor is partly an offset artefact.

**Fix.** `bench.py` must resolve the offsets from `--asr`, not from the environment. argparse
runs after `import model`, so either set `os.environ['ASR_MODEL']` from `sys.argv` at the top
of the module, or in `main()` assign `model.START_OFFSET` and `model.END_OFFSET` from
`model._FITTED[a.asr]` and abort with a clear message when the tag has no fitted entry and no
explicit override. Then rescore the existing results offline (no GPU needed, snippet in the
how-to-run section) and correct the tables in entries 30, 32, 33, 34 and 35 before they are
quoted anywhere else.

### 2. The truncation detector is blind on exactly the runs it was built for

`bench/llm/bench.py:306-311` sets `rec['usage']` and `rec['finish_reason']` *after*
`parse_json(content)`; `bench.py:578` only tests for truncation when
`any(r.get('usage') for r in recs)`.

**What is wrong.** In the joint path, a truncated reply fails `parse_json`, the `except` at
`bench.py:317` records the error, and `usage` and `finish_reason` stay `None`. So
`truncation_checked` never flips, `truncation_warning` never runs, and the run finishes with
390 unanswered questions and no warning. `run_question:273-275` does it in the right order for
the per-question path, so the two code paths diverge on the one thing that bit today.

**Consequence.** This is today's cluster incident (findings 34 reading 6: the joint variants
"failed on the cluster only because the job kept bench.py's default 200-token budget") and the
detector built to catch it cannot fire. A repeat costs a queue slot and an H100 hour.

**Fix.** Move the three `rec[...] = ...` assignments above `parse_json` in
`run_conversation_joint`, exactly as `run_question` has them. Then drop the
`any(r.get('usage'))` gate and run `truncation_warning` on the first conversation
unconditionally, treating "every question errored and nothing reported usage" as its own loud
warning.

### 3. `Joint.split` mis-assigns every answer when the model numbers from zero

`bench/llm/prompts.py:472-482`.

**What is wrong.** Three independent defects, all verified by running the function:

- Zero-based numbering. The schema says `"q": <question number>` and nothing pins the base.
  Feeding `q = 0..9` returns question 1's answer under question 0 (dropped), shifts every
  other answer one position earlier, and leaves question 10 with the `{'answer': 'no',
  'segments': []}` default. Observed output for ten yes answers numbered 0..9:
  `['yes'] * 9 + ['no']`. No error, no warning, a plausible-looking score.
- A non-dict item (`{'answers': ['garbage']}`) raises `AttributeError: 'str' object has no
  attribute 'get'`, which is not in the `except (TypeError, ValueError)` list. It escapes to
  `run_conversation_joint`'s broad handler, so the whole conversation is lost rather than the
  one bad item skipped.
- Duplicate `q` values silently overwrite: `[{'q':1,...},{'q':1,...}]` with n=3 returns
  `[[7], [], []]`, the second entry winning and two questions left at the default.

**Consequence.** The first one is the most dangerous bug in the repo, because it fails silently
with numbers that look right, unlike the truncation incident which produced obvious zeros.
Every span would be attributed to the neighbouring question. If a joint variant is served (it
is the plan for the 27B: fewer tokens, 0.789 under the corrected rule), a model that switches
to zero-based numbering under a different sampler or chat template costs the whole attempt.

**Fix.** Validate the numbering rather than trusting it:

```python
@staticmethod
def split(out: dict, n: int) -> List[dict]:
    items = [x for x in (out.get('answers') or []) if isinstance(x, dict)]
    ks = [_int_or(x.get('q'), -10**6) for x in items]
    if len(items) == n and sorted(ks) == list(range(n)):      # zero-based: accept and shift
        ks = [k + 1 for k in ks]
    if sorted(ks) != list(range(1, n + 1)):
        if len(items) == n:
            return items                                      # positional fallback, in order
        raise ValueError(f'joint answers numbered {ks}, expected 1..{n}')
    per: List[dict] = [None] * n
    for k, x in zip(ks, items):
        per[k - 1] = x
    return per
```

Never return a silent all-no default: a missing answer must be an error the run records, since
`local_evaluator` counts an unanswered question wrong anyway and a fabricated "no" scores the
same while hiding the failure.

### 4. `span_from_ids` lets a stray low id decide the span

`model.py:312-324`.

**What is wrong.** The contiguous-run logic starts at `ids[0]` after sorting and stops at the
first gap wider than one unit. So for cited ids `[0, 7, 8]`, where 7 and 8 are the real
evidence and 0 is a stray citation, the run is `[0]` and the span points at unit 0. Verified:
`segments: [0, 3]` on a five-unit transcript yields the span of unit 0 alone. `anchor_ids`
usually saves this by putting the quote's unit first and dropping non-adjacent citations, but
when `locate_quote` returns `None` (quote shorter than 8 normalised characters, or best
`difflib` ratio below 0.5) it returns the raw cited list and the stray wins.

**Consequence.** A tIoU of zero on questions where the model cited the right sentence among
others. This is part of the 16 to 19 "wrong place" spans that every 27B row reports.

**Fix.** Keep the run that contains the quote anchor, otherwise the longest run: collect all
runs instead of breaking at the first gap, then
`run = next((r for r in runs if anchor_idx in r), max(runs, key=len))`, passing the anchor
index through from `anchor_ids` rather than re-deriving it. Cheap to validate: rescore the
existing result JSONs both ways with finding 1's snippet and keep whichever wins on the 195
annotated positives.

### 5. `anchor_ids` accepts JSON booleans as unit ids and rejects numeric strings

`model.py:304`.

**What is wrong.** `isinstance(i, (int, float))` is true for `bool`, so
`"segments": [true, false]` becomes ids `{0, 1}`. Verified: that input produces a real span
covering units 0 and 1. Conversely `"segments": ["2"]` is dropped and the question returns no
span at all. `prompts.py:176-185` has an `_int_list` helper that handles both correctly, and it
is dead code: nothing in the repo calls it, because `_units_post` delegates to
`model.anchor_ids` for parity.

**Consequence.** Small but silent. A schema-constrained server cannot emit either shape, so
this only bites in `json_object` fallback mode (`bench.py:213-219`), on Ollama where the
json_schema support is explicitly marked unverified at `bench.py:54-59`, or through
`dump_prompts.py` where a hand-written answer file is parsed with no schema at all.

**Fix.** Delete `_int_list` from `prompts.py` and move its logic into `model.anchor_ids`:
`if isinstance(i, bool): continue`, accept a `str` matching `^-?\d+$`, then range-check.

### 6. `_find`'s preference bonus overrides a verbatim exact match

`bench/llm/prompts.py:251-272` (`PREFER_BONUS = 0.15` added to windows inside the cited unit
plus or minus one) and `prompts.py:296-301`.

**What is wrong.** The bonus is applied to the ranking score but the function returns the raw
ratio, so a mediocre window inside a wrongly cited unit beats an exact match elsewhere.
Verified on a transcript containing both "The dose stays at ten milligrams" and "The dose
stays at twenty milligrams": searching for "dose stays at twenty" with no preference finds
index 11 at ratio 1.000; with the cited unit set to the wrong sentence it returns index 1 at
ratio 0.919, and `_words_post` produces the wrong sentence's span (0.36-3.02 instead of
5.36-8.02). The comment at `prompts.py:254` claims the bonus only wins "ties and near-ties";
0.15 is a quarter of the usable range above `MATCH_MIN = 0.72`, which is not a near-tie. There
is a second, milder form of the same defect: because `best_raw` belongs to the score winner, a
bonused window whose raw ratio lies between `best - 0.15` and `MATCH_MIN` is returned and then
rejected by the caller at `prompts.py:299`, discarding a valid match found elsewhere.

**Consequence.** Explains part of why `words` loses about 0.05 against `units` on every model
(findings 34 reading 5). The variant is not a serving candidate, so this is low priority, but
the bug sits in the half of the design meant to beat unit granularity, so the conclusion
"words loses" is not yet safe.

**Fix.** Do not bonus a window when another reaches an effectively exact match (return the
exact hit when `raw >= 0.98`), or apply the bonus only as a tie-break by comparing
`(round(raw, 2), preferred)` tuples. And reject on `score`, not on `raw`, or keep the best
above-threshold raw match as a fallback.

---

## B. Does the bench measure what serves

Short answer: for the `units` variant, almost exactly, by construction, and the design
(`bench/README.md:12-14`, `prompts.py:9-12`) deserves credit for it. For the variants that
actually win, no: there is no serving code path for them at all. Every divergence I found:

| # | serving (`model.py`) | bench (`bench.py` / `prompts.py`) | matters? |
|---|---|---|---|
| 1 | offsets from `ASR_MODEL` env, default large-v3 | transcripts from `--asr`, offsets inherited | **yes, finding 1: 0.013 of score** |
| 2 | Ollama native `POST /api/chat` (`model.py:263`) | OpenAI `POST /v1/chat/completions` | **yes**: pointing `LLM_URL` at vLLM cannot work, so the winning model needs a new client |
| 3 | one request per question, 10 threads (`model.py:360`) | same for `units*`, one request per conversation for `units-joint*` | **yes**: no joint path exists in serving |
| 4 | no few-shot and no demos anywhere in `model.py` | `units-fewshot` (the 0.780/0.793 winner) and `JointDemo` | **yes**: the winner is unimplemented in served code |
| 5 | `num_ctx` 3072 (`model.py:259`); Ollama truncates a longer prompt from the front | vLLM `--max-model-len 8192`, pod 98k | **yes**: `units-joint-demo` is 3,300 prompt tokens (entry 33), already over 3072; `units-joint-demo-all` is ~60k |
| 6 | `num_predict` 200 (`model.py:260`) | `--max-tokens`, default 200, 1500 for four model ids | **yes**: a ten-answer joint JSON needs about 1,200 |
| 7 | `think: False` (Ollama native) | `reasoning_effort: 'none'` or `chat_template_kwargs.enable_thinking` | equivalent in effect, different key |
| 8 | `json.loads(content)` raw (`model.py:266`) | `parse_json` strips `<think>` blocks and code fences | **yes, and the wrong way round**: the bench is the forgiving one, so it hides a failure mode the server would hit |
| 9 | a failed question guesses `(True, None)` (`model.py:357-358`) | a failed question is recorded unanswered (documented at `bench.py:75-76`) | yes, the bench is pessimistic by about half a mark per failure |
| 10 | `logprobs: True, top_logprobs: 6` sent, return value unused | `p_yes_from_logprobs` reads it | harmless, costs bytes |
| 11 | `keep_alive: -1` | not sent | Ollama-only, fine |
| 12 | temperature 0.0 | `--temperature`, default 0.0 | same |
| 13 | `SPAN_ON_NO=1`, one policy | both policies scored from one run (`bench.py:510`) | good, but see below |
| 14 | units built from live faster-whisper output | units built from cached transcripts | same code (`make_units`), different producer; see finding 12 |

Two more things worth naming. First, the headline metric differs between the two scoring tools:
`bench.py` prints `stats.final_score` (spans on every question) as primary and `nulls_on_no`
second, while `dump_prompts.py:98-103` reports only `nulls_on_no`, which is the policy every
findings-log table quotes. Someone reading a raw `bench.py` report will quote a number that is
0.005 too high (0.7849 against 0.7801 on the 27B fewshot run). Make `nulls_on_no` the primary
line in both, since findings 15 established the live scorer's policy. Second, `prompts.units`
reproduces `model.ask_llm`'s user message byte for byte, tag-question parenthetical included
(`prompts.py:113-120` against `model.py:240-244`), and `units_nooffset`'s known asymmetry is
documented at `prompts.py:37-40`. That is the standard the other variants should be held to.

### 7. Serving the bench winner needs code that does not exist yet, with hours left

`model.py` has no vLLM client, no few-shot block, no conversation-level request, and no
`ASR_MODEL` default matching the served ASR. `bench/hpc/RUNPOD.md:22-27` implies the plan is
vLLM on a pod plus the `units-fewshot` or `units-joint-demo-fewshot` prompt.

**Consequence.** The gap between "0.793 measured" and "0.793 served" is a new client, a prompt
builder shared with the bench (which `bench/README.md:12` currently forbids, reasonably), and
a re-validation. That is the critical path to Saturday, not more benching.

**Fix.** Invert the dependency now rather than at 14:00 on Saturday: move `FewShot`, `Joint`,
`JointDemo` and the shared system prompts out of `bench/llm/prompts.py` into a top-level
`prompting.py` that both `model.py` and the bench import, add an `LLM_API=ollama|openai` switch
in `ask_llm`, and have `answer_all` issue a single joint request when the variant is a joint
one. Then the bench and the server run the same builder and `LLM_NUM_CTX` stops being a hidden
ceiling. Budget this as the last code change and leave time for one validation run.

---

## C. Tooling correctness

### 8. `count_run.py` and `span_probe.py` identify "our attempt" by timestamp, not by uuid

`bench/mine/count_run.py:46-55`, `bench/mine/span_probe.py:112-131`. Both capture the set of
`submitted_at` values before queueing, then take `max(submitted_at)` among whatever is new.
`portal_status.queue_validation` returns `queued_attempt_uuid`, which `count_run.py:48` prints
and then never uses.

**What is wrong.** Any attempt appearing in the window is treated as ours: a teammate queueing
from the portal UI, a retry the portal performs itself, or two runs sharing a `submitted_at`
granularity. `span_probe.py:118` has the same pattern inside a retry loop.

**Consequence.** A mis-attributed attempt does not error, it returns a plausible number.
`count_run.py:62` converts it into a binary count and `span_probe.absorb` writes it into
`span_state.json`, which is committed and is the input to `val_diag.py`, `timeline.py` and
`clips.py`. A wrong gold span would propagate into the held-out diagnosis silently. The
`verify` stage (IoU >= 0.995) catches most systematic corruption by refusing to converge, but
it reports "stuck", not "the attempt you read was not yours".

**Fix.** One line each: keep `uuid = q.get('queued_attempt_uuid')` and select the attempt whose
uuid matches, falling back to the timestamp only when the status list does not echo a uuid.
Fail loudly when neither matches.

### 9. `span_probe.py` assumes the binaries are exactly 190/190 with no check

`bench/mine/span_probe.py:56` (`BASE = 0.4`) and `span_probe.py:130`
(`(score - BASE) * N_POS / 0.6`).

**What is wrong.** The arithmetic is valid only when accuracy is exactly 1.0. One wrong binary
makes the true base 0.39789 and every recovered IoU wrong by 0.33, far outside the tolerances
the stage logic uses. Nothing asserts it: the only guard is `errs` being empty.

**Consequence.** A single edit to `agent_answers.md` that flips a binary would silently poison
every subsequent recovery. The `verify` stage would eventually mark everything "stuck", which
is the right outcome for the wrong reason and costs a hundred portal runs to diagnose.

**Fix.** Make the first run of a session a null-span baseline (which `count_run.py` already
does), assert `abs(score - 0.4) < 1e-4`, and store the measured base in `span_state.json` so a
resumed session re-checks it. Separately, `span_probe.py:60` calls `TOL` a relative tolerance
while lines 180, 195 and 197 use it absolutely; either rename it or divide by the expected
value.

### 10. `span_probe.durations()` and `timeline.py` use the last segment end as the audio duration

`bench/mine/span_probe.py:63-68` (and it returns `1e9` when the stem is missing) and
`bench/timeline.py:38`. The transcript schema guarantees a top-level `duration`
(`bench/README.md:69`).

**What is wrong.** Measured across the 39 turbo transcripts, `duration` minus the last unit end
has a median of 0.20 s and a maximum of 0.64 s. `step()` clamps probes with `min(dur, e + W)`,
so a gold span inside that tail cannot be bracketed, and the `search` stage's whole-audio probe
`[0.0, dur]` stops short of the real end. Worse, a missing stem yields `dur = 1e9`, so the
whole-audio probe becomes `[0.0, 1000000000.0]`, the IoU rounds to zero, and the question is
marked "no overlap even with the whole audio" after burning a portal run.

**Fix.** Read `d['duration']` in both places and raise on a transcript that lacks it rather
than defaulting to `1e9`. `load_words` at `bench.py:119-132` already does this correctly and is
the model to copy.

### 11. `answers_md.py` maps hand answers to questions by position only

`bench/mine/answers_md.py:105-123`. The row index `i = r['q'] - 1` indexes `d['answers'][i]`
and `d['questions'][i]` from `request_dump/answers.jsonl`.

**What is wrong.** The markdown carries no question text, so nothing checks that row 3 of
`## conversation_sample_8` is the question the evaluator sent third. `dtos.py:166-171` promises
a stable order within one request but says nothing about order across attempts, and the probe
table in `probe_server.py:41-52` is positional too. A short dump (fewer than ten answers)
raises a bare `IndexError`.

**Consequence.** The whole 1.0 rests on this alignment. A re-dump with a different order would
produce a silently scrambled table.

**Fix.** Write the question text into `agent_answers.md` as a seventh column (or a comment
line) and assert it matches `d['questions'][i]` after normalising whitespace. Five lines, and
it turns the one unverified assumption behind the mined score into a checked one.

Two smaller items in the same file. `answers_md.py:52` sets `check` from `'CHECK' in line`,
which matches anywhere in the raw row including the note and the question text, so a note
reading "UNCHECKED" marks the row checked. And `--gold` (lines 88-96) rewrites `r['start']` and
`r['end']` from `span_state.json` regardless of `--spans`, so `--gold` alone writes recovered
gold spans into the tracked `bench/mine/agent_labels/*.json`; that is how the gold spans
entered git in commit 63f198a.

### 12. `make_units` depends on the ASR keeping leading spaces, with no validation

`model.py:186` joins word strings with `''.join(w.w for w in cur)`. The contract is documented
(`bench/README.md:70-72`: "a leading space if the tokenizer produces one") and never checked.

**What is wrong.** A runner that strips or normalises word text produces a glued transcript
("YourHbA1cis47mmol/mol.") and the LLM sees nonsense while every downstream number still looks
computable. `align_mms.py` re-times an existing transcript, and any new runner is one commit
away from this.

**Consequence.** A whole ASR tag's bench results would be garbage with no error anywhere.

**Fix.** Assert it once where the words are loaded (`load_words` and `_cached_transcript`): if
most words lack a leading space, or the rendered transcript has far fewer spaces than words,
refuse the file. See test 4.

Related observation, offered as data rather than as a defect: with `PAUSE_SPLIT = 0.6`
(`model.py:196`), 95 of 2,305 turbo units (4.1%) begin mid-sentence and 63 end in a token that
looks like an abbreviation. Since findings 35 and 37 both conclude the remaining loss is about
55 "shifted" spans and that a better selector does not fix them, the unit boundaries are a
plausible suspect worth one ablation (`PAUSE_SPLIT=1.2`, or refuse to split at a pause unless
the next word starts with a capital). Note this changes the representation, so the offsets
would need refitting and the existing raw outputs cannot be reused, which makes it a GPU run
rather than a free rescore.

### 13. `bench.py` drops a non-joint variant's demos, and the joint latency figure is fiction

`bench/llm/bench.py:272` calls `client.chat(p.system, p.user, p.schema)` with no fourth
argument, while `Prompt` carries a `demos` field (`prompts.py:73`) and the joint path does pass
it (`bench.py:305`). Today's `FewShot` inlines its examples in the user turn, so nothing is
lost yet; the next variant that uses prior turns will be silently benched without them. One
character fix: `getattr(p, 'demos', None)`.

`bench.py:324` sets each joint record's `latency_ms` to `wall / len(rows)`, so the "per
question latency mean/p50/p95" line describes a tenth of one request. Report it as null for
joint runs, or label the line as derived.

### 14. `dump_prompts.py` only works with joint variants, and silently batches gold answers

`bench/llm/dump_prompts.py:46` and `:73` call `v.build_all(...)`, which only the `Joint` family
has; `--variant` at `dump_prompts.py:109` has no `choices=`, so `--variant units` dies with
`AttributeError` and a typo dies with `KeyError`. `dump_prompts.py:78` builds the
unparseable-answer fallback as `[{...}] * len(rows)`, repeating one dict object ten times;
harmless today because `postprocess` does not mutate it, but it is the classic aliasing trap.

The real issue is the leak in findings 37. The tool is correct per file (each prompt holds out
its own conversation), but nothing prevents one agent from reading several prompt files in one
context, and conversation A's demo block carries A's gold answers, which contaminates B's
answer when both sit in the same batch. 20 of 39 conversations were exposed. The remedy in
entry 37 ("use one agent per conversation") lives only in prose.

**Fix.** Make the tool enforce it. `dump` writes a `batches.json` recording, for every prompt
file, which other conversations appear inside it as demos, and `score` refuses (or flags) any
answer file whose conversation appears as a demo in another file the same agent answered.
Cheaper still: add `--one-per-file`, writing a per-conversation subdirectory with an explicit
`README` saying "answer only this file in this context". Also add `choices=sorted(VARIANTS)`
and a `hasattr(v, 'build_all')` check with a clear message.

---

## D. Reproducibility for a teammate

### 15. None of today's three headline numbers can be reproduced from a clone

- **0.780 (cluster)**: `bench/results/` is gitignored. All 63 result files, including
  `qwen3.8-27b.units-fewshot.large-v3-turbo.json`, exist only on Elias's laptop and on
  blackhole. Every table in findings 30 to 37 is therefore unverifiable by Lucas, Nikolaj or
  Oscar, and `bench/results/probe/leaked.json`, cited by name in entry 37, is equally invisible.
- **0.724 (local)**: needs Ollama with `qwen3:4b` plus `transcripts/*.large-v3-turbo.json`,
  which are gitignored as well. The `transcripts/` directory is the input contract of the whole
  bench and it is not in the repo. `bench/hpc/sync.sh down` fetches it, which needs the VPN and
  the DTU account.
- **1.0 (validation)**: needs `bench/mine/current_answers.json` (gitignored), built by
  `answers_md.py`, which imports `request_dump/answers.jsonl` (gitignored, with 39 MB of dumped
  validation audio beside it). So the table cannot be rebuilt from the committed
  `agent_answers.md` plus `span_state.json` alone, even though those are the real sources.
  `val_diag.py:43`, `timeline.py:84` and `clips.py:74` share the dependency.

**Fix, in order of value per minute.** (a) Commit the `config` and `summary` blocks of every
result with the per-question records stripped: about 1 KB each, 63 files, and it makes the
findings log auditable. A five-line script plus a `bench/results/INDEX.md`. (b) Commit a
`request_dump/manifest.json` holding only the 19 filenames and their ten question texts (no
audio, no answers) and have `answers_md.py` prefer it, so the table can be rebuilt from the
repo. (c) State in `bench/README.md` that `transcripts/` must be fetched with
`bench/hpc/sync.sh down` or rebuilt with `transcribe_cache.py`, and which tags the bench needs.

### 16. `model.py`'s defaults are not the served configuration

`model.py:66` defaults `ASR_MODEL` to `large-v3` while findings 19 and 20 establish
`large-v3-turbo` as the served ASR (worth 0.051 on validation). `model.py:70` defaults
`LLM_MODEL` to `qwen3.5:4b`, a tag that appears nowhere else in the repo: every bench run and
every validation run used `qwen3:4b`. The module docstring at `model.py:28-29` says `ASR_CLEAN`
defaults to 1 while `model.py:88` defaults it to 0 (the code is right, per commit 3907d31 and
the comment above it; the docstring was never updated).

**Consequence.** A teammate running `python api.py` serves an unbenchmarked LLM tag on the
worse ASR, and either Ollama 404s on the tag or quietly pulls a different model. Nothing in the
repo records the environment of any validation run.

**Fix.** Either change the defaults to the served values or, better, commit a `serve.sh` (five
lines, see the how-to-run section) and make `warm_up()` log every resolved setting at INFO on
one line, so a run's configuration is in its own log. Fix the `ASR_CLEAN` docstring line.

### 17. The pod's serving scripts exist only on a stopped pod's volume

`bench/hpc/RUNPOD.md:9` lists `/workspace/serve27b.sh`, `/workspace/bench27b.sh` and
`/workspace/run_benches.sh` as living on the network volume, and `RUNPOD.md:11` notes the
container disk is erased on every stop. None of the three is in the repo.

**Consequence.** The recipe for serving the 27B (port 8000, thinking off, text only,
`--max-num-seqs 64`, 98k context) is a single point of failure on an $8/month volume, sixteen
hours before the deadline, on an account whose owner is not on this team. If the volume is lost
or the balance runs out, the knowledge goes with it.

**Fix.** Copy the three scripts into `bench/hpc/runpod/` tonight and have `RUNPOD.md` point at
the repo copies. They are also the only concrete example of serving the winning model, so they
belong next to `serve_vllm.sh` anyway.

### 18. The handoff tells the next person to run a flag that does not exist

`bench/hpc/HANDOFF.md:75-76`: "Summarise with `python bench/llm/bench.py --summary` (see
`bench/README.md`)". `bench.py` has no `--summary` argument and `--asr` is `required=True`, so
the command fails with an argparse error, and `bench/README.md` never mentions `--summary`
either. `bench/README.md:19-44` presents a layout table as "the contract" while omitting
`dump_prompts.py`, `qa_locate.py`, `threshold_sweep.py`, `RUNPOD.md`, `env.sh`, and the whole
of `bench/mine/`, `bench/ref/`, `timeline.py`, `clips.py` and `portal_status.py`.

**Fix.** Add the `--summary` mode (twenty lines over `bench/results/llm/*.json`, and it would
also serve finding 15's index) or change the line to the ranking snippet the RunPod runbook
refers to. Refresh the layout table: it is the first thing a new engineer reads.

### 19. The documented LSF override mechanism is the one that failed today

`bench/hpc/llm_bench.lsf:20-23` documents `MODELS="..." bsub < bench/hpc/llm_bench.lsf`, and
`HANDOFF.md:49-50` and `:60-61` repeat it as the resubmission recipe. Today's incident list
records that LSF ignored environment variables passed this way. The script's defaults at
`llm_bench.lsf:71-73` then apply instead, so a "resubmit only the failed models" command
quietly runs all eleven models across both ASR tags.

Related: `BENCH_FOR` (`llm_bench.lsf:95-100`) keys the 1500-token budget on the model id, but
the thing that needs the budget is the *variant*. `Qwen3.5-27B` and the two A3B models are not
in the map, so `VARIANTS="units-joint-demo"` on any of them repeats the 200-token truncation
that cost today's joint runs.

**Fix.** Two changes. Accept overrides through a file the job reads (`bench/hpc/job.env`,
sourced when present), or switch the documented command to `bsub -env "all, MODELS=..."` and
verify it once with a `LIMIT=1` smoke job before trusting it. And add a variant-keyed budget,
`case "$V" in *joint*) BENCH_ARGS+=(--max-tokens 1500);; esac`, taking the maximum of the model
and variant requirements.

### 20. Shared variant instances are single-conversation state, safe only by accident

`bench/llm/prompts.py:596-607` stores one instance per variant name; `bench.py:544-545` mutates
it with `set_conversation` before each conversation, and `bench.py:550-552` then runs ten worker
threads that call it.

**What is wrong.** Correct today only because conversations are strictly sequential. Two latent
hazards. The leave-one-out exclusion lives in `self.exclude`, so any future parallelism over
conversations leaks gold answers into the prompt of the conversation under test, which is
exactly the failure that already happened once today in `dump_prompts.py`. And `FewShot.pool` /
`JointDemo.pool` cache with a two-step assignment (`prompts.py:401` and `:542`): on the first
conversation ten threads can each build the entire pool (39 transcript reads apiece), and a
reader can observe `_pool` set while `_pool_asr` is stale.

**Fix.** Pass the exclusion in rather than storing it: either `build(question, units,
exclude=stem)` or a `for_conversation(stem, asr)` factory returning a fresh, immutable builder.
Guard `pool()` with a `threading.Lock` and assign the cache as one tuple. Then assert in the
test suite that no example line or demo turn mentions the conversation under test (test 1).

---

## E. Repository hygiene

**Correctly excluded** (verified against `.gitignore` and `git ls-files`): the portal and RunPod
keys and the control token (root `.gitignore`; `portal_status.py:22` reads the key from a
gitignored path rather than from the repo), `request_dump/` with its 39 MB of dumped validation
audio, `bench/results/`, the miner state, `bench/ref/clips.json` (which would embed validation
audio as base64 data URIs through `clips.py:74`), `bench/ref/timeline.html`, and the local
venvs. I grepped every tracked file for the usual key shapes and found nothing.

**Committed, and worth a deliberate decision.**

- `bench/mine/span_state.json`, `bench/mine/agent_labels/*.json` and
  `bench/mine/agent_answers.md` hold the validation set's recovered gold labels, obtained from
  the scoring endpoint over 358 probe runs. The rules and reputation lens belongs to another
  reviewer; from a code-hygiene view the problems are that (a) nothing beside the files states
  their provenance, so a new engineer could mistake them for organiser-supplied ground truth,
  and (b) `agent_labels/` is a *build product* of `answers_md.py` yet it is tracked, so running
  the tool with `--flip` or `--gold` produces a diff across twenty files. Recommendation: keep
  `agent_answers.md` and `span_state.json` as the two tracked sources, add a two-line
  `bench/mine/README.md` stating where they came from and that the pipeline is never tuned on
  them, and gitignore `agent_labels/` as generated.
- `data/audio/*.mp3` and `data/question_train.csv` arrived with the organisers' own
  `d7764e0 Import Nordic AI Cup 2026 starter kit` commit, so they are the provided layout and
  not something the team added. Leave them.
- `research/leaderboard/validation.2026-09-17T19-59.json` holds other teams' scores. Harmless
  in a private repo, and entry 31 cites it, so it earns its place.
- `bench/results/llm/qwen3.8-27b.units-joint-demo.large-v3-turbo.partial.json` sits next to its
  finished `.json` (gitignored, so laptop hygiene only). `bench.py:451-452` deletes the
  checkpoint when the final lands, so this one came down through `sync.sh down`, which never
  deletes. Worth a line in `sync.sh`: after `down`, remove any `*.partial.json` whose final
  exists, otherwise a stale partial with a *higher* score (this one records 0.7954) will
  eventually be quoted.

**Untracked and un-ignored, which is the worst of both.** `bench/mine/val_labels/` holds one
file, `conversation_sample_3.json`, with all ten questions answered `false` and null spans: an
abandoned pass on the oracle `/label` page. `oracle_server.py:78-84` displays it beside the
verified agent labels as if it were a label set, and it contradicts them. It will also be swept
into the next `git add -A`. Either finish it, delete it, or add it to `.gitignore`; my vote is
delete, since `agent_answers.md` is the source of truth and entry 27 verified it at 190/190.

**Missing.** A `serve.sh` or `.env.example` recording the served configuration (finding 16); the
three RunPod scripts (finding 17); the result summaries and `leaked.json` (finding 15); a
`request_dump/manifest.json` so the mined table can be rebuilt without the audio; pinned
versions for the bench side (`requirements.txt` covers only the FastAPI server, and the Ollama,
faster-whisper and vLLM versions exist only in prose); and any automated test at all, see
below. One operational note: `example.py:22-39` and `:65-80` write the incoming audio and our
answers to disk whenever `REQUEST_DUMP_DIR` is set. It is off by default and wrapped so it
cannot raise, but make sure it is unset for the single evaluation attempt: it adds a
multi-megabyte write inside a 60 s budget for no benefit.

Minor, local-only: `oracle_server.py:172-176` and `:113-118` build a path from the request with
`AUDIO / name` and only check `name.endswith('.mp3')` and `f.exists()`, so `../` in the path
escapes the directory. The server binds `127.0.0.1` so this is a note, not a risk; one
`Path.resolve().is_relative_to(AUDIO)` check closes it.

### The guardrail hook blocks prose

`.claude/hooks/block-evaluation.py:28-32` matches its patterns against the whole serialised tool
input, so any payload that *mentions* the endpoint is denied. `.claude/eval-block.log` shows it
blocking a `cat >> research/07-findings-log.md` heredoc that was documenting the leaderboard,
and blocking a sibling committee reviewer's prompt. The guardrail is the right call and its
nine test payloads (`.claude/hooks/test-payloads.jsonl`) are the only tests in the project, so
this is a refinement rather than a complaint: match only in executable positions, for example
require the URL to sit adjacent to `curl`, `wget`, `Invoke-RestMethod`, `requests.`, `fetch(`
or a `WebFetch` `url` field, and skip `Bash` payloads whose command is a heredoc write or an
append to a `.md` file. Keep the deny-by-default posture and the log.

---

## How to run: the minimum a teammate needs

Put this in `bench/README.md` or a top-level `RUNBOOK.md`. Everything below is read-only or
local except where marked.

**0. What you need, and where it is.** The repo alone is not enough. You additionally need
`transcripts/*.large-v3-turbo.json` (39 files, gitignored; fetch with
`bash bench/hpc/sync.sh down` on the DTU VPN, or rebuild with `python transcribe_cache.py`),
and for anything validation-related `request_dump/` (gitignored, only on Elias's laptop).

**1. Serve the endpoint (the honest pipeline, about 0.68 on validation).** The defaults in
`model.py` are not the served configuration, so always set these:

```bash
cd medical-appointment
ASR_MODEL=large-v3-turbo \
ASR_DEVICE=cuda \
LLM_URL=http://localhost:11434 \
LLM_MODEL=qwen3:4b \
START_RULE=first-word-end \
SPAN_ON_NO=1 \
python api.py                      # listens on 0.0.0.0:9054, submit <host>:9054/predict
```

Ollama must already hold `qwen3:4b` (`ollama pull qwen3:4b`) and be running. `model.warm_up()`
runs at import, so the process takes a minute to become ready and the first real request is not
the slow one.

**2. Re-score every existing bench result under the correct edge rule (no GPU, no network).**
Do this first: it is finding 1 and it is worth about +0.013.

```bash
cd medical-appointment
ASR_MODEL=large-v3-turbo python - <<'PY'
import glob, json, statistics, sys
sys.path.insert(0, '.')
import model
from utils import temporal_iou
for f in sorted(glob.glob('bench/results/llm/*.large-v3-turbo.json')):
    d = json.load(open(f, encoding='utf-8'))
    if not d.get('questions'):
        continue
    cache, tious, correct = {}, [], 0
    for r in d['questions']:
        tid = r['transcript_id']
        if tid not in cache:
            t = json.load(open(f'transcripts/conversation_{tid}.large-v3-turbo.json', encoding='utf-8'))
            ws = []
            for s in t['segments']:
                for w in s.get('words', []):
                    ws.append(model.Word(w['w'], float(w['start']), float(w['end'])))
                if s.get('words'):
                    ws[-1].w += '\x00'
            cache[tid] = (model.make_units(ws), float(t['duration']))
        units, dur = cache[tid]
        correct += r['prediction'] == r['label']
        if not r['gold']:
            continue
        if r['prediction'] != 1:                 # nulls-on-no policy, the live scorer's
            tious.append(0.0)
            continue
        span = model.span_from_ids(model.anchor_ids(r.get('raw') or {}, units), units, dur)
        tious.append(temporal_iou(tuple(r['gold']), span) if span else 0.0)
    acc, m = correct / len(d['questions']), statistics.mean(tious)
    logged = (d['summary'].get('nulls_on_no') or {}).get('score', 0)
    print(f"{f.split('/')[-1]:<56} logged {logged:.4f} -> "
          f"turbo rule acc {acc:.4f} tIoU {m:.4f} score {0.4 * acc + 0.6 * m:.4f}")
PY
```

Run the same block with `ASR_MODEL=large-v3` as a control: it reproduces the logged scores to
four decimals, which is what makes the corrected numbers trustworthy.

**3. Reproduce the local 0.724 (qwen3:4b, `units-joint-demo`, turbo transcripts).** The logged
run used large-v3's offsets, so leave `ASR_MODEL` unset to match it exactly, and set it to
`large-v3-turbo` to get the corrected number (expect roughly +0.01):

```bash
python bench/llm/bench.py --asr large-v3-turbo --variant units-joint-demo \
    --url http://localhost:11434/v1 --model qwen3:4b --workers 1 --max-tokens 1200
# writes bench/results/llm/qwen3-4b.units-joint-demo.large-v3-turbo.json
# read summary.nulls_on_no.score, not summary.score
```

`--max-tokens 1200` is not optional for a joint variant: the default 200 truncates the
ten-answer JSON and, per finding 2, the truncation warning will not fire.

Inspect a prompt with no server at all:
`python bench/llm/bench.py --asr large-v3-turbo --variant units-fewshot --print-prompt`.

**4. Reproduce the cluster 0.780 (corrected 0.793). Needs the DTU VPN and an H100.**

```bash
ssh dtu
cd /dtu/blackhole/1e/205502/nordic/medical-appointment
# submit-time env vars did not reach the job today: edit MODELS/ASR_TAGS/VARIANTS at
# llm_bench.lsf:71-73 instead, or use  bsub -env "all, MODELS=..." < bench/hpc/llm_bench.lsf
bsub < bench/hpc/llm_bench.lsf
bjobs -w
tail -f bench/results/logs/llm_bench.<JOBID>.log
```

For one model and one variant set `MODELS="Qwen/Qwen3.8-27B"`, `ASR_TAGS="large-v3-turbo"`,
`VARIANTS="units-fewshot"`, and `FORCE=1` to redo an existing result. Pull results back with
`bash bench/hpc/sync.sh down` from the laptop. The DTU service window starts Friday 20:00, so
the cluster is unavailable on the final day.

For the RunPod alternative follow `bench/hpc/RUNPOD.md` exactly, noting that its three shell
scripts live on the pod volume and not in the repo (finding 17).

**5. Reproduce the validation 1.0.** This is a table lookup, not the pipeline, and it needs
`request_dump/`, which is not in the repo.

```bash
python bench/mine/answers_md.py --spans --gold   # builds bench/mine/current_answers.json
python bench/mine/probe_server.py                # port 9055, answers from the table
# expose it (cloudflared or ngrok), then a HUMAN queues exactly one validation attempt:
python bench/portal_status.py --queue https://<tunnel>/predict
python bench/portal_status.py                    # read the score back
```

Never point the evaluation endpoint at this. `bench/mine/count_run.py` and
`bench/mine/span_probe.py` also queue real validation runs and should not be run casually.

**6. Diagnose without touching the portal.**

```bash
python bench/mine/val_diag.py --list             # served run vs hand binaries and recovered spans
python bench/timeline.py                         # writes bench/ref/timeline.html
python bench/ref/oracle_server.py                # http://localhost:9060/  (and /label)
```

---

## Five cheap tests that would have caught today's incidents

No test framework exists yet. `pytest` plus these five files is under 200 lines, runs in a few
seconds on CPU with no server and no network, and covers every incident class from today. Put
them in `tests/` with a `conftest.py` that sets `ASR_MODEL=large-v3-turbo` before importing
`model`.

1. **`test_prompts_build.py`: every variant builds, and no variant leaks the conversation under
   test.** For each name in `prompts.VARIANTS` and two real conversations: call
   `set_conversation(stem, asr)` where present, build the prompt (`build_all` for joint
   variants), and assert (a) `system`, `user` and `schema` are non-empty and the schema is
   valid JSON, (b) the prompt text contains no question text and no gold span value belonging
   to the conversation under test, (c) no demo turn's stem equals the held-out stem, and (d)
   the `units` variant's user message equals the string `model.ask_llm` would build, character
   for character. Catches: the `dump_prompts.py` gold leak, the shared-state hazard of finding
   20, the `JOINT_SYSTEM` string surgery at `prompts.py:434-436` silently producing a malformed
   schema description if `model.SYSTEM` is ever edited, and any future drift between
   `prompts.units` and the serving prompt.

2. **`test_joint_split.py`: the joint answer parser.** Parametrise over a correct 1..n reply, a
   zero-based reply, a reply with duplicate `q`, a reply with a missing entry, a reply with a
   non-dict item, `{'answers': []}`, and a reply truncated mid-JSON fed through
   `bench.parse_json`. Assert every case either maps each answer to the right question or
   raises, and that no case returns a silent all-no default. Catches: finding 3 in all three of
   its forms, and the 200-token truncation incident, since the truncated fixture must raise
   rather than score.

3. **`test_spans.py`: `anchor_ids` and `span_from_ids` edge cases.** On a synthetic five-unit
   transcript, parametrise `segments` over `[]`, `[99]`, `[-1, 0]`, `[True, False]`, `['2']`,
   `[3, 0]`, `[0, 7, 8]`, `[2.9]` and out-of-range floats, each with and without a matching
   `quote`. Assert: the span is `None` or lies entirely within `[0, duration]`, `end > start`,
   booleans are never treated as ids, a numeric string is accepted or rejected consistently
   with `prompts._int_list`, and for `[0, 7, 8]` with a quote matching unit 7 the span covers
   units 7 to 8 rather than unit 0. Add one `_words_post` case where an exact verbatim match
   exists outside the cited unit and assert it wins. Catches: findings 4, 5 and 6.

4. **`test_transcripts.py`: the transcript contract and unit coverage.** For every
   `transcripts/*.json`: assert the top-level keys of `bench/README.md:54-65` exist, that
   `duration >= last word end` (the property `span_probe.py` and `timeline.py` currently
   violate by using the last segment end, finding 10), that word times are non-decreasing with
   `end >= start`, that at least 80% of words carry a leading space (finding 12), and that
   `make_units` assigns every non-empty word to exactly one unit with strictly increasing
   starts. Then assert `prompts._unit_ranges` covers every unit index that has words. Catches:
   a new or re-timed ASR runner breaking the contract, the glued-transcript failure, and the
   duration bug.

5. **`test_config_parity.py`: the bench cannot disagree with the serving config.** Assert
   `model._FITTED` has an entry for every ASR tag present in `transcripts/` and for the served
   tag; assert that the bench's effective offsets for a tag equal `model._FITTED[tag]`; and,
   over every file in `bench/results/llm/`, assert `config.start_offset` and `config.end_offset`
   match `model._FITTED[config.asr]`, with an explicit allowlist for files written before the
   fix. Add two cheap assertions in the same file: every variant's estimated prompt length
   (characters divided by four) stays under `LLM_NUM_CTX`, and `max_tokens` for a joint variant
   is at least 1,200. Catches: finding 1 (the most expensive bug found today), finding 16's
   default drift, and the Ollama `num_ctx` ceiling of divergence 5.

Two one-line guards that are not worth a test but should go in tonight: assert the queued
attempt uuid matches the attempt being read in `count_run.py` and `span_probe.one_run`
(finding 8), and assert the null-span baseline score is 0.4000 before trusting any recovered
IoU (finding 9).
