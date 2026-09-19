"""Does the served 27B cite the sentence that states the answer but not the one that names the subject?

Elias's observation on the overlap page: spans like "it came back normal" carry the answer but not the
context the annotators marked (the utterance that says what "it" is). This script goes through every
annotated yes question of the training set and classifies the 27B's cited units against the units
the annotators marked, then measures what a prompt change that pulls in the preceding context would do.

For each of the 195 gold-yes questions:
  units      the clause-and units the served pipeline offered the model (rebuilt with model.make_units)
  chosen     the contiguous run the served pipeline built from the model's answer (model.anchor_ids: the
             quote's unit plus adjacent cited ids, then span_from_ids); its span is asserted equal to the served span
  annotated  the contiguous run of units with the best tIoU against the gold interval (the same oracle
             as bench/units/clause_oracle.py); this is what the model would have had to cite
  relation   same | short-front | short-back | short-both | long-front | long-back | long-both |
             shifted | disjoint, from the two index ranges
  flags      pronoun-led (the chosen text opens on it / that / this / they / ...);
             subject-missing (a content word of the question occurs in the units the annotators added
             in front and in none of the chosen units)

    python bench/context_pattern.py           # prints the summary, writes research/13-context-pattern.md
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
os.environ.setdefault('ASR_MODEL', 'large-v3-turbo')
os.environ.setdefault('UNIT_SPLIT', 'clause-and')
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(HERE / 'llm'))
sys.path.insert(0, str(HERE / 'units'))
import model  # noqa: E402
from model import make_units, span_from_ids  # noqa: E402
from clause_oracle import load_words  # noqa: E402

RESULT = HERE / 'results' / 'llm' / 'qwen3.8-27b.units-fewshot-both.large-v3-turbo.clause-and.p4.json'
OUT = CASE / 'research' / '13-context-pattern.md'
ACC = 389 / 390
MAX_RUN = 8
PRONOUNS = {'it', "it's", 'its', 'that', "that's", 'this', 'these', 'those', 'they', "they're", 'he', "he's", 'she', "she's",
            'there', "there's", 'which', 'so', 'and', 'but', 'then', 'yes', 'yeah', 'no', 'okay', 'ok', 'right', 'well', 'good'}
STOP = set('''a an the is are was were be been being am do does did has have had having will would shall should can could may might must
to of in on at for from by with about as into like through after before between under over during without within along across
and or but nor so yet if then than that this these those it its they them their there here he she his her him we our you your
i me my mine any some all both each every either neither not no yes patient patients patient's doctor doctor's visit
what which who whom whose when where why how also just only still already ever never again more most much many very
being been get got getting go going went come came take took taken make made say said tell told
conversation discussion mention mentioned mentions discussed discuss'''.split())


def content_words(text):
    out = set()
    for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()):
        if w in STOP or len(w) < 3:
            continue
        w = re.sub(r"'s$", '', w)
        for suf in ('ing', 'ed', 'es', 's'):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[:-len(suf)]
                break
        out.add(w)
    return out


def tiou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def chosen_run(ids, n):
    ids = sorted({i for i in ids if 0 <= i < n})
    run = [ids[0]]
    for i in ids[1:]:
        if i - run[-1] <= 2:
            run.append(i)
        else:
            break
    return run[0], run[-1]                      # span_from_ids spans first..last of the cited run


def best_run(units, duration, gold):
    best, best_ij = -1.0, None
    for i in range(len(units)):
        for j in range(i, min(len(units), i + MAX_RUN)):
            sp = span_from_ids(list(range(i, j + 1)), units, duration)
            t = tiou(gold, sp)
            if t > best + 1e-12:
                best, best_ij = t, (i, j)
    return best_ij, best


def relation(c, a):
    (c0, c1), (a0, a1) = c, a
    if (c0, c1) == (a0, a1):
        return 'same'
    if c1 < a0 or a1 < c0:
        return 'disjoint'
    if a0 <= c0 and c1 <= a1:
        return 'short-both' if (c0 > a0 and c1 < a1) else ('short-front' if c0 > a0 else 'short-back')
    if c0 <= a0 and a1 <= c1:
        return 'long-both' if (c0 < a0 and c1 > a1) else ('long-front' if c0 < a0 else 'long-back')
    return 'shifted'


def utext(units, i, j):
    return ' '.join(units[k].text.strip() for k in range(i, j + 1))


def main():
    d = json.loads(RESULT.read_text(encoding='utf-8'))
    assert d['config']['unit_split'] == 'clause-and' and d['config']['asr'] == 'large-v3-turbo'
    model.START_OFFSET, model.END_OFFSET = d['config']['start_offset'], d['config']['end_offset']
    rows = []
    convs = {}
    for q in d['questions']:
        if q['label'] != 1:
            continue
        tid = q['transcript_id']
        if tid not in convs:
            words, duration = load_words(CASE / 'transcripts' / f'conversation_{tid}.large-v3-turbo.json')
            convs[tid] = (make_units(words, 'clause-and'), duration)
        units, duration = convs[tid]
        c = chosen_run(model.anchor_ids(q['raw'], units), len(units))   # the served anchoring: quote's unit plus adjacent cited ids
        sp = span_from_ids(list(range(c[0], c[1] + 1)), units, duration)
        assert [round(x, 2) for x in sp] == [round(x, 2) for x in q['span']], (q['question_id'], sp, q['span'])
        a, a_t = best_run(units, duration, q['gold'])
        rel = relation(c, a)
        qw = content_words(q['question'])
        chosen_t = utext(units, *c)
        front_t = utext(units, a[0], c[0] - 1) if rel in ('short-front', 'short-both') else ''
        back_t = utext(units, c[1] + 1, a[1]) if rel in ('short-back', 'short-both') else ''
        xfront_t = utext(units, c[0], a[0] - 1) if rel in ('long-front', 'long-both') else ''
        xback_t = utext(units, a[1] + 1, c[1]) if rel in ('long-back', 'long-both') else ''
        first = re.sub(r'[^a-z\']', '', chosen_t.split()[0].lower()) if chosen_t.split() else ''
        subj_front = sorted((qw & content_words(front_t)) - content_words(chosen_t))
        subj_back = sorted((qw & content_words(back_t)) - content_words(chosen_t))
        # what one more unit in front would have done, for every question (the prompt-change simulation)
        ext = span_from_ids(list(range(max(0, c[0] - 1), c[1] + 1)), units, duration)
        rows.append({
            'id': q['question_id'], 'tid': tid, 'q': q['question'], 'tiou': q['tiou'], 'rel': rel,
            'chosen': c, 'annot': a, 'annot_tiou': a_t, 'chosen_text': chosen_t, 'annot_text': utext(units, *a),
            'front': front_t, 'back': back_t, 'xfront': xfront_t, 'xback': xback_t,
            'pronoun': first in PRONOUNS, 'first': first, 'subj_front': subj_front, 'subj_back': subj_back,
            'ext_tiou': tiou(q['gold'], ext), 'n_chosen': c[1] - c[0] + 1, 'n_annot': a[1] - a[0] + 1,
        })
    return rows


def summarise(rows):
    n = len(rows)
    L = []
    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs)
    base = mean(r['tiou'] for r in rows)
    L.append(f"{n} gold-yes questions; served mean tIoU {base:.4f}; the annotated units' own ceiling {mean(r['annot_tiou'] for r in rows):.4f}")
    cnt = Counter(r['rel'] for r in rows)
    L.append('relation of the cited run to the annotated run: ' + ', '.join(f'{k} {v}' for k, v in cnt.most_common()))
    for rel in ('short-front', 'short-both', 'short-back', 'long-front', 'long-both', 'long-back', 'shifted', 'disjoint', 'same'):
        rs = [r for r in rows if r['rel'] == rel]
        if not rs:
            continue
        L.append(f"  {rel:<12} n {len(rs):>3}  mean tIoU {mean(r['tiou'] for r in rs):.3f}  pronoun-led {sum(r['pronoun'] for r in rs):>3}"
                 f"  subject in the missing front {sum(bool(r['subj_front']) for r in rs):>3}  one more unit in front {mean(r['ext_tiou'] for r in rs):.3f}")
    sf = [r for r in rows if r['rel'] in ('short-front', 'short-both')]
    L.append(f"the pattern (cited run starts after the annotated run): {len(sf)} of {n}; pronoun-led {sum(r['pronoun'] for r in sf)},"
             f" subject named only in the missing front {sum(bool(r['subj_front']) for r in sf)}, both {sum(r['pronoun'] and bool(r['subj_front']) for r in sf)}")
    lf = [r for r in rows if r['rel'] in ('long-front', 'long-both')]
    L.append(f"the opposite (cited run starts before the annotated run): {len(lf)} of {n}")
    pro = [r for r in rows if r['pronoun']]
    L.append(f"pronoun-led citations overall: {len(pro)}; of these short-front {sum(r['rel'] in ('short-front','short-both') for r in pro)},"
             f" same {sum(r['rel']=='same' for r in pro)}, other {sum(r['rel'] not in ('short-front','short-both','same') for r in pro)}")
    # prompt-change simulations: always one more unit in front, or only when a flag fires
    def sim(name, pred):
        m = mean(r['ext_tiou'] if pred(r) else r['tiou'] for r in rows)
        k = sum(pred(r) for r in rows)
        wins = sum(pred(r) and r['ext_tiou'] > r['tiou'] + 1e-9 for r in rows)
        loses = sum(pred(r) and r['ext_tiou'] < r['tiou'] - 1e-9 for r in rows)
        L.append(f"  {name:<58} fires {k:>3}  wins {wins:>3}  loses {loses:>3}  mean tIoU {m:.4f} ({m-base:+.4f})  score {0.4*ACC+0.6*m:.4f}")
    L.append('adding the unit in front of the cited run:')
    sim('always', lambda r: True)
    sim('when the citation is pronoun-led', lambda r: r['pronoun'])
    sim('when the citation is one unit long', lambda r: r['n_chosen'] == 1)
    sim('when pronoun-led and one unit long', lambda r: r['pronoun'] and r['n_chosen'] == 1)
    sim('when a question content word is missing from the citation', lambda r: bool(content_words(r['q']) - content_words(r['chosen_text'])))
    sim('only when it helps (oracle upper bound)', lambda r: r['ext_tiou'] > r['tiou'])
    return '\n'.join(L)


READING = """## Reading, after going through all 72 differing rows by hand (2026-09-19)

Elias's pattern, the answer cited without the utterance that names its subject, is real and looks like
this in 7 questions: sample_48 q01 and q03 ("It came back normal." where the annotators start two units
earlier at "The first was your thyroid stimulating hormone"), sample_48 q05 ("Both normal." after "That
was normal as well."), sample_48 q02 ("Persistent fatigue." after the patient's confirmation), sample_4
q01 ("No changes. You carry on exactly as you are." after "So, what happens with my treatment?"),
sample_6 q06 and q04 (the penicillin dose after "For the sinuses, penicillin."), sample_18 q01 ("The
prescription is created." after "Yes."). Three of the 15 short-front rows are the same span in sample_48.
The other short-front rows are annotator generosity, a preceding line that adds nothing the question needs
(sample_57 q05, sample_64 q03, sample_66 q01, sample_75 q03, sample_33 q05), or a lead-in clause that
the clause cut separates ("From what you describe,", "For the pain,").

Its mirror image is more common, 18 rows: the model included the utterance that sets up the answer and
the annotators marked only the answer. "And what did it say? It was negative." (sample_43 q04) is scored
on "It was negative." alone; "So, this is your annual follow-up. It is, for the asthma." on the second
sentence alone (sample_4 q04); "Have you had any fever with it? No fever." on "No fever." (sample_86 q01);
likewise sample_75 q01, sample_10 q03, sample_23 q06, sample_82 q02, sample_50 q05. Four rows are the
diabetes summaries where the annotators took the sub-clause ("with no signs of complications") and the
model the whole sentence. The lead-in clauses go both ways: excluded by the annotators in sample_79 q02,
sample_90 q04 and sample_39 q01, included in sample_52 q03 and sample_5 q02.

The same holds at the back: 13 rows where the annotators added the confirmation that follows ("Yes.",
"I really have.", "Good to see you.", "Nothing abnormal to report.") and 9 where the model added it and
the annotators did not.

So the annotators are not consistent about context, and the served prompt already asks for "the one with
the detail plus a neighbour when the fact is spread over two (a question and its answer, a statement and
its number)". The model applies that instruction about as often as the annotators do, in both directions.
What is left is which side each annotator chose on each question, and that is not predictable from the
text: a rule that adds the preceding unit whenever the citation opens on a pronoun fires 53 times, wins 6
and loses 39 (mean tIoU 0.712 to 0.668, score 0.826 to 0.800). A perfect trigger would gain 0.021 tIoU on
17 questions, and research/10 already found that a learned classifier for the previous utterance captures
0.009 of that.

Verdict: do not prompt for more context; it would cost more than it gains. The 16 whole misses are a
different failure: the model cites a valid passage that states the fact and the annotators marked another
mention of it (sample_77, sample_57 q03, sample_33 q03), and two golds sit on the greeting (sample_63 q02,
sample_64 q02, annotation defects).
"""


def table(rows):
    def esc(s):
        return s.replace('|', '\\|').replace('\n', ' ')
    L = ['| question | tIoU | relation | cited units | annotated units | flags |', '|---|---:|---|---|---|---|']
    order = {'short-front': 0, 'short-both': 1, 'short-back': 2, 'long-front': 3, 'long-both': 4, 'long-back': 5, 'shifted': 6, 'disjoint': 7, 'same': 8}
    for r in sorted(rows, key=lambda r: (order[r['rel']], r['tiou'])):
        cited = esc(r['chosen_text'])
        ann = esc(r['annot_text'])
        if r['front']:
            ann = f"**{esc(r['front'])}** " + esc(r['chosen_text']) + (f" **{esc(r['back'])}**" if r['back'] else '')
        elif r['back']:
            ann = esc(r['chosen_text']) + f" **{esc(r['back'])}**"
        if r['xfront'] or r['xback']:
            cited = (f"~~{esc(r['xfront'])}~~ " if r['xfront'] else '') + esc(r['annot_text']) + (f" ~~{esc(r['xback'])}~~" if r['xback'] else '')
        flags = []
        if r['pronoun']:
            flags.append(f"opens on \"{r['first']}\"")
        if r['subj_front']:
            flags.append('subject only in front: ' + ', '.join(r['subj_front']))
        if r['subj_back']:
            flags.append('subject only behind: ' + ', '.join(r['subj_back']))
        if r['ext_tiou'] > r['tiou'] + 1e-9:
            flags.append(f"one more unit in front {r['ext_tiou']:.2f}")
        L.append(f"| {r['id']}<br>{esc(r['q'])} | {r['tiou']:.3f} | {r['rel']} | {cited} | {ann} | {'; '.join(flags)} |")
    return '\n'.join(L)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    rows = main()
    s = summarise(rows)
    print(s)
    doc = ['# Does the 27B cite the answer without its subject? Every gold-yes question of the training set',
           '',
           'Generated by `bench/context_pattern.py` from the served run '
           '(`bench/results/llm/qwen3.8-27b.units-fewshot-both.large-v3-turbo.clause-and.p4.json`), the turbo transcripts and '
           '`data/question_train.csv`. "Annotated units" is the contiguous run of served clause units with the best tIoU against '
           'the gold interval, so the comparison is between unit ranges the model could have cited. In the annotated column, '
           'bold marks units the annotators included and the model left out; in the cited column, strikethrough marks units '
           'the model included and the annotators did not.',
           '', '## Summary', '', '```', s, '```', '', READING, '## Every question, worst first within each relation', '', table(rows), '']
    OUT.write_text('\n'.join(doc), encoding='utf-8')
    print(f'wrote {OUT}')
