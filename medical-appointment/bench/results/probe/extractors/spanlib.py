"""Shared char-range -> time-range conversion, with the word-boundary guard.

The dataset builder (build_turbo.py) anchors a span on WORD ENDS:

    time_start = first_word['end'] + START_OFFSET      (-0.20)
    time_end   = last_word['end']  + END_OFFSET        (-0.02)

Because the start anchors on the END of the first word, a character range that
begins one or two characters too early picks up the whole PREVIOUS word and
anchors on ITS end -- a cliff, not a gradient (measured: mean tIoU 1.0000 at
1 char of slop, 0.7582 at 2 and at 3).  A model that emits real character
offsets will land off word boundaries routinely, so the range must be resolved
to whole words with a guard before conversion or the probe measures the cliff.

THE GUARD.  Take the words the range overlaps, then drop an EDGE word that the
range barely touches: a word is kept only if at least MIN_WORD_FRAC of its own
characters lie inside the predicted range.  Interior words are always fully
contained (frac 1.0), so the kept set stays contiguous.  The surviving words
are then expanded outward to their full extent -- "snap outward to whole word
boundaries".
"""

START_OFFSET = -0.20
END_OFFSET = -0.02
MIN_WORD_FRAC = 0.5


def tiou(a, b):
    """Temporal IoU, the case's own metric."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def snap_to_words(words, cs, ce):
    """Character range -> (first_word, last_word), snapped outward, guarded.

    Returns None only if the conversation has no words at all.
    """
    if not words:
        return None
    if ce <= cs:                      # degenerate range: give it one character
        ce = cs + 1

    cand = [w for w in words if w['char_end'] > cs and w['char_start'] < ce]

    if not cand:
        # The range fell entirely inside whitespace between two words.
        # Take the nearest word by character distance.
        def dist(w):
            if w['char_end'] <= cs:
                return cs - w['char_end']
            if w['char_start'] >= ce:
                return w['char_start'] - ce
            return 0
        return (min(words, key=dist),) * 2

    def frac(w):
        ov = min(w['char_end'], ce) - max(w['char_start'], cs)
        width = max(1, w['char_end'] - w['char_start'])
        return ov / width

    keep = [w for w in cand if frac(w) >= MIN_WORD_FRAC]
    if not keep:                      # range shorter than half of every word it
        keep = [max(cand, key=frac)]  # touches -- keep the best-covered one
    return keep[0], keep[-1]


def chars_to_time(words, cs, ce):
    """Full conversion. Returns (snap_cs, snap_ce, time_start, time_end)."""
    hit = snap_to_words(words, cs, ce)
    if hit is None:
        return None
    first, last = hit
    return (first['char_start'], last['char_end'],
            first['end'] + START_OFFSET, last['end'] + END_OFFSET)
