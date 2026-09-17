"""The endpoint's model: delegates to ``model.answer_all``.

Kept deliberately thin. Everything that can raise is inside ``model`` and is
caught there per question; this layer catches whatever is left so a request
always gets a well-formed reply. A guess is worth half a mark; an exception is
worth nothing.
"""

import logging

import model
from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import decode_audio

logger = logging.getLogger(__name__)

# Load the ASR and exercise the LLM once, at import. The first inference is the
# slowest and there is no grace period for it.
model.warm_up()


def _dump(audio_filename: str, audio_bytes: bytes, questions) -> None:
    """Record an incoming request on disk when REQUEST_DUMP_DIR is set (off by
    default). Never raises: a failed dump must not cost a conversation."""
    import json
    import os
    from pathlib import Path
    d = os.environ.get('REQUEST_DUMP_DIR')
    if not d:
        return
    try:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        stem = Path(audio_filename).stem or 'request'
        if not (p / f'{stem}.mp3').exists():
            (p / f'{stem}.mp3').write_bytes(audio_bytes)
        (p / f'{stem}.questions.json').write_text(json.dumps(list(questions), ensure_ascii=False, indent=1), encoding='utf-8')
    except Exception:
        logger.exception('request dump failed for %s', audio_filename)


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###

def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Answer every question about one conversation."""
    n = len(request.questions)
    try:
        audio_bytes = decode_audio(request.audio_base64)
        _dump(request.audio_filename, audio_bytes, request.questions)
        answers, spans = model.answer_all(audio_bytes, request.audio_filename, request.questions)
        if len(answers) != n or len(spans) != n:
            raise ValueError(f'model returned {len(answers)} answers for {n} questions')
    except Exception:
        logger.exception('%s: whole-conversation fallback', request.audio_filename)
        answers, spans = [True] * n, [None] * n

    return ASRQuestionResponseDto(
        answers=[bool(a) for a in answers],
        evidence_start=[s[0] if s is not None else None for s in spans],
        evidence_end=[s[1] if s is not None else None for s in spans],
    )
