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


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###

def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Answer every question about one conversation."""
    n = len(request.questions)
    try:
        audio_bytes = decode_audio(request.audio_base64)
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
