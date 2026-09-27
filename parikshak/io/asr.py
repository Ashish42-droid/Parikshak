"""Voice commands: a closed grammar, not open dictation.

Roughly twenty phrases, matched against a fixed vocabulary. Open dictation would
be worse in every way that matters here - it needs a bigger model, it is slower,
it is far likelier to hallucinate a command out of ambient noise, and it cannot
be certified because nobody can enumerate what it might do.

The grammar is closed for the same reason the predicate vocabulary is closed:
you can write down everything the system will accept, and prove it.

Crucially, an override does not erase anything. `override` silences an alert for
one step and is written into the hash-chained log with its timestamp. The crew
is always in command, and the record always says what they decided.

ASR is first on the descope ladder (PLAN.md section 16) - it falls back to a GUI
button and costs nothing on stage. So the grammar is kept independent of the
recogniser: the same commands work from a microphone, a click, or a test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Command(str, Enum):
    MARK_DONE = "MARK_DONE"
    OVERRIDE = "OVERRIDE"
    REPEAT = "REPEAT"
    NEXT_STEP = "NEXT_STEP"
    PREVIOUS_STEP = "PREVIOUS_STEP"
    ACKNOWLEDGE = "ACKNOWLEDGE"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    STATUS = "STATUS"
    SILENCE = "SILENCE"
    #: A procedure's own crew token - "label verified", "unit idle". Carried in
    #: `Recognised.token`, and fed to the engine as a frame confirmation.
    CONFIRM = "CONFIRM"


#: The closed grammar. Every phrase the system will act on, in one place.
#: Longer phrases are matched first so "mark step done" does not resolve as
#: "mark done" plus noise.
PHRASES: dict[str, Command] = {
    "mark done": Command.MARK_DONE,
    "mark step done": Command.MARK_DONE,
    "step complete": Command.MARK_DONE,
    "override": Command.OVERRIDE,
    "continue anyway": Command.OVERRIDE,
    "proceed anyway": Command.OVERRIDE,
    "repeat": Command.REPEAT,
    "say again": Command.REPEAT,
    "repeat that": Command.REPEAT,
    "next step": Command.NEXT_STEP,
    "what is next": Command.NEXT_STEP,
    "previous step": Command.PREVIOUS_STEP,
    "go back": Command.PREVIOUS_STEP,
    "acknowledged": Command.ACKNOWLEDGE,
    "acknowledge": Command.ACKNOWLEDGE,
    "pause procedure": Command.PAUSE,
    "hold procedure": Command.PAUSE,
    "resume procedure": Command.RESUME,
    "status": Command.STATUS,
    "where am i": Command.STATUS,
    "quiet": Command.SILENCE,
    "silence": Command.SILENCE,
}

#: Commands that change the flight record rather than just the display. These
#: are logged with their spoken text, and confirmed before acting when the
#: recogniser is not confident.
CONSEQUENTIAL = frozenset({Command.MARK_DONE, Command.OVERRIDE, Command.ACKNOWLEDGE,
                           Command.CONFIRM})

#: Letters and digits, matching pdl.validator.normalise_token exactly. If the
#: two disagreed, a token the validator accepts as distinct could collapse into
#: another one here, and the grammar would resolve it to the wrong step.
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Recognised:
    command: Command
    phrase: str
    confidence: float
    raw: str
    #: The procedure's token, spelled as the procedure declares it, for CONFIRM
    #: and for procedure-declared overrides. None for generic commands.
    token: str | None = None

    @property
    def needs_confirmation(self) -> bool:
        """A consequential command heard indistinctly is asked back, not obeyed.

        Mishearing "mark done" from a fan and a cough would write a false
        completion into a tamper-evident record, which is the one thing the
        record exists to prevent.
        """
        return self.command in CONSEQUENTIAL and self.confidence < 0.85


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_WORD.findall(text.lower()))


class CommandGrammar:
    """Matches an utterance against the closed vocabulary.

    Substring matching, longest phrase first, so a recogniser that returns
    "uh mark done please" still resolves - crew speech has filler and a grammar
    that demands exact strings fails on real people.
    """

    def __init__(self, phrases: dict[str, Command] | None = None, *,
                 min_confidence: float = 0.5, tokens: Iterable[str] = (),
                 override_tokens: Iterable[str] = ()) -> None:
        self.phrases = dict(phrases or PHRASES)
        self.min_confidence = min_confidence
        # (spoken form, command, token as declared). Procedure entries are listed
        # first so that on a tie in length the procedure's own meaning wins -
        # CSP-1 waits for "acknowledged" as a token, which is also a generic
        # command phrase, and inside that procedure the token is what was meant.
        entries: list[tuple[str, Command, str | None]] = []
        for tok in tokens:
            if normalise(tok):
                entries.append((normalise(tok), Command.CONFIRM, tok))
        for tok in override_tokens:
            if normalise(tok):
                entries.append((normalise(tok), Command.OVERRIDE, tok))
        seen = {e[0] for e in entries}
        entries += [(p, c, None) for p, c in self.phrases.items() if p not in seen]
        # Longest first across BOTH sets, so a generic "mark step done" is not
        # swallowed by a shorter procedure token that happens to be inside it.
        self._entries = sorted(entries, key=lambda e: len(e[0]), reverse=True)
        self._ordered = [e[0] for e in self._entries]

    @classmethod
    def for_procedure(cls, procedure, **kw) -> CommandGrammar:
        """The grammar a specific procedure needs: generic commands, plus its
        declared crew tokens, plus its declared override tokens."""
        return cls(tokens=getattr(procedure, "crew_tokens", ()),
                   override_tokens=procedure.alert_policy.override_tokens, **kw)

    @property
    def vocabulary(self) -> list[str]:
        """Every word the recogniser needs. Vosk takes this as its grammar, which
        is what keeps it small and stops it inventing words we never accept."""
        words: set[str] = set()
        for phrase in self._ordered:
            words.update(phrase.split())
        return sorted(words)

    def match(self, text: str, confidence: float = 1.0) -> Recognised | None:
        if confidence < self.min_confidence:
            return None
        cleaned = normalise(text)
        if not cleaned:
            return None
        for phrase, command, token in self._entries:
            if phrase in cleaned:
                return Recognised(command, phrase, confidence, text, token)
        return None

    def matches_override_token(self, text: str, tokens: Iterable[str]) -> str | None:
        """Resolve to one of the procedure's declared `crew_override.tokens`.

        The procedure author decides which words count as an override, so the
        grammar defers to the file rather than hard-coding them here.
        """
        cleaned = normalise(text)
        for token in tokens:
            if normalise(token) and normalise(token) in cleaned:
                return token
        return None


# --------------------------------------------------------------------------
class VoskRecogniser:
    """Vosk small-en, constrained to the grammar's vocabulary.

    Passing the word list restricts the decoder to phrases we accept, which
    makes it both faster and far less likely to produce a command out of rack
    noise. Lazily imported - ASR is descope item one and must never be a hard
    dependency.
    """

    def __init__(self, model_path: str, grammar: CommandGrammar | None = None,
                 sample_rate: int = 16000) -> None:
        try:
            import vosk
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "VoskRecogniser needs vosk; ASR is optional - fall back to the "
                "GUI button (PLAN.md descope ladder item 1)") from exc
        import json

        self.grammar = grammar or CommandGrammar()
        self._json = json
        self._model = vosk.Model(model_path)
        self._rec = vosk.KaldiRecognizer(
            self._model, sample_rate,
            json.dumps(self.grammar.vocabulary + ["[unk]"]))

    def feed(self, pcm: bytes) -> Recognised | None:  # pragma: no cover - needs vosk
        if not self._rec.AcceptWaveform(pcm):
            return None
        result = self._json.loads(self._rec.Result())
        text = result.get("text", "")
        words = result.get("result", [])
        confidence = min((w.get("conf", 1.0) for w in words), default=1.0)
        return self.grammar.match(text, confidence)
