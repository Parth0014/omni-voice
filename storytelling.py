"""Plan faithful, paragraph-aware scripts for OmniVoice narration.

The warm preset makes small, explicit native-speed adjustments and adds
paragraph pauses. It cannot manufacture emotion or emphasis: those depend on
the reference performance and the model. Source wording is retained, apart
from a stable spoken quote attribution and user-provided pronunciation entries.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from numbers import Real

NONVERBAL_TAGS = frozenset({"laughter", "sigh", "confirmation-en", "question-en", "question-ah",
                            "question-oh", "question-ei", "question-yi", "surprise-ah", "surprise-oh",
                            "surprise-wa", "surprise-yo", "dissatisfaction-hnn"})

_BLOCK_TYPES = frozenset({"heading", "paragraph", "quote", "list"})
_QUOTE_MODES = frozenset({"preserve", "exclude", "two_voice"})
_VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
_CONSONANTS = frozenset("B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split())
_ABBREVIATIONS = frozenset(
    "mr mrs ms dr prof rev fr sr jr st vs etc e.g i.e no fig inc ltd dept approx "
    "jan feb mar apr jun jul aug sep sept oct nov dec".split()
)
_BRACKETED = r"\[[^\[\]]*\]"
_HONORIFIC = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|Rev|Fr|St)\.\s+\S+", re.IGNORECASE)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _plain_entry(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} cannot contain control characters")
    if "<" in value or ">" in value:
        raise ValueError(f"{name} must be spoken text, not markup")
    return _clean(value)


class _Pronunciation:
    """Replace a lexicon once, retaining source spans for safe chunk boundaries."""

    def __init__(self, lexicon: Mapping | None) -> None:
        if lexicon is not None and not isinstance(lexicon, Mapping):
            raise ValueError("pronunciation must be a mapping of words to spoken forms")
        entries: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw_key, raw_value in (lexicon or {}).items():
            key = _plain_entry(raw_key, name="Pronunciation key")
            value = _plain_entry(raw_value, name=f"Pronunciation for {key!r}")
            if "[" in key or "]" in key:
                raise ValueError("Pronunciation keys cannot contain brackets")
            if not any(char.isalnum() for char in key):
                raise ValueError("Pronunciation keys must contain a word")
            if key.casefold() in seen:
                raise ValueError(f"Duplicate pronunciation key: {key!r}")
            seen.add(key.casefold())
            if "[" in value or "]" in value:
                if not re.fullmatch(_BRACKETED, value):
                    raise ValueError(f"Pronunciation for {key!r} must be one complete [ARPAbet] sequence")
                phonemes = value[1:-1].split()
                if not phonemes or any(
                    token not in _CONSONANTS
                    and not (token[-1:] in {"0", "1", "2"} and token[:-1] in _VOWELS)
                    for token in phonemes
                ):
                    raise ValueError(
                        f"Invalid ARPAbet for {key!r}: use uppercase CMU phonemes "
                        "and vowel stress digits 0, 1, or 2"
                    )
                value = "[" + " ".join(phonemes) + "]"
            entries.append((key, value))
        entries.sort(key=lambda entry: len(entry[0]), reverse=True)
        self.values = {f"word{index}": value for index, (_, value) in enumerate(entries)}
        alternatives = [f"(?P<protected>{_BRACKETED})"]
        alternatives.extend(
            rf"(?P<word{index}>(?<!\w){re.escape(key)}(?!\w))"
            for index, (key, _) in enumerate(entries)
        )
        self.pattern = re.compile("|".join(alternatives), re.IGNORECASE)

    def replace(self, text: str) -> str:
        return self.pattern.sub(
            lambda match: self.values.get(match.lastgroup, match.group()), text
        )

    def protected_spans(self, text: str) -> list[tuple[int, int]]:
        return [match.span() for match in self.pattern.finditer(text)] + [
            match.span() for match in _HONORIFIC.finditer(text)
        ]


def _inside(index: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < index < end for start, end in spans)


def _sentence_spans(text: str, protected: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find punctuation boundaries without interpreting abbreviations as sentences."""
    ends = []
    for index, char in enumerate(text):
        if char not in ".!?" or _inside(index, protected):
            continue
        if char == ".":
            if index and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
                continue
            token_match = re.search(r"[\w.]+$", text[: index + 1])
            token = token_match.group() if token_match else ""
            if token[:-1].lower() in _ABBREVIATIONS:
                continue
            if re.fullmatch(r"(?:[A-Za-z]\.)+", token):
                continue
        end = index + 1
        while end < len(text) and text[end] in '.!?"\'\u201d\u2019)]}':
            end += 1
        if end < len(text) and not text[end].isspace():
            continue
        if not _inside(end, protected):
            ends.append(end)
    ends.append(len(text))
    spans = []
    start = 0
    for end in sorted(set(ends)):
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    return spans


def _split_long_span(
    text: str,
    start: int,
    end: int,
    pronunciation: _Pronunciation,
    protected: list[tuple[int, int]],
    max_chars: int,
) -> list[tuple[int, int]]:
    result = []
    while len(pronunciation.replace(text[start:end].strip())) > max_chars:
        candidates = []
        for match in re.finditer(r"\s+", text[start:end]):
            cut = start + match.start()
            if cut <= start or _inside(cut, protected):
                continue
            spoken_length = len(pronunciation.replace(text[start:cut].strip()))
            if spoken_length <= max_chars:
                candidates.append((cut, spoken_length))
        if not candidates:
            raise ValueError(
                f"A word or pronunciation entry exceeds max_chars={max_chars}; "
                "increase max_chars or shorten that pronunciation entry"
            )
        # A substantial clause keeps its context; tiny comma fragments do not.
        clauses = [
            cut for cut, length in candidates
            if length >= max_chars * 0.6 and text[:cut].rstrip().endswith((",", ";", ":", "\u2014"))
        ]
        cut = clauses[-1] if clauses else candidates[-1][0]
        result.append((start, cut))
        start = cut
        while start < end and text[start].isspace():
            start += 1
    if text[start:end].strip():
        result.append((start, end))
    return result


def _chunks(text: str, pronunciation: _Pronunciation, max_chars: int) -> list[tuple[str, str]]:
    protected = pronunciation.protected_spans(text)
    units = []
    for start, end in _sentence_spans(text, protected):
        units.extend(_split_long_span(text, start, end, pronunciation, protected, max_chars))
    packed: list[tuple[int, int]] = []
    for start, end in units:
        if packed and len(pronunciation.replace(text[packed[-1][0]:end].strip())) <= max_chars:
            packed[-1] = (packed[-1][0], end)
        else:
            packed.append((start, end))
    return [
        (text[start:end].strip(), pronunciation.replace(text[start:end].strip()))
        for start, end in packed
    ]


def _directions(directions: Mapping | None, block_count: int) -> dict[int, dict]:
    if directions is not None and not isinstance(directions, Mapping):
        raise ValueError("directions must map zero-based block indexes to pace/pause settings")
    result = {}
    for raw_index, settings in (directions or {}).items():
        if isinstance(raw_index, str) and re.fullmatch(r"0|[1-9][0-9]*", raw_index):
            index = int(raw_index)
        elif isinstance(raw_index, int) and not isinstance(raw_index, bool):
            index = raw_index
        else:
            raise ValueError(f"Invalid direction block index: {raw_index!r}")
        if not 0 <= index < block_count:
            raise ValueError(f"Direction block index {index} is outside the source blocks")
        if index in result:
            raise ValueError(f"Duplicate direction block index: {index}")
        if not isinstance(settings, Mapping):
            raise ValueError(f"Directions for block {index} must contain pace/pause settings")
        unknown = set(settings) - {"pace", "pause_after_ms", "before", "after"}
        if unknown:
            raise ValueError(f"Unknown direction keys for block {index}: {sorted(unknown, key=str)!r}")
        for key, low, high in (("pace", 0.8, 1.2), ("pause_after_ms", 0, 2000)):
            if key not in settings:
                continue
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{key} for block {index} must be a finite number")
            if not low <= value <= high:
                raise ValueError(f"{key} for block {index} must be between {low} and {high}")
            if key == "pause_after_ms" and value != int(value):
                raise ValueError(f"pause_after_ms for block {index} must be a whole number")
        for key in ("before", "after"):
            if key in settings and (not isinstance(settings[key], str) or settings[key] not in NONVERBAL_TAGS):
                raise ValueError(f"{key} must be one supported OmniVoice non-verbal tag name")
        result[index] = dict(settings)
    return result


def build_story_plan(
    blocks: list[dict],
    quote_mode: str = "preserve",
    pronunciation: dict | None = None,
    directions: dict | None = None,
    delivery: str = "warm",
    max_chars: int = 420,
) -> list[dict]:
    """Build ordered inference chunks without rewriting the story.

    ``preserve`` speaks quotes and attributions with the narration voice;
    ``exclude`` omits quotes; ``two_voice`` assigns only quote words to the
    quote voice, keeping the ``<speaker> says,`` attribution with the narrator.

    Pronunciations match whole words/phrases, longest first, case-insensitively
    in one pass. Values are exact spoken forms or complete CMU ARPAbet strings,
    for example ``{"bass": "[B EY1 S]"}``. ``original_text`` is the script
    before those replacements; ``text`` is the actual inference input.

    Directions use zero-based source block indexes (integer or JSON string).
    ``pace`` is native speed, 0.8--1.2. ``pause_after_ms`` is a whole number,
    0--2000, applied at the end of the block, including split blocks.
    Warm uses 1.0 speed normally, .98 for headings and .97 for quotes and the
    last narrative paragraph. Natural uses 1.0 everywhere unless directed.
    These are pacing heuristics, not model emotion or word-stress controls.

    Blocks stay separate. Sentences are packed toward ``max_chars``; oversized
    sentences split at substantial clauses or words. A single indivisible
    token longer than the limit raises an error. Empty/unsupported blocks are
    silent, matching extraction's supported narration types.
    """
    if quote_mode not in _QUOTE_MODES:
        raise ValueError("quote_mode must be preserve, exclude, or two_voice")
    if delivery not in {"warm", "natural"}:
        raise ValueError("delivery must be warm or natural")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 80 <= max_chars <= 2000:
        raise ValueError("max_chars must be an integer between 80 and 2000")
    if not isinstance(blocks, list) or any(not isinstance(block, dict) for block in blocks):
        raise ValueError("blocks must be a list of extracted block dictionaries")
    lexicon = _Pronunciation(pronunciation)
    block_directions = _directions(directions, len(blocks))
    active = [
        index for index, block in enumerate(blocks)
        if block.get("type") in _BLOCK_TYPES
        and isinstance(block.get("text"), str)
        and block["text"].strip()
        and not (block["type"] == "quote" and quote_mode == "exclude")
    ]
    plan = []
    for index in active:
        block = blocks[index]
        block_type = block["type"]
        source = _clean(block["text"])
        raw_speaker = block.get("speaker") if block_type == "quote" else None
        speaker = _clean(raw_speaker) if isinstance(raw_speaker, str) and raw_speaker.strip() else None
        is_last = index == active[-1]
        speed = 1.0
        if delivery == "warm":
            if block_type == "heading":
                speed = 0.98
            elif block_type == "quote" or (is_last and block_type == "paragraph"):
                speed = 0.97
        pause = {"heading": 500, "paragraph": 400, "quote": 500, "list": 350}[block_type]
        if is_last:
            pause = 650
        direction = block_directions.get(index, {})
        speed = float(direction.get("pace", speed))
        pause = int(direction.get("pause_after_ms", pause))

        sections = []
        if block_type == "quote" and quote_mode == "two_voice":
            if speaker:
                sections.append((f"{speaker} says,", "narration", True))
            sections.append((source, "quote", False))
        elif block_type == "quote" and speaker:
            sections.append((f'{speaker} says, "{source}"', "narration", False))
        else:
            sections.append((source, "narration", False))

        block_plan_start = len(plan)
        for section_text, role, attribution in sections:
            pieces = _chunks(section_text, lexicon, max_chars)
            for piece_index, (original, spoken) in enumerate(pieces):
                paragraph_end = not attribution and piece_index == len(pieces) - 1
                plan.append(
                    {
                        "original_text": original,
                        "text": spoken,
                        "role": role,
                        "block_type": block_type,
                        "block_index": index,
                        "speaker": speaker,
                        "paragraph_end": paragraph_end,
                        "pause_after_ms": pause if paragraph_end else 120,
                        "speed_factor": speed if not attribution or "pace" in direction else 1.0,
                    }
                )
        spoken_start = block_plan_start
        if quote_mode == "two_voice" and block_type == "quote" and speaker:
            spoken_start = next(i for i in range(block_plan_start, len(plan)) if plan[i]["role"] == "quote")
        if "before" in direction:
            plan[spoken_start]["text"] = f"[{direction['before']}] " + plan[spoken_start]["text"]
        if "after" in direction:
            plan[-1]["text"] += f" [{direction['after']}]"
    return plan
