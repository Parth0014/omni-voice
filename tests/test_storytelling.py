from copy import deepcopy

import pytest

from storytelling import build_story_plan


def _blocks():
    return [
        {"type": "heading", "text": "The gift"},
        {"type": "paragraph", "text": "Ada watched the rain. It felt like a small gift."},
        {"type": "quote", "speaker": "Ada", "text": "I am thankful for today."},
        {"type": "list", "text": "The rain. A familiar voice."},
        {"type": "quote", "speaker": None, "text": "We have enough."},
        {"type": "paragraph", "text": "She remembered that kindness."},
    ]


def test_preserve_keeps_every_block_in_order_and_does_not_rewrite_content():
    blocks = _blocks()
    original = deepcopy(blocks)
    plan = build_story_plan(blocks)

    assert [chunk["block_index"] for chunk in plan] == list(range(6))
    assert [chunk["text"] for chunk in plan] == [
        "The gift",
        "Ada watched the rain. It felt like a small gift.",
        'Ada says, "I am thankful for today."',
        "The rain. A familiar voice.",
        "We have enough.",
        "She remembered that kindness.",
    ]
    assert all(chunk["original_text"] == chunk["text"] for chunk in plan)
    assert all(chunk["role"] == "narration" for chunk in plan)
    assert all(chunk["paragraph_end"] for chunk in plan)
    assert blocks == original


def test_exclude_omits_only_quotes_and_keeps_source_indexes():
    plan = build_story_plan(_blocks(), quote_mode="exclude")
    assert [chunk["block_index"] for chunk in plan] == [0, 1, 3, 5]
    assert build_story_plan([{"type": "quote", "text": "A quote."}], quote_mode="exclude") == []


def test_two_voice_routes_attribution_to_narrator_and_only_words_to_quote_voice():
    blocks = [
        {"type": "quote", "speaker": "Ada", "text": "I am grateful."},
        {"type": "quote", "speaker": "Grace", "text": "So am I."},
        {"type": "quote", "text": "An unnamed voice."},
    ]
    plan = build_story_plan(blocks, quote_mode="two_voice")
    assert [(chunk["text"], chunk["role"]) for chunk in plan] == [
        ("Ada says,", "narration"),
        ("I am grateful.", "quote"),
        ("Grace says,", "narration"),
        ("So am I.", "quote"),
        ("An unnamed voice.", "quote"),
    ]
    assert [chunk["paragraph_end"] for chunk in plan] == [False, True, False, True, True]
    assert plan[0]["speed_factor"] == 1.0
    assert plan[1]["speed_factor"] == 0.97


def test_blank_and_unsupported_blocks_stay_silent():
    assert build_story_plan([
        {"type": "paragraph", "text": " \n "},
        {"type": "heading", "text": None},
        {"type": "embed", "text": "Player controls"},
    ]) == []


def test_pronunciation_is_case_insensitive_whole_word_longest_first_and_nonrecursive():
    source = "Ada met Adaline in New York. ADA thanked Ann and Jo."
    plan = build_story_plan(
        [{"type": "paragraph", "text": source}],
        pronunciation={"Ada": "Ay-duh", "Ann": "Jo", "Jo": "Joe", "New": "new", "New York": "Noo York"},
    )
    assert plan[0]["original_text"] == source
    assert plan[0]["text"] == "Ay-duh met Adaline in Noo York. Ay-duh thanked Jo and Joe."


def test_pronunciation_applies_to_attributions_and_keeps_existing_phonemes_intact():
    plan = build_story_plan(
        [{"type": "quote", "text": "The bass sounded like [B EY1 S].", "speaker": "Ada"}],
        quote_mode="two_voice",
        pronunciation={"Ada": "Ay-duh", "bass": "[B EY1 S]", "B": "bee"},
    )
    assert plan[0]["text"] == "Ay-duh says,"
    assert plan[1]["text"] == "The [B EY1 S] sounded like [B EY1 S]."


@pytest.mark.parametrize("lexicon", [
    [], {"Ada": ""}, {"": "Ada"}, {"Ada": "[b EY1 S]"}, {"Ada": "[B EY S]"},
    {"Ada": "[B1 EY1 S]"}, {"Ada": "[B EY3 S]"}, {"Ada": "[B EY1 S"},
    {"Ada": "[whisper]"}, {"Ada": "[]"}, {"Ada": "<emphasis>Ada</emphasis>"},
    {"Ada": "Ay\nduh"}, {"Ada": "A", "ada": "B"}, {"[Ada]": "A"},
])
def test_invalid_pronunciations_fail_before_inference(lexicon):
    with pytest.raises(ValueError):
        build_story_plan(_blocks(), pronunciation=lexicon)


def test_paragraphs_keep_multiple_sentences_together_with_context():
    text = "She opened the letter. It was a small kindness. She read it twice. " * 5
    plan = build_story_plan([{"type": "paragraph", "text": text}])
    assert len(plan) == 1
    assert plan[0]["text"] == text.strip()


def test_oversized_sentence_splits_without_dropping_or_repeating_words():
    text = "She remembered " + ", ".join(["the gentle help of a friend"] * 30) + "."
    plan = build_story_plan([{"type": "paragraph", "text": text}], max_chars=180)
    assert len(plan) > 1
    assert " ".join(chunk["original_text"] for chunk in plan) == text
    assert all(len(chunk["text"]) <= 180 for chunk in plan)
    assert [chunk["paragraph_end"] for chunk in plan] == [False] * (len(plan) - 1) + [True]
    assert all(chunk["pause_after_ms"] == 120 for chunk in plan[:-1])


def test_decimal_title_and_phonemes_are_not_broken_at_boundaries():
    text = (
        "She carried the small gift through the quiet streets and thanked Dr. Ada "
        "for the 3.14 dollar note. The sound was [B EY1 S] and it stayed with her. "
    ) * 4
    plan = build_story_plan([{"type": "paragraph", "text": text}], max_chars=100)
    assert " ".join(chunk["text"] for chunk in plan) == text.strip()
    assert sum(chunk["text"].count("Dr. Ada") for chunk in plan) == 4
    assert sum(chunk["text"].count("3.14") for chunk in plan) == 4
    assert sum(chunk["text"].count("[B EY1 S]") for chunk in plan) == 4


def test_multiword_pronunciation_stays_atomic_when_splitting():
    source = ("A quiet evening brought them all together in New York City once again. " * 5).strip()
    plan = build_story_plan(
        [{"type": "paragraph", "text": source}],
        pronunciation={"New York City": "[N UW1 Y AO1 R K S IH1 T IY0]"},
        max_chars=100,
    )
    assert " ".join(chunk["original_text"] for chunk in plan) == source
    assert sum(chunk["text"].count("[N UW1 Y AO1 R K S IH1 T IY0]") for chunk in plan) == 5
    assert all(len(chunk["text"]) <= 100 for chunk in plan)


def test_oversized_indivisible_token_fails_clearly():
    with pytest.raises(ValueError, match="exceeds max_chars"):
        build_story_plan([{"type": "paragraph", "text": "x" * 90}], max_chars=80)
    with pytest.raises(ValueError, match="exceeds max_chars"):
        build_story_plan(
            [{"type": "paragraph", "text": "Ada"}],
            pronunciation={"Ada": "very " * 30}, max_chars=80,
        )


def test_directions_override_native_pace_and_only_final_block_pause():
    blocks = [{"type": "paragraph", "text": "The story continued with a quiet act of kindness. " * 8}]
    plan = build_story_plan(blocks, directions={"0": {"pace": 1.05, "pause_after_ms": 850}}, max_chars=100)
    assert all(chunk["speed_factor"] == 1.05 for chunk in plan)
    assert all(chunk["pause_after_ms"] == 120 for chunk in plan[:-1])
    assert plan[-1]["pause_after_ms"] == 850


@pytest.mark.parametrize("directions", [
    [], {True: {}}, {-1: {}}, {99: {}}, {"00": {}}, {"x": {}}, {0: None},
    {0: {"emotion": "warm"}}, {0: {"pace": 0.7}}, {0: {"pace": 1.3}},
    {0: {"pace": float("nan")}}, {0: {"pace": float("inf")}}, {0: {"pace": True}},
    {0: {"pause_after_ms": -1}}, {0: {"pause_after_ms": 2001}}, {0: {"pause_after_ms": 1.5}},
    {0: {"pause_after_ms": "500"}}, {0: {}, "0": {}},
])
def test_invalid_directions_fail_clearly(directions):
    with pytest.raises(ValueError):
        build_story_plan(_blocks(), directions=directions)


def test_natural_is_uniform_and_warm_heuristics_are_small_and_explicit():
    assert {chunk["speed_factor"] for chunk in build_story_plan(_blocks(), delivery="natural")} == {1.0}
    assert [chunk["speed_factor"] for chunk in build_story_plan(_blocks())] == [0.98, 1.0, 0.97, 1.0, 0.97, 0.97]


@pytest.mark.parametrize("options", [
    {"quote_mode": "two-voice"}, {"delivery": "whisper"}, {"max_chars": 79},
    {"max_chars": 2001}, {"max_chars": 100.0}, {"max_chars": True},
])
def test_invalid_planner_options_fail_clearly(options):
    with pytest.raises(ValueError):
        build_story_plan(_blocks(), **options)


def test_nonverbal_beats_are_explicit_and_belong_to_quote_speaker():
    blocks = [{"type": "quote", "speaker": "Ada", "text": "I am grateful."}]
    plan = build_story_plan(blocks, quote_mode="two_voice",
                            directions={"0": {"before": "sigh", "after": "laughter"}})
    assert plan[0]["text"] == "Ada says,"
    assert plan[1]["text"] == "[sigh] I am grateful. [laughter]"
    assert plan[1]["original_text"] == "I am grateful."


@pytest.mark.parametrize("tag", ["emphasis", "[sigh]", [], {}, None, 1])
def test_invalid_nonverbal_tags_fail_before_inference(tag):
    with pytest.raises(ValueError, match="non-verbal"):
        build_story_plan(_blocks(), directions={0: {"before": tag}})
