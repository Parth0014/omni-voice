from transcript_quality import check_transcript


def test_reference_leak_is_rejected_even_if_audio_is_valid():
    reference = "The most beautiful moments in life rarely ask for anything."
    report = check_transcript("She opened the door and thanked her sister.", reference, reference)
    assert report["status"] == "reject"
    assert report["reference_leak_detected"]


def test_target_that_intentionally_contains_reference_words_is_not_rejected():
    text = "The most beautiful moments in life rarely ask for anything."
    assert check_transcript(text, text, text)["status"] == "pass"


def test_missing_words_need_review_not_an_emotion_score():
    report = check_transcript("She opened the door and thanked her sister.", "She thanked her sister.", "A calm voice.")
    assert report["status"] == "review"
    assert "emotion" not in report


def test_case_punctuation_and_nonverbal_tags_are_not_word_errors():
    result = check_transcript("[sigh] Thank you, my friend!", "thank you my friend", "Reference.")
    assert result["word_error_rate"] == 0
