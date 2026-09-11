"""ASR-based content checks; these heuristics do not score voice acting."""

import re
import unicodedata

VERSION = "transcript-v1"


def words(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("\u2019", "'")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return re.findall(r"[\w]+(?:'[\w]+)?", text)


def word_error_rate(expected, actual):
    left, right = words(expected), words(actual)
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        updated = [i]
        for j, b in enumerate(right, 1):
            updated.append(min(updated[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = updated
    return row[-1] / max(1, len(left))


def check_transcript(expected, actual, reference_text):
    target, observed, reference = words(expected), words(actual), words(reference_text)
    target_ngrams = {tuple(target[i:i + 6]) for i in range(max(0, len(target) - 5))}
    reference_ngrams = {tuple(reference[i:i + 6]) for i in range(max(0, len(reference) - 5))} - target_ngrams
    observed_ngrams = {tuple(observed[i:i + 6]) for i in range(max(0, len(observed) - 5))}
    leaked = bool(reference_ngrams & observed_ngrams)
    wer = word_error_rate(expected, actual)
    if not observed or leaked:
        status = "reject"
    elif wer > .20:
        status = "review"
    else:
        status = "pass"
    return {"version": VERSION, "status": status, "word_error_rate": wer,
            "reference_leak_detected": leaked, "expected_text": expected, "transcript": actual,
            "limitation": "ASR can mishear names, phonemes and non-verbal sounds; inspect flagged takes."}
