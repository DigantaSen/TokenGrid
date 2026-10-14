from app.tasks import Task, extract_answer, grade, normalize_number


def num(gold):
    return Task("t", "gsm8k", "q", gold, "numeric")


def mc(gold):
    return Task("t", "mmlu", "q", gold, "mc")


def test_normalize_number():
    assert normalize_number(" 1,234 ") == "1234"
    assert normalize_number("$18.00.") == "18"
    assert normalize_number("2.5") == "2.5"


def test_numeric_takes_last_answer_line():
    assert extract_answer("Answer: 12\nrecheck\nAnswer: 18\ntook 3 steps", "numeric") == "18"
    assert grade(num("1200"), "so the total is 1,200 dollars")[1]
    assert not grade(num("18"), "Answer: 181")[1]


def test_mc_variants():
    for text in ["Answer: C", "answer: (c)", "Answer: **C**", "Answer: B\nAnswer: C"]:
        assert extract_answer(text, "mc") == "C", text


def test_unparseable_counts_as_wrong():
    assert grade(mc("A"), "") == (None, False)
    assert grade(mc("A"), "probably the first one") == (None, False)
