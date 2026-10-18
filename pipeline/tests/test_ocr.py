import random

from ocr import cer, levenshtein, normalize


def slow_levenshtein(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (ca != cb)))
        row = new
    return row[-1]


def test_levenshtein_matches_textbook_version():
    rng = random.Random(0)
    assert levenshtein("kitten", "sitting") == 3
    for _ in range(200):
        a = "".join(rng.choice("ab $1.") for _ in range(rng.randint(0, 25)))
        b = "".join(rng.choice("ab $1.") for _ in range(rng.randint(0, 25)))
        assert levenshtein(a, b) == slow_levenshtein(a, b)


def test_cer_ignores_layout_and_markdown():
    assert normalize("| Item | $5.00 |\n----\n**Total**") == "Item $5.00 Total"
    assert cer("Total\n$5.00", "Total  $5.00") == 0.0
    assert cer("abcd", "abXd") == 0.25
