"""imdb.py 的单元测试：确认清洗逻辑在边界情况下不炸。

    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imdb import clean_review, review_to_sentences, review_to_words  # noqa: E402


def test_removes_html_tags():
    assert "br" not in review_to_words("Great movie.<br /><br />Loved it.")


def test_unescapes_html_entities():
    # &amp; 先被还原成 &，再作为非字母字符被剔除。
    assert review_to_words("Tom &amp; Jerry", remove_stopwords=False) == ["tom", "jerry"]


def test_lowercases_and_drops_digits():
    assert review_to_words("Rated 10 out of 10 STARS") == ["rated", "stars"]


def test_stopwords_toggle():
    text = "this is a very good movie"
    assert review_to_words(text, remove_stopwords=True) == ["good", "movie"]
    assert "is" in review_to_words(text, remove_stopwords=False)


@pytest.mark.parametrize("raw", ["", "   ", "12345", "!!!???", "<br /><br />", None])
def test_degenerate_input_returns_empty(raw):
    """全是标点/数字/空的输入必须安全返回空，而不是抛异常。"""
    assert review_to_words(raw) == []
    assert clean_review(raw) == ""


def test_sentence_splitting():
    sentences = review_to_sentences("Good film. I liked it! Did you?")
    assert len(sentences) == 3
    # 分句时保留停用词——它们是 Word2Vec 的上下文。
    assert sentences[1] == ["i", "liked", "it"]


def test_sentences_skip_empty_chunks():
    assert review_to_sentences("Wow!!! ... 123 ... Nice.") == [["wow"], ["nice"]]


def test_clean_review_is_space_joined():
    assert clean_review("An excellent, excellent film!") == "excellent excellent film"
