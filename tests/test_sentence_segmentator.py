from os.path import abspath, dirname, join

from bonsai.sentence_segmentator import get_paragraphs, get_sentences

file_path = dirname(abspath(__file__))


def test_get_sentences():

    paragraphs = get_paragraphs(
        join(file_path, "./markdown/telemetry-stability.md")
    )

    for paragraph in paragraphs:
        sentences = get_sentences(paragraph)
        assert len(sentences) >= 1
