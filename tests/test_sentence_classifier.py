from bonsai.sentence_classifier import classify_sentence


def test_classify_sentence():

    assert classify_sentence("It is mandatory to write new files") == 2
    assert classify_sentence("It is optional to write new files") == 1
    assert classify_sentence("New files are written by the user") == 0
