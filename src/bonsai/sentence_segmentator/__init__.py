from spacy import load
from typing import List
from re import compile as re_compile, DOTALL
from markdown import markdown
from nltk import download
from nltk.tokenize import sent_tokenize

download("punkt")
# In order to use en_core_web_sm:
# python -m spacy download en_core_web_sm
nlp = load("en_core_web_sm")

doc = nlp(
    u"This is the first sentence. "
    "This is another sentence. "
    "This is the last sentence."
)

for sentence in doc.sents:
    print(sentence)

paragraph_regex = re_compile(r"<p>.*?</p>", DOTALL)
html_tag_regex = re_compile(r"<.*?>")
newline_regex = re_compile(r"\n")


def get_paragraphs(markdown_file_path: str) -> List[str]:

    paragraphs = []

    with open(markdown_file_path) as markdown_file_file:

        for paragraph in paragraph_regex.findall(
            markdown(markdown_file_file.read())
        ):

            paragraphs.append(
                html_tag_regex.sub(
                    "",
                    newline_regex.sub(" ", paragraph)
                )
            )

    return paragraphs


def get_sentences(paragraph: str) -> List[str]:

    return sent_tokenize(paragraph)
