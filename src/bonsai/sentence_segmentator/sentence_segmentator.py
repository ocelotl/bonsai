from spacy import load

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
