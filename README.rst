Bonsai
======

Bonsai is a project that aims to use AI to check and fix specifications to make
them *formal specifications*. A formal specification is a specification that
uses RFC 2119 keywords like *MUST*, *SHOULD*, *MAY*, etc. to create *normative
sentences*. A *formal normative sentence* uses RFC 2119 keywords. A sentence
that does not use any RFC 2119 keywords but has the same *meaning* that one
who does is called an *informal normative sentence*.

A model is used to convert a specification into sentences that can be analyzed
to determine if they are normative sentences or not.

Bonsai is trained using sentences that are generated with ChatGPT. Those
sentences are used to create a model that can check if a sentence is a
normative sentence (formal or informal) or not.

If an informal normative sentence is found, then another model is used to
generate the equivalent formal normative sentence.

This means there are 3 AI models that make up Bonsai:

1. The sentence segmentator, the model that transforms a document into
   sentences.
2. The sentence classifier, the model that classifies a sentence into a formal
   normative sentence, an informal normative sentence or a non-normative
   sentence.
3. The sentence generator, the model that generates a formal normative sentence
   for an informal normative sentence.

Development
-----------

First, install `pipenv`. To run the tests, run `pipenv run pytest`. To add a
new dependency for `bonsai`, run `pipenv install new-dependency`. To add a new
development dependency for `bonsai`, run `pipenv install -d new-dependency`.
Add both `Pipfile` and `Pipfile.lock` changes to git.

Run `pipenv run spacy download en_core_web_sm` too.
