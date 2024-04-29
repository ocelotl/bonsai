Bonsai Sentence Classifier
==========================

The *sentence classifier* is an AI model that classifies sentences into
categories.

This data needs *sentences* and *labels*. Both can be found in the `data`
directory. Each sentence in `data/sentences.json` is associated with a label
in `data/labels.json`.

There are 3 categories of sentences:

#. Regular sentences (label 0)
#. Informal Normative MAY sentences (label 1)
#. Informal Normative MUST sentences (label 2)

The latter 2 categories group sentences that do not include the words MAY or
MUST, but are semantically equivalent to sentences that would include these
words.

The model is generated with `sentence_classifier_trainer.py`. This script is to
be run in Google Colab, and it uses the two files mentioned above,
`sentences.json` and `labels.json` to generate three files:

#. `config.json`
#. `model.safetensors`
#. `training_args.bin`

Once this script is executed in Google Colab, these 3 files will be generated
in a Google Drive directory. Download these 3 files and save them in the
`model` directory.

The model can now be used to classify a sentence with the `classify_sentence`
function.
