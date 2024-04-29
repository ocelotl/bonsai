from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
from torch import no_grad, argmax
from os.path import abspath, dirname, join

file_path = dirname(abspath(__file__))

# Load model and tokenizer from disk
_model = DistilBertForSequenceClassification.from_pretrained(
    join(file_path, "model")
)
_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def classify_sentence(sentence):

    # Perform inference
    with no_grad():
        output = _model(**_tokenizer(sentence, return_tensors='pt'))

    # Get the predicted class
    return argmax(output.logits, dim=1).item()
