from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from pathlib import Path

formal_normative_sentence_generator_path = Path(
    "/home/tigre/github/ocelotl/bonsai/src/bonsai/"
    "formal_normative_sentence_generator/data/model"
)
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(
    formal_normative_sentence_generator_path,
)
loaded_tokenizer = AutoTokenizer.from_pretrained(
    formal_normative_sentence_generator_path
)


def generate_sentence(input_sentence: str):

    inputs = loaded_tokenizer(input_sentence, return_tensors="pt")
    outputs = loaded_model.generate(**inputs)
    return loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)


print(generate_sentence("It is mandatory to do that"))
