from json import load
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from google.colab import drive
from pathlib import Path

verb = "MUST"
size = "11509"

drive.mount('/content/drive')

formal_normative_sentence_generator_path = Path(
    "/content/drive/My Drive/bonsai/formal_normative_sentence_generator"
)

with open(
    formal_normative_sentence_generator_path.joinpath(f"{verb}_{size}.json")
) as sentences_json_file:
    sentences = load(sentences_json_file)

# Convert the list of lists into a dictionary
data_dict = {
    "input": [item[0] for item in sentences],
    "output": [item[1] for item in sentences]
}

# Create a Hugging Face Dataset from the dictionary
dataset = Dataset.from_dict(data_dict)

# Split the dataset into train and test sets
split_datasets = dataset.train_test_split(test_size=0.2)
train_dataset = split_datasets['train']
test_dataset = split_datasets['test']

# Initialize the tokenizer
model_checkpoint = "facebook/bart-base"  # or "t5-small", etc.
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize the dataset


def tokenize_function(examples):
    model_inputs = tokenizer(
        examples['input'], max_length=128, truncation=True)
    labels = tokenizer(examples['output'], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Prepare the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_checkpoint)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

trainer.save_model(str(formal_normative_sentence_generator_path))
# Evaluate the model
metrics = trainer.evaluate()
print(metrics)

# Generate a prediction
inputs = tokenizer("It is mandatory do have an API.", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
