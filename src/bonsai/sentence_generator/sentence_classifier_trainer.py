"""
This code is to be run on google colab using 2 files:

sentences_size.json
labels_size.json

Replace size with the actual size number that corresponds to the actual file
name.
"""

from torch import tensor
from json import load
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from google.colab import drive
from pathlib import Path

size = 500

drive.mount('/content/drive')
sentence_classifier_path = Path(
    f"/content/drive/My Drive/sentence_classifier/{size}"
)


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: tensor(val[idx]) for key, val in self.encodings.items()
        }
        item['labels'] = tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


with open(
    sentence_classifier_path.joinpath(f"sentences_{size}.json")
) as sentences_json_file:
    sentences = load(sentences_json_file)

with open(
    sentence_classifier_path.joinpath(f"labels_{size}.json")
) as labels_json_file:
    labels = load(labels_json_file)

# Split the dataset into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=3
)

# Create datasets
train_dataset = CustomDataset(
    train_sentences, train_labels, tokenizer, max_length=128
)
val_dataset = CustomDataset(
    val_sentences, val_labels, tokenizer, max_length=128
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./distilbert_classification",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

trainer.save_model(str(sentence_classifier_path))
