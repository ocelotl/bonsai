from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# Load and preprocess dataset
dataset = load_dataset(
    'csv',
    data_files={
        'train': 'path/to/your/train.csv',
        'test': 'path/to/your/test.csv'
    }
)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")


def tokenize_function(examples):
    model_inputs = tokenizer(
        examples['input'], max_length=128, truncation=True)
    labels = tokenizer(examples['output'], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer, model="facebook/bart-base")

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

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

# Initialize and train the model
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)

# Generate a prediction
inputs = tokenizer("Your input sentence here", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
