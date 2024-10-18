import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling


def load_dataset_from_json(json_path):
    data_files = {"train": json_path}
    dataset = load_dataset('json', data_files=data_files)
    return dataset


def tokenize_function(examples, tokenizer, max_length):
    inputs = tokenizer(examples['prompt'], truncation=True, max_length=max_length)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['completion'], truncation=True, max_length=max_length)
    inputs["labels"] = labels["input_ids"]
    return inputs


def fine_tune_model(dataset, model_name, output_dir, num_train_epochs=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    max_length = 128

    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=num_train_epochs,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=1e-4,
        warmup_steps=50,
        fp16=False,
        optim="adamw_torch",
        max_grad_norm=0.5,
        weight_decay=0.01,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)


def main():
    json_path = os.path.join('data', 'training_data.json')
    dataset = load_dataset_from_json(json_path)

    # Use environment variable for model name, with fallback
    model_name = os.getenv('TRAINING_MODEL')
    if model_name:
        print(f"Using specified model: {model_name}")
    else:
        model_name = 'distilgpt2'
        print(f"TRAINING_MODEL not set. Falling back to default model: {model_name}")

    output_dir = "./results"

    fine_tune_model(dataset, model_name, output_dir)


if __name__ == "__main__":
    main()
