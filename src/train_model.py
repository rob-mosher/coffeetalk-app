import os
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from utils.hardware_profiles import HardwareProfile


def load_dataset_from_json(json_path):
    logging.debug(f"Loading dataset from {json_path}")
    data_files = {"train": json_path}
    dataset = load_dataset('json', data_files=data_files)
    logging.debug(f"Dataset loaded with {len(dataset['train'])} examples")
    return dataset


def tokenize_function(examples, tokenizer, max_length):
    logging.debug(f"Tokenizing batch of {len(examples['prompt'])} examples")
    inputs = tokenizer(examples['prompt'], truncation=True, max_length=max_length)
    labels = tokenizer(examples['completion'], truncation=True, max_length=max_length)
    inputs["labels"] = labels["input_ids"]
    return inputs


def get_model_and_tokenizer(model_name):
    logging.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    logging.debug("Model and tokenizer loaded successfully")
    return model, tokenizer


def prepare_dataset(dataset, tokenizer, max_length):
    logging.info("Preparing dataset for training...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    logging.info("Dataset preparation completed")
    return tokenized_dataset


def get_training_args(output_dir, profile):
    logging.info("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=profile.num_train_epochs,
        per_device_train_batch_size=profile.per_device_train_batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        learning_rate=profile.learning_rate,
        weight_decay=profile.weight_decay,
        fp16=profile.fp16,
        logging_dir='./logs',
        logging_steps=10,
    )

    logging.debug("Training Arguments:")
    logging.debug(f"Output Directory: {output_dir}")
    logging.debug(f"Number of Train Epochs: {profile.num_train_epochs}")
    logging.debug(f"Per Device Train Batch Size: {profile.per_device_train_batch_size}")
    logging.debug(f"Gradient Accumulation Steps: {profile.gradient_accumulation_steps}")
    logging.debug(f"Learning Rate: {profile.learning_rate}")
    logging.debug(f"Weight Decay: {profile.weight_decay}")
    logging.debug(f"FP16: {profile.fp16}")

    return training_args


def fine_tune_model(dataset, model_name, output_dir, hardware_profile):
    logging.info("Starting model fine-tuning process...")

    model, tokenizer = get_model_and_tokenizer(model_name)
    tokenized_datasets = prepare_dataset(dataset, tokenizer, hardware_profile.max_length)

    training_args = get_training_args(output_dir, hardware_profile)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    logging.info("Starting training...")
    trainer.train()

    logging.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    logging.debug("Model saved successfully")


def main(target_repo_path: str, hardware_profile: HardwareProfile):
    logging.info("Starting model training process...")

    json_path = os.path.join('data', 'training_data.json')
    dataset = load_dataset_from_json(json_path)

    model_name = os.getenv('TRAINING_MODEL', 'distilgpt2')
    logging.info(f"Using model: {model_name}")

    output_dir = os.path.join(target_repo_path, "results")
    fine_tune_model(dataset, model_name, output_dir, hardware_profile)

    logging.info("Model training completed successfully")
