import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from hardware_profiles import profiles


def setup_environment(profile):
    for key, value in profile.env_vars.items():
        os.environ[key] = value
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def load_dataset_from_json(json_path):
    data_files = {"train": json_path}
    return load_dataset('json', data_files=data_files)


def tokenize_function(examples, tokenizer, max_length):
    inputs = tokenizer(examples['prompt'], truncation=True, max_length=max_length)
    labels = tokenizer(examples['completion'], truncation=True, max_length=max_length)
    inputs["labels"] = labels["input_ids"]
    return inputs


def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def prepare_dataset(dataset, tokenizer, max_length):
    return dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names
    )


def get_training_args(output_dir, profile):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=profile.num_train_epochs,
        per_device_train_batch_size=profile.per_device_train_batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        learning_rate=profile.learning_rate,
        weight_decay=profile.weight_decay,
        fp16=profile.fp16,
        logging_dir='./logs',
        logging_steps=10,
        use_cpu=profile.no_cuda,
        use_mps_device=profile.use_mps_device,
    )


def fine_tune_model(dataset, model_name, output_dir, hardware_profile):
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

    trainer.train()
    trainer.save_model(output_dir)


def main(target_repo_path: str, hardware_profile_name: str):
    hardware_profile = profiles[hardware_profile_name]

    print(f"Using hardware profile: {hardware_profile_name}")

    setup_environment(hardware_profile)

    json_path = os.path.join('data', 'training_data.json')
    dataset = load_dataset_from_json(json_path)

    model_name = os.getenv('TRAINING_MODEL', 'distilgpt2')
    print(f"Using model: {model_name}")

    output_dir = os.path.join(target_repo_path, "results")
    fine_tune_model(dataset, model_name, output_dir, hardware_profile)

    print("Model training completed successfully.")
