import os
import json
import logging
from languages import javascript, typescript, python, go
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from utils.constants import DEFAULT_MODEL_NAME

load_dotenv()

LANGUAGE_MODULES = {
    '.js': javascript,
    '.jsx': javascript,
    '.ts': typescript,
    '.tsx': typescript,
    '.py': python,
    '.go': go,
}

IGNORE_DIRS = {'node_modules', '.git', '__pycache__'}


def extract_code_snippets(file_path):
    _, file_extension = os.path.splitext(file_path)

    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        language_module = LANGUAGE_MODULES.get(file_extension)
        if language_module:
            logging.debug(f"Processing {file_path} with {language_module.__name__}")
            snippets = language_module.extract_code_snippets(content)
            return snippets
        else:
            logging.debug(f"Skipping unsupported file type: {file_path}")
            return []
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return []


def get_code_snippets_from_repo(target_repo_path):
    logging.debug(f"Scanning repository: {target_repo_path}")
    snippets = []
    for root, dirs, files in os.walk(target_repo_path):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        logging.debug(f"Scanning directory: {root}")

        for file in files:
            file_extension = os.path.splitext(file)[1]
            if file_extension in LANGUAGE_MODULES:
                file_path = os.path.join(root, file)
                file_snippets = extract_code_snippets(file_path)
                snippets.extend(file_snippets)
                logging.debug(f"Found {len(file_snippets)} snippets in {file_path}")
    return snippets


def create_training_data(snippets, hardware_profile):
    training_data = []

    # Load the model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(DEFAULT_MODEL_NAME)
    tokenizer = GPT2Tokenizer.from_pretrained(DEFAULT_MODEL_NAME)

    # Configure the tokenizer
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set
    model.config.pad_token_id = model.config.eos_token_id

    logging.debug("Model Settings:")
    logging.debug(f"Model Name: {DEFAULT_MODEL_NAME}")
    logging.debug(f"Pad Token ID: {model.config.pad_token_id}")

    device = hardware_profile.get_device()
    logging.info(f"Using device: {device}")
    model.to(device)

    for i, snippet in enumerate(snippets, 1):
        logging.debug(f"Processing snippet {i}/{len(snippets)}")
        prompt = f"Explain the following code snippet:\n\n{snippet}\n\nExplanation:"

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True,
                           truncation=True, max_length=hardware_profile.max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logging.debug(f"Input shapes: {inputs}")

        # Generate completion
        try:
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=hardware_profile.max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True
                )

            completion = tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract only the generated explanation
            explanation = completion.split("Explanation:")[1].strip()

            training_data.append({"prompt": prompt, "completion": explanation})
            logging.debug(f"Successfully generated explanation for snippet {i}")
        except RuntimeError as e:
            logging.error(f"Error generating completion for snippet {i}: {e}")
            continue

    return training_data


def save_training_data_to_json(training_data, output_file):
    logging.debug(f"Saving training data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(training_data, outfile, indent=2)
    logging.debug("Training data saved successfully")


def main(target_repo_path: str, hardware_profile):
    logging.info("Generating training data...")

    logging.info("Extracting code snippets...")
    snippets = get_code_snippets_from_repo(target_repo_path)
    logging.info(f"Extracted {len(snippets)} code snippets")

    logging.info("Creating training data...")
    training_data = create_training_data(snippets, hardware_profile)
    logging.info(f"Created {len(training_data)} training data entries")

    logging.info("Saving training data...")
    output_file = os.path.join('data', 'training_data.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_training_data_to_json(training_data, output_file)
    logging.info(f"Training data saved successfully to: {output_file}")

    logging.info("Generation of training data completed")
