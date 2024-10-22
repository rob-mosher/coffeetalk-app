import os
import json
import languages.javascript as javascript
import languages.typescript as typescript
import languages.python as python
import languages.go as go
from dotenv import load_dotenv

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
            snippets = language_module.extract_code_snippets(content)
            return snippets
        else:
            return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def get_code_snippets_from_repo(target_repo_path):
    snippets = []
    for root, dirs, files in os.walk(target_repo_path):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            file_extension = os.path.splitext(file)[1]
            if file_extension in LANGUAGE_MODULES:
                file_path = os.path.join(root, file)
                file_snippets = extract_code_snippets(file_path)
                snippets.extend(file_snippets)
    return snippets


def create_training_data(snippets):
    training_data = []
    for snippet in snippets:
        prompt = f"Explain the following code snippet:\n\n{
            snippet}\n\nExplanation:"
        completion = " "  # Placeholder for the explanation, determined later
        training_data.append({"prompt": prompt, "completion": completion})
    return training_data


def save_training_data_to_json(training_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(training_data, outfile, indent=2)


def main(target_repo_path: str):
    print("Generating training data...")

    print("Extracting code snippets...")
    snippets = get_code_snippets_from_repo(target_repo_path)
    print(f"Extracted {len(snippets)} code snippets.")

    print("Creating training data...")
    training_data = create_training_data(snippets)
    print(f"Created {len(training_data)} training data entries.")

    print("Saving training data...")
    output_file = os.path.join('data', 'training_data.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_training_data_to_json(training_data, output_file)
    print(f"Training data saved successfully to: {output_file}")

    print("Generation of training data completed.")
