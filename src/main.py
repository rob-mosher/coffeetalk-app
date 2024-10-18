import os
import subprocess


def run_script(script_name, target_repo_path):
    env = os.environ.copy()
    env['TARGET_REPO_PATH'] = target_repo_path
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    result = subprocess.run(['python', script_path], env=env, check=True)
    if result.returncode != 0:
        raise Exception(f"Error running {script_name}")


def main():
    target_repo_path = os.getenv('TARGET_REPO_PATH')

    if not target_repo_path:
        print("TARGET_REPO_PATH environment variable not found.")
        target_repo_path = input(
            "Please enter the absolute or relative path to the target repository: ")

    if not os.path.exists(target_repo_path):
        print(f"Error: The specified repository path '{target_repo_path}' does not exist.")
        return

    print("Generating training data...")
    run_script('generate_training_data.py', target_repo_path)

    # print("Training model...")
    # run_script('train.py', target_repo_path)

    print("Data generation and model training completed successfully.")


if __name__ == "__main__":
    main()
