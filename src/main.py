import os
from generate_training_data import main as generate_training_data
from train_model import main as train_model
from utils import (
    determine_hardware_profile,
    determine_target_repo_path,
    setup_hardware_environment
)


def main():
    try:
        target_repo_path = determine_target_repo_path()
        hardware_profile = determine_hardware_profile()

        print("Configuring environment variables for hardware profile...")
        setup_hardware_environment(hardware_profile)

        print("Generating training data...")
        generate_training_data(target_repo_path)

        print("Training model...")
        train_model(target_repo_path, hardware_profile)

        print("All tasks completed successfully.")
    except ValueError as e:
        print(str(e))
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
