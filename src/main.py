import os
from generate_training_data import main as generate_training_data
from train_model import main as train_model
from utils import (
    get_hardware_profile,
    get_target_repo_path,
    setup_hardware_environment
)


def main():
    try:
        print("Starting CoffeeTalk...")

        target_repo_path = get_target_repo_path()
        hardware_profile = get_hardware_profile()

        setup_hardware_environment(hardware_profile)
        generate_training_data(target_repo_path, hardware_profile)
        train_model(target_repo_path, hardware_profile)

        print("All CoffeeTalk tasks completed successfully.")
    except ValueError as e:
        print(str(e))
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
