import argparse
import logging
from generate_training_data import main as generate_training_data
from train_model import main as train_model
from utils import (
    get_hardware_profile,
    get_target_repo_path,
    setup_hardware_environment
)
from utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description='CoffeeTalk: Code Analysis Tool')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        logging.info("Starting CoffeeTalk...")

        target_repo_path = get_target_repo_path()
        hardware_profile = get_hardware_profile()

        setup_hardware_environment(hardware_profile)
        generate_training_data(target_repo_path, hardware_profile)
        train_model(target_repo_path, hardware_profile)

        logging.info("All CoffeeTalk tasks completed successfully")
    except ValueError as e:
        logging.error(str(e))
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
