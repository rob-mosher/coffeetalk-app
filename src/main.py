import os
from generate_training_data import main as generate_training_data
from train_model import main as train_model


def determine_hardware_profile():
    from hardware_profiles import profiles
    hardware_profile_name = os.getenv('HARDWARE_PROFILE', 'apple_silicon')
    if hardware_profile_name not in profiles:
        print(f"Invalid hardware profile: {hardware_profile_name}. Using default (apple_silicon).")
        hardware_profile_name = 'apple_silicon'
    print(f"Using hardware profile: {hardware_profile_name}")
    return hardware_profile_name


def determine_target_repo_path():
    target_repo_path = os.getenv('TARGET_REPO_PATH')
    if not target_repo_path:
        print("TARGET_REPO_PATH environment variable not found.")
        target_repo_path = input(
            "Please enter the absolute or relative path to the target repository: ")

    if not os.path.exists(target_repo_path):
        raise ValueError(f"Error: The specified repository path '{
                         target_repo_path}' does not exist.")

    print(f"Using target repository: {target_repo_path}")
    return target_repo_path


def main():
    try:
        target_repo_path = determine_target_repo_path()
        hardware_profile = determine_hardware_profile()

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
