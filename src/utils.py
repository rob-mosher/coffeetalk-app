import os
import torch
from hardware_profiles import profiles


def determine_hardware_profile():
    hardware_profile_name = os.getenv('HARDWARE_PROFILE', 'apple_silicon')
    if hardware_profile_name not in profiles:
        print(f"Invalid hardware profile: {hardware_profile_name}. Using default (apple_silicon).")
        hardware_profile_name = 'apple_silicon'

    hardware_profile = profiles[hardware_profile_name]
    print(f"Using hardware profile: {hardware_profile_name}")
    return hardware_profile


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


def setup_hardware_environment(hardware_profile):
    os.environ.update(hardware_profile.env_vars)
    os.environ['TOKENIZERS_PARALLELISM'] = str(hardware_profile.tokenizers_parallelism).lower()

    torch.set_num_threads(hardware_profile.torch_num_threads)
    torch.set_num_interop_threads(hardware_profile.torch_num_interop_threads)
