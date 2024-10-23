import os
import torch
import logging
from .hardware_profiles import profiles
from .constants import DEFAULT_HARDWARE_PROFILE


def get_hardware_profile():
    logging.info("Getting hardware profile...")

    hardware_profile_name = os.getenv('HARDWARE_PROFILE', DEFAULT_HARDWARE_PROFILE)
    if hardware_profile_name not in profiles:
        logging.warning(f"Invalid hardware profile: {hardware_profile_name}, using default ({DEFAULT_HARDWARE_PROFILE})")
        hardware_profile_name = DEFAULT_HARDWARE_PROFILE

    hardware_profile = profiles[hardware_profile_name]
    logging.info(f"Using hardware profile: {hardware_profile_name}")

    hardware_profile.display_settings()

    return hardware_profile


def get_target_repo_path():
    logging.info("Getting target repository path...")

    target_repo_path = os.getenv('TARGET_REPO_PATH')
    if not target_repo_path:
        logging.warning("TARGET_REPO_PATH environment variable not found")
        target_repo_path = input(
            "Please enter the absolute or relative path to the target repository: ")

    if not os.path.exists(target_repo_path):
        raise ValueError(f"Error: The specified repository path '{
                         target_repo_path}' does not exist")

    logging.info(f"Using target repository: {target_repo_path}")
    return target_repo_path


def setup_hardware_environment(hardware_profile):
    logging.info("Configuring environment variables for hardware profile...")

    os.environ.update(hardware_profile.env_vars)
    os.environ['TOKENIZERS_PARALLELISM'] = str(hardware_profile.tokenizers_parallelism).lower()

    torch.set_num_threads(hardware_profile.torch_num_threads)
    torch.set_num_interop_threads(hardware_profile.torch_num_interop_threads)

    logging.info(f"Using device: {hardware_profile.get_device()}")
    logging.info("Environment variables configured successfully")
