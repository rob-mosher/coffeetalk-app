import pytest
import os
import torch
from src.utils.hardware_profiles import HardwareProfile, profiles
from src.utils.constants import DEFAULT_HARDWARE_PROFILE
from src.utils.lib import setup_hardware_environment

@pytest.fixture(autouse=True)
def clear_env():
    """Clear relevant environment variables before each test"""
    if 'HARDWARE_PROFILE' in os.environ:
        del os.environ['HARDWARE_PROFILE']
    yield

def test_hardware_profile_creation():
    # TODO: Implement this test
    pass

def test_default_profile():
    # TODO: Implement this test
    pass

def test_environment_setup():
    profile = profiles['cpu']
    setup_hardware_environment(profile)
    
    assert os.environ.get('TOKENIZERS_PARALLELISM') == 'false'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    
    assert torch.get_num_threads() == profile.torch_num_threads
    assert torch.get_num_interop_threads() == profile.torch_num_interop_threads

def test_device_selection():
    cpu_profile = profiles['cpu']
    assert str(cpu_profile.get_device()) == 'cpu'
