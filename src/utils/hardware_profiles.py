"""
This module defines hardware profiles for different computing environments.
"""

from dataclasses import dataclass, field
from typing import Dict
import logging
import torch


@dataclass
class HardwareProfile:
    ### Model and training hyperparameters ###
    gradient_accumulation_steps: int  # Steps to accumulate gradients before backward/update pass
    learning_rate: float  # Step size at each iteration towards loss function minimum
    max_length: int  # Maximum sequence length for input
    num_train_epochs: int  # Total number of training epochs to perform
    per_device_train_batch_size: int  # Batch size per device during training
    weight_decay: float  # L2 penalty (regularization term) parameter

    ### Device and precision settings ###
    fp16: bool = False  # Use half-precision floating point format for efficiency
    use_cpu: bool = True  # Use CPU for computations
    use_mps_device: bool = False  # Use Apple's Metal Performance Shaders for GPU acceleration

    ### Environment variables ###
    env_vars: Dict[str, str] = field(default_factory=dict)  # Additional environment variables

    ### Parallelism settings ###
    tokenizers_parallelism: bool = False  # Control parallelism in tokenization

    ### PyTorch CPU settings ###
    # Note: torch_num_threads and torch_num_interop_threads only affect CPU operations
    # and have no impact on GPU computations.
    torch_num_interop_threads: int = 1  # Threads for interop parallelism on CPU
    torch_num_threads: int = 1  # Threads for intraop parallelism on CPU

    def get_device(self):
        """Determine the appropriate device based on the hardware profile settings."""
        if self.use_cpu:
            logging.debug("Using CPU device")
            return torch.device("cpu")
        elif self.use_mps_device and torch.backends.mps.is_available():
            logging.debug("Using MPS device")
            return torch.device("mps")
        elif torch.cuda.is_available():
            logging.debug("Using CUDA device")
            return torch.device("cuda")
        else:
            logging.warning("No GPU device available, falling back to CPU")
            return torch.device("cpu")

    def display_settings(self):
        logging.debug("Hardware Profile Settings:")
        logging.debug(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        logging.debug(f"Learning Rate: {self.learning_rate}")
        logging.debug(f"Max Length: {self.max_length}")
        logging.debug(f"Number of Train Epochs: {self.num_train_epochs}")
        logging.debug(f"Per Device Train Batch Size: {self.per_device_train_batch_size}")
        logging.debug(f"Weight Decay: {self.weight_decay}")
        logging.debug(f"FP16: {self.fp16}")
        logging.debug(f"Use CPU: {self.use_cpu}")
        logging.debug(f"Use MPS Device: {self.use_mps_device}")
        logging.debug(f"Environment Variables: {self.env_vars}")
        logging.debug(f"Tokenizers Parallelism: {self.tokenizers_parallelism}")
        logging.debug(f"PyTorch Num Interop Threads: {self.torch_num_interop_threads}")
        logging.debug(f"PyTorch Num Threads: {self.torch_num_threads}")


apple_silicon = HardwareProfile(
    ### Model and training hyperparameters ###
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    max_length=128,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    weight_decay=0.01,

    ### Device and precision settings ###
    fp16=False,
    use_cpu=False,
    use_mps_device=True,

    ### Environment variables ###
    env_vars={
        'CUDA_VISIBLE_DEVICES': '',
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
        'PYTORCH_MPS_ENABLE': '1',
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
    },

    ### Parallelism settings ###
    tokenizers_parallelism=False,

    ### PyTorch CPU settings ###
    torch_num_interop_threads=1,
    torch_num_threads=1,
)

# TODO: test this
cuda_gpu = HardwareProfile(
    ### Model and training hyperparameters ###
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_length=256,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    weight_decay=0.01,

    ### Device and precision settings ###
    fp16=True,
    use_cpu=False,
    use_mps_device=False,

    ### Environment variables ###
    env_vars={},

    ### Parallelism settings ###
    tokenizers_parallelism=True,

    ### PyTorch CPU settings ###
    torch_num_interop_threads=1,
    torch_num_threads=1,
)

# TODO: test this
cpu = HardwareProfile(
    ### Model and training hyperparameters ###
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    max_length=128,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    weight_decay=0.01,

    ### Device and precision settings ###
    fp16=False,
    use_cpu=True,
    use_mps_device=False,

    ### Environment variables ###
    env_vars={
        'CUDA_VISIBLE_DEVICES': '',
    },

    ### Parallelism settings ###
    tokenizers_parallelism=False,

    ### PyTorch CPU settings ###
    torch_num_interop_threads=1,
    torch_num_threads=1,
)

profiles = {
    "apple_silicon": apple_silicon,
    "cuda_gpu": cuda_gpu,
    "cpu": cpu
}
