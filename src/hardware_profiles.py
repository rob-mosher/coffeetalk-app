"""
This module defines hardware profiles for different computing environments.
"""

from dataclasses import dataclass, field
from typing import Dict
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
            return torch.device("cpu")
        elif self.use_mps_device and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            # Fallback to CPU if no other options are available
            return torch.device("cpu")

    def display_settings(self):
        print("Hardware Profile Settings:")
        print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Max Length: {self.max_length}")
        print(f"Number of Train Epochs: {self.num_train_epochs}")
        print(f"Per Device Train Batch Size: {self.per_device_train_batch_size}")
        print(f"Weight Decay: {self.weight_decay}")
        print(f"FP16: {self.fp16}")
        print(f"Use CPU: {self.use_cpu}")
        print(f"Use MPS Device: {self.use_mps_device}")
        print(f"Environment Variables: {self.env_vars}")
        print(f"Tokenizers Parallelism: {self.tokenizers_parallelism}")
        print(f"PyTorch Num Interop Threads: {self.torch_num_interop_threads}")
        print(f"PyTorch Num Threads: {self.torch_num_threads}")


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
