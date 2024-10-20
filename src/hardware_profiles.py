from dataclasses import dataclass, field
from typing import Dict


@dataclass
class HardwareProfile:
    max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: int
    fp16: bool
    env_vars: Dict[str, str] = field(default_factory=dict)
    use_mps_device: bool = False
    use_cpu: bool = True
    tokenizers_parallelism: bool = False
    torch_num_interop_threads: int = 1
    torch_num_threads: int = 1


apple_silicon = HardwareProfile(
    max_length=128,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    fp16=True,
    env_vars={
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
        'CUDA_VISIBLE_DEVICES': '',
        'PYTORCH_MPS_ENABLE': '0',
    },
    use_mps_device=False,
    use_cpu=True,
    torch_num_threads=1,
    torch_num_interop_threads=1
)

# TODO: test this
cuda_gpu = HardwareProfile(
    max_length=256,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    fp16=True,
    env_vars={},
    use_mps_device=False,
    use_cpu=False,
    torch_num_threads=1,
    torch_num_interop_threads=1
)

# TODO: test this
cpu = HardwareProfile(
    max_length=128,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    weight_decay=0.01,
    num_train_epochs=3,
    fp16=False,
    env_vars={
        'CUDA_VISIBLE_DEVICES': '',
    },
    use_mps_device=False,
    use_cpu=True,
    torch_num_threads=1,
    torch_num_interop_threads=1
)

profiles = {
    "apple_silicon": apple_silicon,
    "cuda_gpu": cuda_gpu,
    "cpu": cpu
}
