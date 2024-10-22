# CoffeeTalk ☕

CoffeeTalk is a tool that transforms codebases into fine-tuned language models. It offers flexible training configurations tailored to various hardware profiles, accommodating both memory-constrained systems and high-performance setups. This makes it accessible for developers with a wide range of hardware resources.

CoffeeTalk can be used by developers to enhance code understanding, automate code documentation, and generate highly contextual code explanations. It can also serve as a backend for AI systems that require deep code comprehension capabilities, enabling them to assist in code reviews, refactoring, and debugging. By integrating CoffeeTalk, both human developers and AI systems can leverage its capabilities to improve productivity and code quality through a nuanced understanding of code context.

## Setup and Usage

1. **Create a virtual environment:**
   ```
   python -m venv venv
   ```
2. **Activate the virtual environment:**
   
   On Mac/Linux:
   ```
   source venv/bin/activate
   ```
3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. **Run the script:**

   If `TARGET_REPO_PATH` is not set, you will be prompted for it at each invocation:
   ```
   python src/main.py
   ```
   **or:** (for example)
   ```
   TARGET_REPO_PATH="/path/to/your/target/repo" python src/main.py
   ```
5. **Deactivate the virtual environment:**
   ```
   deactivate
   ```

## Process Diagram

```mermaid
graph TD
    Start[Target Code / Repository] --> CoffeeTalk

    subgraph CoffeeTalk["CoffeeTalk ☕"]
        LD[Language Detection] --> CSE

        PTM_1[Pre-trained Model] --> Inference
        CSE[Code Snippet Extraction] --> Inference
        HP_1[Hardware Profile] --> Inference

        Inference{Inference} --> TD

        PTM_2[Pre-trained Model] --> Training
        TD[Training Data] --> Training
        HP_2[Hardware Profile] --> Training

        Training{Training} --> FTM
    end

    FTM[Fine-tuned Model] --> End[Inference by User/AI]
```

## Configuration

### Hardware Profiles

CoffeeTalk supports different hardware profiles to optimize training for various systems. The profiles are defined in `src/hardware_profiles.py` and can be selected by setting the `HARDWARE_PROFILE` environment variable. Available profiles include:

- **apple_silicon**: Optimized for Apple Silicon Macs with limited memory.
- **cuda_gpu**: For systems with CUDA-enabled GPUs.
- **cpu**: For CPU-only systems.

### Model Selection

You can choose the model to use for training by setting the `TRAINING_MODEL` environment variable. If not set, the script defaults to `distilgpt2`, which is a highly memory-efficient model.
