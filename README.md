# coffeetalk-app

## Setup and Usage

1. Create a virtual environment:
   ```
   python -m venv venv
   ```
2. Activate the virtual environment:
   On Mac/Linux:
   ```
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the script. If `TARGET_REPO_PATH` is not declared, you will be prompted for it at each invocation:
   ```
   python src/main.py
   ```
   or: (for example)
   ```
   TARGET_REPO_PATH="/path/to/your/target/repo" python src/main.py
   ```
5. When finished, deactivate the virtual environment:
   ```
   deactivate
   ```

## Configuration Notes

### Training Configuration for M-based Macs

The current training configuration in `src/train_model.py` is optimized for M-based Macs (M1, M2, etc.) with limited memory:

- Using half-precision (16-bit) floating-point format to reduce memory usage.
- `max_length=256`: Reduced sequence length to decrease the size of each training example.
- `per_device_train_batch_size=1` and `gradient_accumulation_steps=8`: Very small batch size with increased gradient accumulation to manage memory constraints.
- `dataloader_num_workers=0`: Disabled multi-processing for data loading to reduce memory usage.

## Configuration

### Model Selection

You can choose the model to use for training by setting the `TRAINING_MODEL` environment variable. If not set, the script defaults to `distilgpt2`, which is a highly memory-efficient model.
