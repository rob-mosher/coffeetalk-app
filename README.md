# coffeetalk-app

## Setup and Usage Instructions

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
