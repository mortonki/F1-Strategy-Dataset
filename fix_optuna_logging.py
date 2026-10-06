import sys

file_path = "/home/mordicus/Documents/Projektit/Data Science/Kaggle/F1 Strategy Dataset/src/f1_strategy_dataset/hyperparameter_search_optuna.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if "mlflow.log_params({" in line and i >= 300: # Approximate position
        new_lines.append("    with mlflow.start_run(nested=True):\n")
        new_lines.append("        " + line.replace("    ", "").lstrip().replace("mlflow.log_params({", "mlflow.log_params({").rstrip() + "\n") # This is getting complicated.
        # Let's just do it more simply.
        pass
    i += 1

# Actually, let's just rewrite the whole block from 305 to 336.
# I'll use the content I saw earlier.
