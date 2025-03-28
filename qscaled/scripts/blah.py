import os
import shutil
from qscaled.constants import QSCALED_PATH

# Define source and destination base directories
src_base = os.path.expanduser("~/.qscaled/prezip/dmc_sweep_safe")
dst_base = os.path.expanduser("~/.qscaled/prezip/dmc_sweep")

# Ensure the destination directory exists
os.makedirs(dst_base, exist_ok=True)

# Iterate through the files in the source directory
for root, _, files in os.walk(src_base):
    for file in files:
        if file.endswith(".npy"):
            # Extract components from the filename
            parts = root.split("/")[-1].split("_")  # Extract BRO_256_0.5_3e-4
            if len(parts) != 4:
                continue  # Skip unexpected formats

            _, bs, utd, lr = parts  # Extract batch size, utd, learning rate
            env_name = file[:-4]  # Remove .npy extension
            
            # Define the new directory structure
            new_dir = os.path.join(dst_base, f"utd_{utd}", env_name, "online_returns")
            os.makedirs(new_dir, exist_ok=True)

            # Define new filename
            new_filename = f"bs_{bs}_lr_{lr}.npy"
            src_path = os.path.join(root, file)
            dst_path = os.path.join(new_dir, new_filename)

            # Copy the file
            shutil.copy2(src_path, dst_path)

print("Files copied successfully!")