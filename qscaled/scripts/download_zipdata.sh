#!/bin/bash

TARGET_DIR="$HOME/.qscaled/zip/"
mkdir -p "$TARGET_DIR"
BASE_URL="https://prestonfu.com/assets/data/qscaled/"

FILES=(
    "dmc_baseline.zip"
    "dmc_ours.zip"
    "dmc_sweep.zip"
    "gym_baseline_utd2.zip"
    "gym_ours.zip"
    "gym_sweep.zip"
)

for FILE in "${FILES[@]}"; do
    DEST="$TARGET_DIR$FILE"
    URL="$BASE_URL$FILE"
    
    # Download only if the file does not already exist
    if [ ! -f "$DEST" ]; then
        echo "Downloading $FILE..."
         wget -q --show-progress -O "$DEST" "$URL"
    else
        echo "$FILE already exists, skipping download."
    fi
done

echo "All files downloaded to $TARGET_DIR."
