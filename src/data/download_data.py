import kagglehub
from pathlib import Path
import shutil

DATASET = "kazanova/sentiment140"
RAW_DIR = Path("data/raw")

def main():

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download Path
    ds_path = Path(kagglehub.dataset_download("kazanova/sentiment140"))
    print("Downloaded dataset to:", ds_path)

    # Copy files into data/raw
    for file in ds_path.iterdir():
        if file.is_file():
            destination = RAW_DIR / file.name
            shutil.copy(file, destination)
            print(f"Copied {file.name} to {destination}")

    print("Copy to data/raw complete.")
    
if __name__ == "__main__":
    main()