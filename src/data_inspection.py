from pathlib import Path

DATA_DIR = Path("data/extracted")

def list_dataset_files(data_dir: Path = DATA_DIR):
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir.resolve()}")
    return [p for p in data_dir.rglob("*") if p.is_file()]

if __name__ == "__main__":
    for file in list_dataset_files():
        print(file)
