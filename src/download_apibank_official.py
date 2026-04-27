import argparse
from pathlib import Path
from urllib.request import urlretrieve


OFFICIAL_REPO = "https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank"
OFFICIAL_DATASET = "https://huggingface.co/datasets/liminghao1630/API-Bank"
BASE_URL = "https://huggingface.co/datasets/liminghao1630/API-Bank/resolve/main"

DEFAULT_OUT_DIR = "data/apibank/raw"
FILES = [
    "training-data/lv1-api-train.json",
    "training-data/lv2-api-train.json",
    "test-data/level-1-api.json",
    "test-data/level-2-api.json",
]


def download_file(relative_path: str, out_dir: Path, overwrite: bool):
    url = f"{BASE_URL}/{relative_path}"
    target = out_dir / Path(relative_path).name
    if target.exists() and not overwrite:
        print(f"[skip] {target}")
        return

    print(f"[download] {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, target)
    print(f"[saved] {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"Official repository: {OFFICIAL_REPO}")
    print(f"Official dataset: {OFFICIAL_DATASET}")
    out_dir = Path(args.out_dir)
    for relative_path in FILES:
        download_file(relative_path, out_dir, args.overwrite)


if __name__ == "__main__":
    main()
