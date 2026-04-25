import argparse
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


FILE_ID = "1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk"
OFFICIAL_REPO = "https://github.com/OpenBMB/ToolBench"
OFFICIAL_BROWSER_URL = f"https://drive.google.com/uc?id={FILE_ID}"
OFFICIAL_TSINGHUA_URL = "https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/"
OFFICIAL_URL = (
    "https://drive.google.com/uc?export=download&id="
    f"{FILE_ID}&confirm=yes"
)
DEFAULT_OUTPUT_DIR = "outputs/toolbench_raw"
DEFAULT_ZIP_PATH = "outputs/toolbench_raw/data.zip"
REQUIRED_MEMBERS = [
    "data/toolllama_G123_dfs_train.json",
    "data/toolllama_G123_dfs_eval.json",
]


def official_download_error(exc: Exception, zip_path: str) -> RuntimeError:
    return RuntimeError(
        "Official ToolBench download failed.\n"
        f"Error: {exc}\n\n"
        "Do not use mirrored datasets for this run. Use one of the official "
        "OpenBMB links instead:\n"
        f"- Repository: {OFFICIAL_REPO}\n"
        f"- Google Drive browser URL: {OFFICIAL_BROWSER_URL}\n"
        f"- Tsinghua Cloud: {OFFICIAL_TSINGHUA_URL}\n\n"
        "After downloading the official data.zip, place it at:\n"
        f"  {zip_path}\n"
        "Then resume with:\n"
        "  uv run python src/download_toolbench_official.py --skip-download\n"
        "  uv run python src/prepare_toolbench.py"
    )


def download_with_gdown(file_id: str, zip_path: str):
    cmd = [
        sys.executable,
        "-m",
        "gdown",
        file_id,
        "-O",
        zip_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise official_download_error(exc, zip_path) from exc


def download(url: str, zip_path: str, method: str):
    target = Path(zip_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print(f"Zip already exists: {target}")
        return

    if method == "gdown":
        download_with_gdown(FILE_ID, zip_path)
    else:
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:
            raise official_download_error(exc, zip_path) from exc

    print(f"Downloaded official ToolBench data to: {target}")


def unzip(zip_path: str, output_dir: str, extract_all: bool):
    target = Path(zip_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(target, "r") as zf:
        members = None if extract_all else REQUIRED_MEMBERS
        missing = [name for name in REQUIRED_MEMBERS if name not in zf.namelist()]
        if missing:
            raise FileNotFoundError(
                "Official ToolBench zip is missing required files:\n"
                + "\n".join(f"- {name}" for name in missing)
            )
        zf.extractall(out, members=members)

    if extract_all:
        print(f"Extracted all files from {target} to: {out}")
    else:
        print(f"Extracted required ToolBench files from {target} to: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=OFFICIAL_URL)
    parser.add_argument("--zip-path", type=str, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method", type=str, choices=["gdown", "urllib"], default="gdown")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-unzip", action="store_true")
    parser.add_argument(
        "--extract-all",
        action="store_true",
        help="Extract the full official archive instead of only train/eval JSON files.",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download(args.url, args.zip_path, args.method)

    if not args.skip_unzip:
        unzip(args.zip_path, args.output_dir, extract_all=args.extract_all)


if __name__ == "__main__":
    main()
