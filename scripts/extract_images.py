import csv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dffd_dataset"
MANIFEST_DIR = ROOT / "data" / "manifests"
IMAGES_DIR = ROOT / "data" / "images"


def collect_targets():
    targets = {}  # zip_file -> set of archive_path
    for manifest_path in sorted(MANIFEST_DIR.glob("*.csv")):
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                targets.setdefault(row["zip_file"], set()).add(row["archive_path"])
    return targets


def extract_from_zip(zip_name, archive_paths):
    zip_path = DATASET_DIR / zip_name
    to_extract = []
    for archive_path in archive_paths:
        dest = IMAGES_DIR / archive_path
        if not dest.exists():
            to_extract.append(archive_path)

    if not to_extract:
        print(f"{zip_name}: all {len(archive_paths)} files already extracted")
        return 0

    print(f"{zip_name}: extracting {len(to_extract)} of {len(archive_paths)} files")
    with zipfile.ZipFile(zip_path) as z:
        for archive_path in to_extract:
            dest = IMAGES_DIR / archive_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(archive_path) as src, open(dest, "wb") as out:
                out.write(src.read())
    return len(to_extract)


def main():
    targets = collect_targets()
    total = sum(len(v) for v in targets.values())
    print(f"{len(targets)} zip files, {total} unique images referenced across all manifests")
    extracted = 0
    for zip_name, archive_paths in sorted(targets.items()):
        extracted += extract_from_zip(zip_name, archive_paths)
    print(f"done, extracted {extracted} new files")


if __name__ == "__main__":
    main()
