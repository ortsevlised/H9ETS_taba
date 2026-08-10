import csv
import random
import re
import zipfile
from functools import lru_cache
from pathlib import Path

SEED = 42
DATASET_DIR = Path(__file__).resolve().parent.parent / "dffd_dataset"
LISTS_DIR = DATASET_DIR / "_lists"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "manifests"

# Source registry: prefix used in the official list files -> (source name, zip file, archive top dir)
FAKE_PREFIX_MAP = {
    "F_FAP0": ("faceapp", "faceapp.zip", "faceapp"),
    "F_FAP1": ("faceapp", "faceapp.zip", "faceapp"),
    "F_PGN1": ("pggan_v1", "pggan_v1.zip", "pggan_v1"),
    "F_PGN2": ("pggan_v2", "pggan_v2.zip", "pggan_v2"),
    "F_STGN": ("stargan", "stargan.zip", "stargan"),
    "F_SyCA": ("stylegan_celeba", "stylegan_celeba.zip", "stylegan_celeba"),
    "F_SyFQ": ("stylegan_ffhq", "stylegan_ffhq.zip", "stylegan_ffhq"),
}
REAL_PREFIX = "R_FFHQ"
REAL_SOURCE = ("ffhq", "ffhq.zip", "ffhq")

# Fake sources usable in each split. FaceApp is train-only: its
# validation/test identities (0-2257) all map back to FFHQ photos numbered
# 0-9999, which is entirely FFHQ's train range, so using FaceApp fakes in
# validation/test would leak identity against the FFHQ real class.
FAKE_SOURCES_BY_SPLIT = {
    "train": ["faceapp", "pggan_v1", "pggan_v2", "stargan", "stylegan_celeba", "stylegan_ffhq"],
    "validation": ["pggan_v1", "pggan_v2", "stargan", "stylegan_celeba", "stylegan_ffhq"],
    "test": ["pggan_v1", "pggan_v2", "stargan", "stylegan_celeba", "stylegan_ffhq"],
}

# Held-out source for the post-hoc generalisation check.
GENERALISATION_HOLDOUT_SOURCE = "stylegan_ffhq"

TARGET_REAL = {"train": 9000, "validation": 900, "test": 4500}
TARGET_FAKE_TOTAL = {"train": 9000, "validation": 900, "test": 4500}


SOURCE_ARCHIVE = {"ffhq": ("ffhq.zip", "ffhq")}
for _prefix, (_name, _zip, _dir) in FAKE_PREFIX_MAP.items():
    SOURCE_ARCHIVE[_name] = (_zip, _dir)


@lru_cache(maxsize=None)
def zip_namelist_set(zip_file):
    with zipfile.ZipFile(DATASET_DIR / zip_file) as z:
        return frozenset(z.namelist())


def filter_available(filenames, source_name, split):
    """The official DFFD list files reference some images that were not
    included in these redistributed zips (confirmed for StarGAN test, where
    ~14,000 of 35,960 listed files are absent). Filter every bucket against
    the zip's actual contents before sampling so extraction never fails."""
    zip_file, archive_dir = SOURCE_ARCHIVE[source_name]
    names = zip_namelist_set(zip_file)
    return [fn for fn in filenames if f"{archive_dir}/{split}/{fn}" in names]


def load_list(name):
    path = LISTS_DIR / name
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def bucket_real(filenames, split):
    ids = [fn for fn in filenames if fn.startswith(REAL_PREFIX)]
    return filter_available(ids, "ffhq", split)


def bucket_fake_by_source(filenames, split):
    buckets = {}
    for fn in filenames:
        prefix = fn[:6]
        mapping = FAKE_PREFIX_MAP.get(prefix)
        if mapping is None:
            continue
        source_name = mapping[0]
        buckets.setdefault(source_name, []).append(fn)
    return {name: filter_available(fns, name, split) for name, fns in buckets.items()}


def sample(rng, items, k, label):
    if k > len(items):
        raise ValueError(f"{label}: requested {k} but only {len(items)} available")
    return rng.sample(sorted(items), k)


def _build_manifest_rows(rng, splits, excluded_sources=frozenset(), label_prefix=""):
    """Shared sampling loop for both the main and holdout manifests: sample
    real images then fake images (evenly split across the allowed sources for
    that split, minus any excluded_sources) for each split in order. Pulled
    out of build_main_manifest/build_holdout_train_val_manifest, which used to
    duplicate this ~35-line loop; callers must keep calling these in the same
    order they always have; changing that order changes what the shared `rng`
    draws next and would desync the already-extracted, already-trained-on
    manifests."""
    rows = []
    for split in splits:
        real_files = bucket_real(load_list(f"{split}_real.txt"), split)
        real_sample = sample(rng, real_files, TARGET_REAL[split], f"{label_prefix}real/{split}")
        for fn in real_sample:
            rows.append({
                "filename": fn,
                "source": REAL_SOURCE[0],
                "split": split,
                "label": 0,
                "zip_file": REAL_SOURCE[1],
                "archive_path": f"{REAL_SOURCE[2]}/{split}/{fn}",
            })

        fake_files = load_list(f"{split}_fake.txt")
        fake_buckets = bucket_fake_by_source(fake_files, split)
        allowed_sources = [s for s in FAKE_SOURCES_BY_SPLIT[split] if s not in excluded_sources]
        per_source = TARGET_FAKE_TOTAL[split] // len(allowed_sources)
        remainder = TARGET_FAKE_TOTAL[split] - per_source * len(allowed_sources)
        for i, source_name in enumerate(allowed_sources):
            k = per_source + (1 if i < remainder else 0)
            available = fake_buckets.get(source_name, [])
            fake_sample = sample(rng, available, k, f"{label_prefix}fake/{split}/{source_name}")
            zip_file = f"{source_name}.zip"
            for fn in fake_sample:
                rows.append({
                    "filename": fn,
                    "source": source_name,
                    "split": split,
                    "label": 1,
                    "zip_file": zip_file,
                    "archive_path": f"{source_name}/{split}/{fn}",
                })
    return rows


def build_main_manifest(rng):
    return _build_manifest_rows(rng, ["train", "validation", "test"])


def build_holdout_train_val_manifest(rng):
    """Re-sampled train/validation manifests with GENERALISATION_HOLDOUT_SOURCE
    excluded entirely, for the models used in the generalisation check. Freed
    quota is redistributed evenly across the remaining allowed sources."""
    return _build_manifest_rows(
        rng, ["train", "validation"],
        excluded_sources={GENERALISATION_HOLDOUT_SOURCE},
        label_prefix="holdout-",
    )


def build_generalisation_manifest(rng, main_rows):
    """Held-out check: test set fakes from GENERALISATION_HOLDOUT_SOURCE that
    were NOT used in the main test split, plus an equal number of real FFHQ
    test images not used in the main manifest."""
    used_fake_filenames = {
        r["filename"] for r in main_rows
        if r["source"] == GENERALISATION_HOLDOUT_SOURCE
    }
    used_real_filenames = {
        r["filename"] for r in main_rows if r["source"] == REAL_SOURCE[0]
    }

    test_fake_files = load_list("test_fake.txt")
    fake_buckets = bucket_fake_by_source(test_fake_files, "test")
    holdout_pool = [
        fn for fn in fake_buckets.get(GENERALISATION_HOLDOUT_SOURCE, [])
        if fn not in used_fake_filenames
    ]

    test_real_files = bucket_real(load_list("test_real.txt"), "test")
    real_pool = [fn for fn in test_real_files if fn not in used_real_filenames]

    n = min(len(holdout_pool), len(real_pool), 1000)
    fake_sample = sample(rng, holdout_pool, n, "generalisation/fake")
    real_sample = sample(rng, real_pool, n, "generalisation/real")

    rows = []
    for fn in real_sample:
        rows.append({
            "filename": fn,
            "source": REAL_SOURCE[0],
            "split": "generalisation",
            "label": 0,
            "zip_file": REAL_SOURCE[1],
            "archive_path": f"{REAL_SOURCE[2]}/test/{fn}",
        })
    for fn in fake_sample:
        rows.append({
            "filename": fn,
            "source": GENERALISATION_HOLDOUT_SOURCE,
            "split": "generalisation",
            "label": 1,
            "zip_file": f"{GENERALISATION_HOLDOUT_SOURCE}.zip",
            "archive_path": f"{GENERALISATION_HOLDOUT_SOURCE}/test/{fn}",
        })
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "source", "split", "label", "zip_file", "archive_path"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows, title):
    print(f"--- {title} ---")
    print("total rows", len(rows))
    from collections import Counter
    by_split_label = Counter((r["split"], r["label"]) for r in rows)
    for key in sorted(by_split_label):
        print(" split/label", key, "->", by_split_label[key])
    by_split_source = Counter((r["split"], r["source"]) for r in rows)
    for key in sorted(by_split_source):
        print(" split/source", key, "->", by_split_source[key])


def main():
    rng = random.Random(SEED)
    main_rows = build_main_manifest(rng)
    write_csv(main_rows, OUT_DIR / "main_manifest.csv")
    summarise(main_rows, "main manifest")

    gen_rows = build_generalisation_manifest(rng, main_rows)
    write_csv(gen_rows, OUT_DIR / "generalisation_manifest.csv")
    summarise(gen_rows, "generalisation manifest")

    holdout_rows = build_holdout_train_val_manifest(rng)
    write_csv(holdout_rows, OUT_DIR / "holdout_train_val_manifest.csv")
    summarise(holdout_rows, "holdout train/validation manifest (excludes stylegan_ffhq)")

    # sanity: no filename appears under two different labels
    seen = {}
    for r in main_rows + gen_rows + holdout_rows:
        key = (r["source"], r["filename"])
        if key in seen and seen[key] != r["label"]:
            raise AssertionError(f"label conflict for {key}")
        seen[key] = r["label"]
    print("sanity check passed: no label conflicts")


if __name__ == "__main__":
    main()
