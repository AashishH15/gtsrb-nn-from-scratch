"""GTSRB subset loading and preprocessing (NumPy / Pillow only)."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# Five visually distinct, well-represented GTSRB classes
CLASS_IDS = (0, 1, 14, 33, 38)
CLASS_NAMES = {
    0: "Speed limit 20",
    1: "Speed limit 30",
    14: "Stop",
    33: "Turn right ahead",
    38: "Keep right",
}

IMAGE_SIZE = 32
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
TRAINING_URL = (
    "https://sid.erda.dk/public/archives/"
    "daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
)


def _training_root(data_dir: Path) -> Path:
    """Locate Final_Training/Images under data_dir (handles common layouts)."""
    candidates = [
        data_dir / "GTSRB" / "Final_Training" / "Images",
        data_dir / "Final_Training" / "Images",
        data_dir / "Images",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find GTSRB training images under {data_dir}. "
        "Expected .../GTSRB/Final_Training/Images/. "
        "Run download_gtsrb() or place the dataset there manually."
    )


def download_gtsrb(data_dir: Path | None = None, force: bool = False) -> Path:
    """Download and extract GTSRB Final Training Images if missing."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        root = _training_root(data_dir)
        if not force:
            return root
    except FileNotFoundError:
        pass

    zip_path = data_dir / "GTSRB_Final_Training_Images.zip"
    print(f"Downloading GTSRB training set to {zip_path} ...")
    urllib.request.urlretrieve(TRAINING_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    zip_path.unlink(missing_ok=True)
    return _training_root(data_dir)


def _load_class_images(class_dir: Path, image_size: int) -> np.ndarray:
    """Load all .ppm images in a class folder, resize, return float32 [0, 1]."""
    paths = sorted(class_dir.glob("*.ppm"))
    if not paths:
        raise FileNotFoundError(f"No .ppm images in {class_dir}")

    images = []
    for path in paths:
        with Image.open(path) as img:
            rgb = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            images.append(np.asarray(rgb, dtype=np.float32) / 255.0)
    return np.stack(images, axis=0)


def load_subset(
    data_dir: Path | None = None,
    class_ids: tuple[int, ...] = CLASS_IDS,
    image_size: int = IMAGE_SIZE,
    download: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load a GTSRB subset as float32 images in [0, 1].

    Returns
    -------
    X : (N, H, W, 3) float32
    y : (N,) int64 labels remapped to 0 .. K-1
    label_names : list of length K
    """
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if download:
        root = download_gtsrb(data_dir)
    else:
        root = _training_root(data_dir)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    label_names: list[str] = []

    for new_label, class_id in enumerate(class_ids):
        class_dir = root / f"{class_id:05d}"
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        imgs = _load_class_images(class_dir, image_size)
        xs.append(imgs)
        ys.append(np.full(len(imgs), new_label, dtype=np.int64))
        label_names.append(CLASS_NAMES.get(class_id, f"Class {class_id}"))

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y, label_names


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified shuffle split into train and validation sets."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for label in np.unique(y):
        idxs = np.where(y == label)[0]
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    tr = np.array(train_idx, dtype=np.int64)
    va = np.array(val_idx, dtype=np.int64)
    return X[tr], y[tr], X[va], y[va]


def summarize_split(
    y_train: np.ndarray,
    y_val: np.ndarray,
    label_names: list[str],
) -> str:
    """Human-readable class counts for train and val."""
    lines = ["Dataset summary (5-class GTSRB subset)", "-" * 44]
    lines.append(f"{'Class':<22} {'Train':>7} {'Val':>7}")
    for i, name in enumerate(label_names):
        n_tr = int(np.sum(y_train == i))
        n_va = int(np.sum(y_val == i))
        lines.append(f"{name:<22} {n_tr:>7} {n_va:>7}")
    lines.append("-" * 44)
    lines.append(f"{'Total':<22} {len(y_train):>7} {len(y_val):>7}")
    lines.append(
        f"Images: 32x32 RGB, float32 normalized to [0, 1], "
        f"shape train={y_train.shape[0]}, val={y_val.shape[0]}"
    )
    return "\n".join(lines)


def load_prepared(
    data_dir: Path | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    download: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Convenience: load subset and return stratified train/val split."""
    X, y, names = load_subset(data_dir=data_dir, download=download)
    X_train, y_train, X_val, y_val = train_val_split(
        X, y, val_ratio=val_ratio, seed=seed
    )
    return X_train, y_train, X_val, y_val, names
