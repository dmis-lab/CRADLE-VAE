import os

DIXIT_URL = "https://ndownloader.figshare.com/files/34011689"


def download_dixit_dataset(force: bool = False):
    cache_dir = os.environ.get("SAMS_VAE_DATASET_DIR", "datasets")
    path = os.path.join(cache_dir, "dixit.h5ad")
    is_cached = os.path.exists(path)
    if not force and not is_cached:
        os.makedirs(cache_dir, exist_ok=True)
        os.system(f"wget -O {path} {DIXIT_URL}")
    return path
