import os

DIXIT_URL = "https://ndownloader.figshare.com/files/34011689"

def download_dixit_dataset(force: bool = False):
    home_dir = "/".join(os.getcwd().split("/")[:-3])
    cache_dir = f"{home_dir}/datasets"
    path = os.path.join(cache_dir, "dixit.h5ad")
    is_cached = os.path.exists(path)
    if not force and not is_cached:
        os.makedirs(cache_dir, exist_ok=True)
        os.system(f"wget -O {path} {DIXIT_URL}")
    return path
