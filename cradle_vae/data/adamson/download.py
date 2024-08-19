import os

ADAMSON_URL = "https://zenodo.org/records/7041849/files/AdamsonWeissman2016_GSM2406681_10X010.h5ad?download=1"

def download_replogle_dataset(force: bool = False):
    home_dir = "/".join(os.getcwd().split("/")[:-3])
    cache_dir = f"{home_dir}/datasets"
    path = os.path.join(cache_dir, "adamson.h5ad")
    is_cached = os.path.exists(path)
    if not force and not is_cached:
        os.makedirs(cache_dir, exist_ok=True)
        os.system(f"wget -O {path} {ADAMSON_URL}")
    return path
