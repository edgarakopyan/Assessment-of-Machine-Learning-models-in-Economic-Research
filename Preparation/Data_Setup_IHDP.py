import urllib.request
from pathlib import Path
import os


dir = os.getcwd()

# Download IHDP data

def download_ihdp(path: Path, url: str) -> None:

    path = Path(path)

    if path.exists():
        return

    urllib.request.urlretrieve(url, path)

    return

train_url = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
test_url = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"

download_ihdp(path = dir + "/ihdp_npci_1-100.train.npz", url = train_url)
download_ihdp(path = dir + "/ihdp_npci_1-100.test.npz", url = test_url)
