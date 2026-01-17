import os
import pathlib
from typing import Optional
from huggingface_hub import snapshot_download
from .codec import decode_dir


class DataLoader:
    """
    Handles downloading and decoding the CAME-Bench dataset.
    """

    REPO_ID = "Seattleyrz/CAME-Bench"
    DEFAULT_DATA_DIR = "came_bench_data"

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = pathlib.Path(data_dir or self.DEFAULT_DATA_DIR).resolve()
        self.encoded_dir = self.data_dir / "encoded"
        self.decoded_dir = self.data_dir / "decoded"

    def download(self, token: Optional[str] = None):
        """
        Downloads the encoded dataset from Hugging Face.
        """
        print(f"Downloading CAME-Bench from {self.REPO_ID} to {self.encoded_dir}...")
        snapshot_download(
            repo_id=self.REPO_ID,
            repo_type="dataset",
            local_dir=self.encoded_dir,
            allow_patterns=["encoded_benchmark_codec/**", "codec.py", "verify_codec.py", "README.md"],
            token=token
        )
        print("Download complete.")

    def decode(self):
        """
        Decodes the downloaded dataset.
        """
        encoded_data_path = self.encoded_dir / "encoded_benchmark_codec"
        if not encoded_data_path.exists():
            # Fallback if the folder structure is flat or different
            encoded_data_path = self.encoded_dir

        # Check if metadata.json exists in encoded_data_path
        if not (encoded_data_path / "metadata.json").exists():
            # Try to find where encoded_benchmark_codec is
            candidates = list(self.encoded_dir.rglob("metadata.json"))
            if candidates:
                encoded_data_path = candidates[0].parent
            else:
                raise FileNotFoundError(f"Could not find metadata.json in {self.encoded_dir}")

        print(f"Decoding data from {encoded_data_path} to {self.decoded_dir}...")
        decode_dir(encoded_data_path, self.decoded_dir, strict=True)
        print("Decode complete.")

    def load(self, force_download: bool = False, force_decode: bool = False):
        """
        Ensures data is available (downloading and decoding if necessary).
        """
        if force_download or not self.encoded_dir.exists() or not any(self.encoded_dir.iterdir()):
            self.download()

        if force_decode or not self.decoded_dir.exists() or not any(self.decoded_dir.iterdir()):
            self.decode()

        return self.decoded_dir


def load_dataset(data_dir: Optional[str] = None) -> pathlib.Path:
    """
    Convenience function to setup and return the path to the decoded dataset.
    """
    loader = DataLoader(data_dir)
    return loader.load()
