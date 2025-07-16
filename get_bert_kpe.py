#!/usr/bin/env python3
"""
Clone the BERT-KPE repository and download the checkpoint from Google Drive,
then extract the checkpoint into the cloned repository directory.
"""

import os
import subprocess

try:
    import gdown
except ImportError:
    print("gdown is not installed. Install it via 'pip install gdown' and try again.")
    exit(1)

import zipfile


def main():
    # Repository parameters
    repo_url = "https://github.com/thunlp/BERT-KPE.git"
    repo_dir = "BERT-KPE"

    # Clone the repository if it doesn't exist
    if not os.path.isdir(repo_dir):
        print(f"Cloning repository {repo_url} into {repo_dir}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    else:
        print(f"Repository directory '{repo_dir}' already exists. Skipping clone.")

    # Google Drive file ID for the checkpoint archive
    file_id = "13FvONBTM4NZZCR-I7LVypkFa0xihxWnM"
    url = f"https://drive.google.com/uc?id={file_id}"
    output_zip = os.path.join(repo_dir, "checkpoint.zip")

    # Download the checkpoint archive
    if not os.path.isfile(output_zip):
        print(f"Downloading checkpoint archive to '{output_zip}'...")
        gdown.download(url, output_zip, quiet=False)
    else:
        print(f"Checkpoint archive '{output_zip}' already exists. Skipping download.")

    # Extract the archive into the repository directory
    print(f"Extracting '{output_zip}' into '{repo_dir}'...")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(repo_dir)

    print("Done.")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:09:05 2025

@author: niran
"""

