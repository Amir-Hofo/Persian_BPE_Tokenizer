import os
import requests
from zipfile import ZipFile

url= "https://huggingface.co/datasets/RohanAiLab/persian_blog/resolve/main/blogs.zip"
output_path= "blogs.zip"
extract_folder= "blogs"

if not os.path.exists(output_path):
    print("Downloading blogs.zip...")
    r= requests.get(url, stream= True)
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download completed.")

if not os.path.exists(extract_folder):
    print("Extracting blogs.zip...")
    with ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print("Extraction completed.")
