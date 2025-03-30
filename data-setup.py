import zipfile
import os
import sys

ZIP_FILE = "./march-machine-learning-mania-2025.zip"

def unzip_to_data(zip_path):
    if not os.path.isfile(zip_path):
        print(f"Error: File '{zip_path}' does not exist.")
        return

    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        print(f"Extracted '{zip_path}' to '{output_dir}'")

if __name__ == "__main__":
    zip_file = sys.argv[1] if len(sys.argv) > 1 else ZIP_FILE
    unzip_to_data(zip_file)
