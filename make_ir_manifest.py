import argparse
from pathlib import Path

def build_manifest(parent_folder, manifest_filename):
    """
    Recursively finds all *_ir_images.txt files in the parent_folder
    and writes their absolute paths to the manifest file.
    """
    # Convert string paths to Path objects
    parent_path = Path(parent_folder)
    manifest_path = Path(manifest_filename)

    # Ensure the parent folder actually exists
    if not parent_path.exists() or not parent_path.is_dir():
        print(f"Error: The directory '{parent_folder}' does not exist.")
        return

    count = 0
    # Open the manifest file for writing
    with manifest_path.open('w', encoding='utf-8') as manifest_file:
        # rglob performs a recursive search through all subfolders
        for file_path in parent_path.rglob('*_ir_images.txt'):
            # .resolve() gets the strict absolute path
            absolute_path = file_path.resolve()
            
            # Write the path to the file, followed by a newline
            manifest_file.write(f"{absolute_path}\n")
            count += 1

    print(f"Success! Manifest built at: {manifest_path.resolve()}")
    print(f"Total files found: {count}")

if __name__ == "__main__":
    # Set up argument parsing for command-line usage
    parser = argparse.ArgumentParser(description="Build a manifest of *_ir_images.txt files.")
    parser.add_argument(
        "-p", "--parent", 
        type=str, 
        default=".", 
        help="Path to the parent directory to search (defaults to current directory)"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="ir_manifest.txt", 
        help="Name or path of the output manifest file (defaults to manifest.txt)"
    )

    args = parser.parse_args()
    
    build_manifest(args.parent, args.output)