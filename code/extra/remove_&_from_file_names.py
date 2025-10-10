import os

def clean_names(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        # Rename files
        for filename in filenames:
            if "&" in filename:
                new_filename = filename.replace("&", "")
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed file: {old_path} -> {new_path}")

        # Rename folders
        for dirname in dirnames:
            if "&" in dirname:
                new_dirname = dirname.replace("&", "")
                old_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, new_dirname)
                os.rename(old_path, new_path)
                print(f"Renamed folder: {old_path} -> {new_path}")

if __name__ == "__main__":
    root = "data"  # <-- change this
    clean_names(root)
    root = "results"
    clean_names(root)
