import os
import pandas as pd

# Load mapping from CSV
mapping = pd.read_csv("data/hari_BC/csv/BnW_combined.csv")  # columns: PID, New_PID
pid_to_new = dict(zip(mapping["PID"], mapping["New_PID"]))

# Root folder to start search
root_folder = "results"

# Keep track of unmatched PIDs (first 4 chars of filename)
unmatched_pids = set()

# Walk through all files recursively
for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        old_path = os.path.join(dirpath, filename)
        new_filename = filename
        matched = False

        # Replace PID in filename with New_PID (if found)
        for pid, new_pid in pid_to_new.items():
            if pid in new_filename:
                new_filename = new_filename.replace(pid, new_pid)
                matched = True
                break  # stop after first match

        if matched:
            # If filename changed, rename (disabled for safety now)
            if new_filename != filename:
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} → {new_path}")
        else:
            # Collect only the first 4 characters of the filename
            unmatched_pids.add(filename[:4])

# Save unmatched PIDs into a CSV
if unmatched_pids:
    unmatched_df = pd.DataFrame(sorted(unmatched_pids), columns=["Unmatched_PID"])
    unmatched_df.to_csv("unmatched_pid.csv", index=False)
    print(f"\nUnmatched PIDs saved to unmatched_pid.csv")
else:
    print("All files matched successfully.")