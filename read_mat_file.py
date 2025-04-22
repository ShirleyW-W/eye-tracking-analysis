"""
load_all_fixation_data.py

This script loads multiple .mat fixation files (e.g., subj5, subj6, ...) from a directory,
extracts fixation information for each participant, cleans it, and compiles everything into
a single CSV file (`fixations_all_cleaned.csv`) for downstream analysis.
"""

import scipy.io
import pandas as pd
import numpy as np
import os
import re

# Set the directory with all the .mat files
data_dir = "/Users/shirleyw/Documents/Code/MetaLab_code/eye_tracking/eye_individual with maps"
output_path = "fixations_all_cleaned.csv"

def unwrap_fixation(fixation_struct):
    img = fixation_struct['img'][0, 0][0]
    subjects_field = fixation_struct['subjects'][0, 0]
    subject_data = subjects_field[0, 0]
    subject_struct = subject_data[0]

    subj_id = subject_struct['subj'][0][0] if subject_struct['subj'].ndim > 1 else subject_struct['subj'][0]
    fix_x = subject_struct['fix_x'].flatten()
    fix_y = subject_struct['fix_y'].flatten()
    durations = subject_struct['fix_duration'].flatten()

    return {
        'subject_id': subj_id,
        'image': img,
        'fix_x': fix_x,
        'fix_y': fix_y,
        'durations': durations
    }

def flatten_fixation_dict(fx):
    image = fx['image']
    subject_id = int(re.search(r'\d+', str(fx['subject_id'])).group())
    fix_x = fx['fix_x']
    fix_y = fx['fix_y']
    durations = fx['durations']

    flat_rows = []
    for x, y, d in zip(fix_x, fix_y, durations):
        flat_rows.append({
            'subject_id': subject_id,
            'image': image,
            'fix_x': x,
            'fix_y': y,
            'duration': d
        })
    return flat_rows

# Main script
all_cleaned_rows = []

for filename in os.listdir(data_dir):
    if filename.startswith("fixations_subj") and filename.endswith(".mat"):
        file_path = os.path.join(data_dir, filename)
        print(f"Processing {filename}...")

        data = scipy.io.loadmat(file_path)
        fixations = data['fixations']

        for i in range(len(fixations)):
            fx = unwrap_fixation(fixations[i, 0])
            flat = flatten_fixation_dict(fx)
            all_cleaned_rows.extend(flat)

# Convert to single DataFrame
df_all = pd.DataFrame(all_cleaned_rows)
df_all.to_csv(output_path, index=False)
print(f"Saved all cleaned fixation data to: {output_path}")
