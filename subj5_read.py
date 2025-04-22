"""
This script loads a .mat file containing fixation data for a subject (subj5) to check the file structure and extract relevant information.
"""

import scipy.io
import pandas as pd
import numpy as np
from PIL import Image
import ast
from scipy.io import loadmat
import re

file_path = "/Users/shirleyw/Documents/Code/MetaLab_code/eye_tracking/eye_individual with maps/fixations_subj5.mat"
data = loadmat(file_path)

# Check the variable names
print(data.keys())

fixations = data['fixations']

def unwrap_fixation(fixation_struct):
    img = fixation_struct['img'][0, 0][0]  # This one’s stable

    # Dig into subjects field
    subjects_field = fixation_struct['subjects'][0, 0]  # shape (1, 1)
    subject_data = subjects_field[0, 0]  # pull out the nested object
    subject_struct = subject_data[0]     # should be a numpy void with named fields

    subj_id = subject_struct['subj'][0][0] if subject_struct['subj'].ndim > 1 else subject_struct['subj'][0]
    fix_x = subject_struct['fix_x'].flatten()
    fix_y = subject_struct['fix_y'].flatten()
    durations = subject_struct['fix_duration'].flatten()

    return {
        'image': img,
        'subject_id': subj_id,
        'fix_x': fix_x,
        'fix_y': fix_y,
        'durations': durations
    }


# View first 4 fixations
#for i in range(4):
    print(f"\n--- Fixation {i} ---")
    fx = unwrap_fixation(fixations[i, 0])
    print(f"Image: {fx['image']}")
    print(f"Subject ID: {fx['subject_id']}")
    print(f"X: {fx['fix_x']}")
    print(f"Y: {fx['fix_y']}")
    print(f"Durations: {fx['durations']}")


results = []
for i in range(len(fixations)):
    fx = unwrap_fixation(fixations[i, 0])
    results.append(fx)
# Convert to DataFrame
df = pd.DataFrame(results)
print(df.head())
# Save to CSV
df.to_csv('fixations_subj5_result.csv', index=False)




# Load the original CSV
df = pd.read_csv("fixations_subj5_result.csv")

# Function to extract the first number (e.g., subject ID)
def extract_subject_id(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else None

# Function to parse fixation arrays from strings
def parse_fixation_array(s):
    # Remove 'array(', 'dtype=...', and closing parens
    clean = re.sub(r'array\(|dtype=[^)]+\)|\)', '', s)
    # Remove all square brackets
    clean = clean.replace('[', '').replace(']', '')
    # Strip and convert
    nums = []
    for x in clean.split(','):
        x = x.strip()
        if x:  # skip empty strings
            try:
                nums.append(float(x))
            except ValueError:
                print(f"Warning: could not convert '{x}' to float")
    return np.array(nums)


# Clean each row and expand it
cleaned_rows = []

for _, row in df.iterrows():
    image = row['image']
    subject_id = extract_subject_id(row['subject_id'])

    fix_x = parse_fixation_array(row['fix_x'])
    fix_y = parse_fixation_array(row['fix_y'])
    durations = parse_fixation_array(row['durations'])

    for x, y, d in zip(fix_x, fix_y, durations):
        cleaned_rows.append({
            'image': image,
            'subject_id': subject_id,
            'fix_x': x,
            'fix_y': y,
            'duration': d
        })

# Create cleaned DataFrame
clean_df = pd.DataFrame(cleaned_rows)

# Save it to CSV (optional)
clean_df.to_csv("fixations_subj5_cleaned.csv", index=False)

# Show a preview
print(clean_df.head())
