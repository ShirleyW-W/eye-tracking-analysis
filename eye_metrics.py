"""
This script computes two gaze behavior metrics—fixation entropy and number of unique regions visited for each participant in an eye-tracking dataset.

The script outputs a CSV file (`eye_metrics_summary.csv`) summarizing the average entropy and average number of unique regions visited per participant.
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from scipy.stats import entropy

# Parameters
IMG_WIDTH, IMG_HEIGHT = 800, 600
GRID_ROWS, GRID_COLS = 4, 4

# --- Helper functions ---
def parse_fixation_array(s):
    """Convert a string like '[[123.4 567.8]]' to np.array([123.4, 567.8])"""
    clean = re.sub(r'array\(|dtype=[^)]+\)|[\[\]]', '', s)
    nums = [float(x) for x in clean.split() if x]
    return np.array(nums)

def compute_entropy(fix_map):
    """Compute entropy of a fixation map"""
    flat = fix_map.flatten()
    probs = flat / flat.sum() if flat.sum() > 0 else flat
    return entropy(probs, base=2) if probs.sum() > 0 else 0.0

def compute_unique_rois(fix_xs, fix_ys):
    """Given fixation coords, compute number of unique ROIs visited"""
    roi_set = set()
    for x, y in zip(fix_xs, fix_ys):
        if 0 <= x < IMG_WIDTH and 0 <= y < IMG_HEIGHT:
            col = int(x / IMG_WIDTH * GRID_COLS)
            row = int(y / IMG_HEIGHT * GRID_ROWS)
            roi_set.add((row, col))
    return len(roi_set)

# --- Load and parse data ---
df = pd.read_csv("fixations_all_cleaned.csv")

results = defaultdict(lambda: {'entropies': [], 'roi_counts': []})

# --- Process each row/image ---
for _, row in df.iterrows():
    subj_id = int(row['subject_id'])
    fix_x = parse_fixation_array(row['fix_x'])
    fix_y = parse_fixation_array(row['fix_y'])

    # Skip images with no fixations
    if len(fix_x) == 0 or len(fix_y) == 0:
        continue

    # Build fixation map
    fix_map = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint16)
    for x, y in zip(fix_x, fix_y):
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < IMG_WIDTH and 0 <= iy < IMG_HEIGHT:
            fix_map[iy, ix] += 1

    # Compute metrics
    ent = compute_entropy(fix_map)
    unique_rois = compute_unique_rois(fix_x, fix_y)

    results[subj_id]['entropies'].append(ent)
    results[subj_id]['roi_counts'].append(unique_rois)

# --- Summarize results ---
summary = []
for subj, metrics in results.items():
    avg_entropy = np.mean(metrics['entropies']) if metrics['entropies'] else 0
    avg_rois = np.mean(metrics['roi_counts']) if metrics['roi_counts'] else 0
    summary.append({
        'subject_id': subj,
        'avg_entropy': round(avg_entropy, 3),
        'avg_unique_regions': round(avg_rois, 3)
    })

# --- Save or print ---
summary_df = pd.DataFrame(summary)
summary_df.to_csv("participant_summary.csv", index=False)
print(summary_df)
