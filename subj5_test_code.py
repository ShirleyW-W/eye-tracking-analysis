import pandas as pd
import numpy as np
from scipy.stats import entropy

# Load cleaned fixation data
df = pd.read_csv("fixations_subj5_cleaned.csv")

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
GRID_BINS = (10, 10)  # You can tweak this

def fixation_matrix(fix_x, fix_y, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    matrix = np.zeros((width, height))
    for x, y in zip(fix_x, fix_y):
        xi, yi = int(x), int(y)
        if 0 <= xi < width and 0 <= yi < height:
            matrix[xi, yi] = 1
    return matrix

def compute_entropy_from_matrix(matrix):
    flat = matrix.flatten()
    prob = flat / flat.sum() if flat.sum() > 0 else flat
    prob = prob[prob > 0]
    return entropy(prob)

def compute_unique_regions(fix_x, fix_y, bins=GRID_BINS, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    heatmap, _, _ = np.histogram2d(fix_x, fix_y, bins=bins, range=[[0, width], [0, height]])
    return np.count_nonzero(heatmap)

# Compute per-image stats
entropy_scores = []
region_counts = []

for image, group in df.groupby("image"):
    fix_x = group['fix_x'].values
    fix_y = group['fix_y'].values

    fix_map = fixation_matrix(fix_x, fix_y)
    entropy_val = compute_entropy_from_matrix(fix_map)
    unique_regions = compute_unique_regions(fix_x, fix_y)

    entropy_scores.append(entropy_val)
    region_counts.append(unique_regions)

# Final summary for this participant
participant_entropy = np.mean(entropy_scores)
participant_regions = np.mean(region_counts)

print(f"Participant {df['subject_id'].iloc[0]} — Avg Entropy: {participant_entropy:.3f}, Avg Unique Regions Visited: {participant_regions:.1f}")
