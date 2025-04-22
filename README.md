This repository contains Python code and data for analyzing fixation patterns from eye-tracking experiments. It focuses on understanding where participants look when viewing images by computing:

Fixation Entropy — to measure gaze dispersion

Unique Regions Visited (ROIs) — to estimate exploration coverage via grid analysis

**read_mat_file.py** reads raw .mat files (from MATLAB) and cleans + organizes fixation data into a structured CSV format.
Each row contains:
    subject_id
    image (filename)
    fix_x, fix_y (fixation coordinates)
    duration (fixation time)
Output: fixations_all_cleaned.csv

**eye_metrics.py** performs all computations on the cleaned dataset, including: Creating fixation maps (800×600 resolution), Computing entropy of fixation distributions per image, Dividing the image into a spatial grid (e.g., 6×6) to count unique ROIs visited, Averaging both metrics per participant
