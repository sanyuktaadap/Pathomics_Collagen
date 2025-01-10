"""
File: Main file to plot collagen fibers at patch level
"""

# header files to load
from compute_bifs import compute_bifs as compute_bifs
import numpy as np
import cv2
from skimage import morphology
from PIL import Image
import os
import glob
import argparse
from tqdm import tqdm

# MAIN CODE
filter_scale = 3
feature_descriptor = 5
orient_num = 180 // 10

# read patches and masks
parser = argparse.ArgumentParser()
parser.add_argument('--input_patch', help='Input patches', default='data/patches/')
parser.add_argument('--input_mask', help='Input masks', default='data/masks/')
parser.add_argument('--output_path', help='Input masks', default='results/collagen_fiber_masks/')
args = parser.parse_args()
patches_folder = args.input_patch
mask_folder = args.input_mask
output_path = args.output_path
patches_files = glob.glob(os.path.join(patches_folder, "*"))
model_input_size = 3000

os.makedirs(output_path, exist_ok=True)
for file in tqdm(patches_files):
	if os.path.isfile(os.path.join(mask_folder, file.split("/")[-1])) == 0:
		continue
	features = []

	# read patch and mask
	mask = 255 - cv2.imread(os.path.join(mask_folder, file.split("/")[-1]), cv2.IMREAD_GRAYSCALE)
	patch = cv2.imread(file)
	patch = cv2.resize(patch, (model_input_size, model_input_size))
	patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

	# extract collagen fiber mask
	frag_thresh = filter_scale * 10
	bifs, jet = compute_bifs(patch, filter_scale, 0.015, 1.5)
	collagen_mask = bifs == feature_descriptor
	collagen_mask = np.logical_and(collagen_mask, mask)
	collagen_mask = morphology.remove_small_objects(collagen_mask.astype(bool), min_size=frag_thresh)

	# save collagen fiber mask
	collagen_mask = Image.fromarray(collagen_mask)
	image = Image.fromarray(np.array(patch))
	collagen_mask = collagen_mask.convert("RGBA")
	image = image.convert("RGBA")
	new_image = Image.blend(image, collagen_mask, 0.25)
	new_image.save(os.path.join(output_path, file.split("/")[-1]))