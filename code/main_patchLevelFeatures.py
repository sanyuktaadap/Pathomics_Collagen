"""
Authors: Arpit Aggarwal
File: Main file to extract collagen features at patch level
"""

# header files to load
from compute_bifs import compute_bifs as compute_bifs
import numpy as np
import cv2
from skimage import measure, morphology
from PIL import Image
from disorder_feat_extract import contrast_entropy as disorder_feat_extract
import math
import os
import glob
import argparse
import csv


# MAIN CODE
win_sizes = [200, 250, 300, 350, 400, 450, 500, 550, 600]
filter_scale = 3
feature_descriptor = 5
orient_num = 180 // 10

# read patches and masks
parser = argparse.ArgumentParser()
parser.add_argument('--input_patch', help='Input patches', default='data/patches/')
parser.add_argument('--input_mask', help='Input masks', default='data/masks/')
parser.add_argument('--output_heatmap1_win200', help='Input masks', default='results/heatmaps_1/200/')
parser.add_argument('--output_heatmap1_win400', help='Input masks', default='results/heatmaps_1/400/')
parser.add_argument('--output_heatmap1_win600', help='Input masks', default='results/heatmaps_1/600/')
parser.add_argument('--output_heatmap2_win200', help='Input masks', default='results/heatmaps_2/200/')
parser.add_argument('--output_heatmap2_win400', help='Input masks', default='results/heatmaps_2/400/')
parser.add_argument('--output_heatmap2_win600', help='Input masks', default='results/heatmaps_2/600/')
parser.add_argument('--output_feature', help='Input masks', default='results/patches/')
args = parser.parse_args()
patches_folder = args.input_patch
mask_folder = args.input_mask
patches_files = glob.glob(patches_folder+"*png")

for file in patches_files:
	print(file)
	features = []

	# read patch and mask
	mask = 255 - cv2.imread(mask_folder+file.split("/")[-1], cv2.IMREAD_GRAYSCALE)
	patch = cv2.imread(file)
	patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
	
	# FIRST SET OF FEATURES FROM ENTIRE STROMAL AREAS
	# extract collagen fiber mask
	frag_thresh = filter_scale * 10
	bifs, jet = compute_bifs(patch, filter_scale, 0.015, 1.5)
	collagen_mask = bifs == feature_descriptor
	collagen_mask = np.logical_and(collagen_mask, mask)
	collagen_mask = morphology.remove_small_objects(collagen_mask.astype(bool), min_size=frag_thresh)

	# save collagen fiber mask
	#collagen_mask = Image.fromarray(collagen_mask)
	#collagen_mask.save("../results/collagen/"+file.split("/")[-1])
	
	# collagen centroid and orientation information extraction
	collagen_mask = collagen_mask.astype(np.uint16)
	collagen_mask_np_u8 = collagen_mask.copy().astype(np.uint8)
	contours, hierarchy = cv2.findContours(collagen_mask_np_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for i, contour in enumerate(contours):
		cv2.drawContours(collagen_mask, [contour], -1, color=(i+1), thickness=-1)
	properties = ('centroid', 'orientation', 'area')
	collogen_props = measure.regionprops_table(collagen_mask, properties=properties)
	colg_center = np.array([collogen_props['centroid-0'], collogen_props['centroid-1']]).T
	colg_area = collogen_props['area']
	colg_orient = collogen_props['orientation']
	colg_orient = np.array([math.degrees(orient) for orient in colg_orient])
	colg_orient_bin = np.fix(colg_orient / 10) + 9

	for win_size in win_sizes:
		step_size = win_size
		map = np.zeros((int((patch.shape[0] - win_size) / step_size) + 1, int((patch.shape[1] - win_size) / step_size) + 1, 13))
		height, width = collagen_mask.shape
		for win_x in range(0, width - win_size + 1, step_size):
				for win_y in range(0, height - win_size + 1, step_size):
						in_window_indices = np.where(
								(colg_center[:, 0] >= win_x) & 
								(colg_center[:, 0] < win_x + win_size) & 
								(colg_center[:, 1] >= win_y) & 
								(colg_center[:, 1] < win_y + win_size)
						)
						in_window_indices = in_window_indices[0]
						if len(in_window_indices) >= 2:
								inwin_colg_orient = colg_orient_bin[in_window_indices]
								inwin_colg_area = colg_area[in_window_indices]
								orient_occur_feats = disorder_feat_extract(inwin_colg_orient, inwin_colg_area, orient_num, 1)
								if orient_occur_feats is not None:
									map[win_y // step_size, win_x // step_size, :] = np.array(list(orient_occur_feats.values()))
		
		# save heatmaps and features
		#if win_size == 200:
		#	np.save(args.output_heatmap_win200+file.split("/")[-1][:-4]+".npy", map[:,:,4].T)
		#elif win_size == 400:
		#	np.save(args.output_heatmap_win400+file.split("/")[-1][:-4]+".npy", map[:,:,4].T)
		#else:
		#	np.save(args.output_heatmap_win600+file.split("/")[-1][:-4]+".npy", map[:,:,4].T)
		mean = np.mean(map[:,:,4])
		max = np.max(map[:,:,4])
		features.append(mean)
		features.append(max)


	# SECOND SET OF FEATURES FROM PERITUMORAL AREAS
	# get dilated mask
	im_dilated = cv2.imread(mask_folder+file.split("/")[-1], cv2.IMREAD_GRAYSCALE)
	for index1 in range(0, 50):
		im_dilated = cv2.dilate(im_dilated, np.ones((5, 5), np.uint8), iterations=1)
	im_new = im_dilated
	for index1 in range(0, 150):
		im_new = cv2.dilate(im_new, np.ones((5, 5), np.uint8), iterations=1)

	# extract collagen fiber mask
	frag_thresh = filter_scale * 10
	bifs, jet = compute_bifs(patch, filter_scale, 0.015, 1.5)
	collagen_mask = bifs == feature_descriptor
	collagen_mask = np.logical_and(collagen_mask, im_new)
	collagen_mask = np.logical_and(collagen_mask, 255-im_dilated)
	collagen_mask = morphology.remove_small_objects(collagen_mask.astype(bool), min_size=frag_thresh)

	# save collagen fiber mask
	#collagen_mask = Image.fromarray(collagen_mask)
	#collagen_mask.save("../results/collagen/"+file.split("/")[-1])
	
	# collagen centroid and orientation information extraction
	collagen_mask = collagen_mask.astype(np.uint16)
	collagen_mask_np_u8 = collagen_mask.copy().astype(np.uint8)
	contours, hierarchy = cv2.findContours(collagen_mask_np_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	for i, contour in enumerate(contours):
		cv2.drawContours(collagen_mask, [contour], -1, color=(i+1), thickness=-1)

	properties = ('centroid', 'orientation', 'area')
	collogen_props = measure.regionprops_table(collagen_mask, properties=properties)
	colg_center = np.array([collogen_props['centroid-0'], collogen_props['centroid-1']]).T
	colg_area = collogen_props['area']
	colg_orient = collogen_props['orientation']
	colg_orient = np.array([math.degrees(orient) for orient in colg_orient])
	colg_orient_bin = np.fix(colg_orient / 10) + 9

	for win_size in win_sizes:
		step_size = win_size
		map = np.zeros((int((patch.shape[0] - win_size) / step_size) + 1, int((patch.shape[1] - win_size) / step_size) + 1, 13))
		height, width = collagen_mask.shape
		for win_x in range(0, width - win_size + 1, step_size):
				for win_y in range(0, height - win_size + 1, step_size):
						in_window_indices = np.where(
								(colg_center[:, 0] >= win_x) & 
								(colg_center[:, 0] < win_x + win_size) & 
								(colg_center[:, 1] >= win_y) & 
								(colg_center[:, 1] < win_y + win_size)
						)
						in_window_indices = in_window_indices[0]
						if len(in_window_indices) >= 2:
								inwin_colg_orient = colg_orient_bin[in_window_indices]
								inwin_colg_area = colg_area[in_window_indices]
								orient_occur_feats = disorder_feat_extract(inwin_colg_orient, inwin_colg_area, orient_num, 1)
								if orient_occur_feats is not None:
									map[win_y // step_size, win_x // step_size, :] = np.array(list(orient_occur_feats.values()))
		
		# save heatmaps and features
		#if win_size == 200:
		#	np.save(args.output_heatmap_win200+file.split("/")[-1][:-4]+".npy", map[:,:,4].T)
		#elif win_size == 400:
		#	np.save(args.output_heatmap_win400+file.split("/")[-1][:-4]+".npy", map[:,:,4].T)
		#else:
		#	np.save(args.output_heatmap_win600+file.split("/")[-1][:-4]+".npy", map[:,:,4].T)
		mean = np.mean(map[:,:,4])
		max = np.max(map[:,:,4])
		features.append(mean)
		features.append(max)
	
	with open(args.output_feature+file.split("/")[-1][:-4]+".csv", mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(features)