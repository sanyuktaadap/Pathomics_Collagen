"""
Description of the file: Script for extracting patches in 40x ensuring empty regions are ignored.
"""


# header files
import math, sys, time, glob, os
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import ImageStat, Image
import numpy as np
from tqdm import tqdm

def patch_extraction(image_path, output_path, tile_size=1000):
    filename = os.path.basename(image_path).split('.png')[0].replace(" ", "_")

    if image_path.lower().endswith(".png"):
        # Process image
        image = Image.open(image_path).convert("RGB")
        gray_image = image.convert("L")
        width, height = image.size

        for i in range(0, width, tile_size):
            for j in range(0, height, tile_size):
                if i + tile_size > width or j + tile_size > height:
                    continue
                patch_name = f"{filename}_{i}_{j}.png"
                if os.path.exists(os.path.join(output_path, patch_name)):
                    continue
                box = (i, j, i + tile_size, j + tile_size)
                tile = image.crop(box)
                tile_mask = gray_image.crop(box)
                fn = lambda x: 0 if x > 220 or x < 50 else 1
                bin_mask = tile_mask.point(fn, mode='1')
                if ImageStat.Stat(bin_mask).mean[0] > 0.3:
                    tile.save(os.path.join(output_path, patch_name))

    else:
        # --- WSI branch with desired downsample = 16x ---
        slide = openslide.OpenSlide(image_path)
        dz = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=True)
        dz_level = dz.level_count - 1  # deepest DeepZoom level

        desired_downsample = 16.0  # "where 2 is 16"
        w0, h0 = slide.level_dimensions[0]
        mask_w, mask_h = int(w0 / desired_downsample), int(h0 / desired_downsample)

        # Build mask image at effective 16x downsample
        if slide.level_count > 2:
            # Use level 2 directly if present
            mask = slide.read_region((0, 0), 2, slide.level_dimensions[2]).convert("L")
            # If its size isn't exactly w0/16, resize to match the desired grid
            if mask.size != (mask_w, mask_h):
                mask = mask.resize((mask_w, mask_h), Image.BILINEAR)
        else:
            # Fall back: read from the highest available level and resize to w0/16
            fallback_level = slide.level_count - 1
            fw, fh = slide.level_dimensions[fallback_level]
            mask = slide.read_region((0, 0), fallback_level, (fw, fh)).convert("L")
            mask = mask.resize((mask_w, mask_h), Image.BILINEAR)

        # Binarize mask (simple bright/dark exclusion)
        fn = lambda x: 0 if x > 220 or x < 50 else 1
        mask = mask.point(fn, mode='1')

        # Mask tile size in mask space (tile_size @ level-0 → /16 in mask coords)
        mask_tile_size = int(round(tile_size / desired_downsample))

        for i in range(dz.level_tiles[dz_level][0]):
            for j in range(dz.level_tiles[dz_level][1]):
                coord_info = dz.get_tile_coordinates(dz_level, (i, j))
                if coord_info[2] != (tile_size, tile_size):
                    continue
                x0, y0 = coord_info[0]  # top-left in level-0 coords

                # Center of tile in level-0, then map to mask coords (divide by 16)
                cenX_mask = int(round((x0 + tile_size / 2) / desired_downsample))
                cenY_mask = int(round((y0 + tile_size / 2) / desired_downsample))

                half = mask_tile_size // 2
                left, top = cenX_mask - half, cenY_mask - half
                right, bottom = left + mask_tile_size, top + mask_tile_size

                # Skip if the mask window goes out of bounds
                if left < 0 or top < 0 or right > mask_w or bottom > mask_h:
                    continue

                mask_region = mask.crop((left, top, right, bottom))
                if ImageStat.Stat(mask_region).mean[0] > 0.3:
                    tile = dz.get_tile(dz_level, (i, j)).convert("RGB")
                    patch_name = f"{filename}_{x0}_{y0}.png"
                    tile.save(os.path.join(output_path, patch_name))


# extract patches
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='Input path to WSI or images', default='data/hari_BC/White_cohort/')
    parser.add_argument('--output_path', help='Output path to patches', default='data/hari_BC/patches/White_cohort')
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    cohorts = ["White_cohort"]
    for cohort in cohorts:
        os.makedirs(os.path.join(output_path, cohort), exist_ok=True)
        files = glob.glob(os.path.join(input_path, cohort, "*"))
        for file in tqdm(files):
            # print(file)
            patch_extraction(image_path=file, output_path=os.path.join(output_path, cohort), tile_size=3000)