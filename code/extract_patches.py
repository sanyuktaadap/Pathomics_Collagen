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
                    # patch_name = f"{filename}_{i}_{j}.png"
                    tile.save(os.path.join(output_path, patch_name))

    else:
        # Process WSI image
        slide = openslide.OpenSlide(image_path)
        dz = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=True)
        dz_level = dz.level_count - 1
        downsample_ratio = slide.level_downsamples[2]
        mask_tile_size = int(tile_size * slide.level_downsamples[0] // downsample_ratio)

        # Read full mask from level 2
        mask = slide.read_region((0, 0), 2, slide.level_dimensions[2]).convert("L")
        fn = lambda x: 0 if x > 220 or x < 50 else 1
        mask = mask.point(fn, mode='1')

        # loop through each patch of the slide of size=(tile_size, tile_size)
        for i in range(dz.level_tiles[dz_level][0]):
            for j in range(dz.level_tiles[dz_level][1]):
                coord_info = dz.get_tile_coordinates(dz_level, (i, j))
                if coord_info[2] != (tile_size, tile_size):
                    continue
                coord = coord_info[0]
                cenX = (coord[0] + tile_size * slide.level_downsamples[0] // 2) // downsample_ratio
                cenY = (coord[1] + tile_size * slide.level_downsamples[0] // 2) // downsample_ratio
                mask_region = mask.crop((cenX - mask_tile_size // 2, cenY - mask_tile_size // 2,
                                         cenX + mask_tile_size // 2, cenY + mask_tile_size // 2))
                if ImageStat.Stat(mask_region).mean[0] > 0.3:
                    tile = dz.get_tile(dz_level, (i, j)).convert("RGB")
                    patch_name = f"{filename}_{coord[0]}_{coord[1]}.png"
                    tile.save(os.path.join(output_path, patch_name))

# extract patches
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', help='Cohort name', default='White_cohort')
    parser.add_argument('--input_path', help='Input path to WSI or images', default='data/hari_BC/White_cohort/')
    parser.add_argument('--output_path', help='Output path to patches', default='data/hari_BC/patches/White_cohort')
    args = parser.parse_args()
    cohort = args.cohort
    input_path = args.input_path
    output_path = args.output_path

    print(input_path)

    os.makedirs(output_path, exist_ok=True)
    files = glob.glob(input_path + "*")

    for file in tqdm(files):
        print(file)
        patch_extraction(image_path=file, output_path=output_path, tile_size=3000)