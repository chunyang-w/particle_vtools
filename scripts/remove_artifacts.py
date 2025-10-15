""" # noqa: E501
Author: Chunyang Wang
Created on: 2025-10-14
Email: cw1722@ic.ac.uk
GitHub username: Chunyang Wang

Description:
Clean up the segmented images by removing small artifacts.

Example usage:
python script/remove_artifacts.py \
    --input /Users/chunyang/projects/particle/data/Ketton155Data/155_segmented_cleaned/ \
    --output /Users/chunyang/projects/particle/data/Ketton155Data/155_segmented_filtered \
    --tag "" \

"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
import cc3d
import numpy as np
from skimage import io
from tqdm import tqdm
import glob
import os
from natsort import natsorted
import tifffile


def save_bool_tif(
    volume: np.ndarray,
    output_path: str,
    compression: str = "zlib",
    compression_level: int = 6,
    tile_size: int = 256,
    imagej_compat: bool = False
):
    """
    Save a binary (0/1 or integer) volume as a compact 1-bit TIFF.

    - Automatically converts integer arrays to boolean (0/1).
    - Uses lossless compression and tiling for smaller file size and faster I/O.
    - Automatically switches to BigTIFF if >4 GB.
    """
    # Convert to boolean (ensure 1-bit per pixel on disk)
    arr = (volume > 0).astype(np.bool_)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)

    # 🔧 Compression settings
    comp = "deflate" if compression == "zlib" else compression
    comp_args = {"level": compression_level} if compression == "zlib" else {}

    # Optional tiling (helps compression + speed for big slices)
    tile = None
    if tile_size is not None:
        tile = (tile_size, tile_size)

    # BigTIFF if >4 GiB
    estimated_uncompressed = arr.size // 8  # 1 bit per pixel → /8
    bigtiff = estimated_uncompressed >= (4 * 1024**3)

    # Save TIFF
    tifffile.imwrite(
        output_path,
        arr,
        dtype=np.bool_,
        bitspersample=1,
        bigtiff=bigtiff,
        compression=comp,
        compressionargs=comp_args,
        tile=tile,
        photometric="minisblack",
        imagej=imagej_compat,
        metadata=None
    )
    print(f"[OK] Saved {output_path} | shape={arr.shape} | dtype=bool | 1-bit TIFF | BigTIFF={bigtiff}")


def remove_small_components_cc3d(binary_volume, min_size=100000):
    """
    Fastest option using cc3d library for 3D connected components
    """
    # Get connected components (26-connectivity for 3D)
    labels = cc3d.connected_components(binary_volume, connectivity=26)

    # Count sizes of each component
    component_sizes = np.bincount(labels.ravel())

    # Create mask for components to keep (size >= min_size)
    # Skip background (label 0)
    keep_components = component_sizes >= min_size
    keep_components[0] = False  # Always remove background

    # Create output volume
    filtered_volume = np.zeros_like(binary_volume)
    for label_id in np.where(keep_components)[0]:
        filtered_volume[labels == label_id] = 1
    return filtered_volume


parser = argparse.ArgumentParser(
    description="Clean up the segmented images by removing small artifacts.")

parser.add_argument(
    '--input', type=str,
    required=True,
    help='Input file path')
parser.add_argument(
    '--output', type=str,
    required=True,
    help='Output folder path')
parser.add_argument(
    '--tag', type=str,
    default="",
    help='Tag for the script')

args = parser.parse_args()


def main(args):
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("*" * 50)

    """
    Define main logic here.
    """
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    input_files = natsorted(
        glob.glob(os.path.join(input_folder, "*.tif"))
    )
    print(f"Found {len(input_files)} input files")

    # This magic number is a tick to slice off the tiff file - 
    # for 155 data small blobs only appear in the first 200 slices
    # use this slice off to win some time
    cut_off = 300

    for file in tqdm(input_files):
        print(file)
        file_name = os.path.basename(file)
        output_path = os.path.join(output_folder, file_name)
        seg_tif = io.imread(file)
        if cut_off is not None:
            filtered_seg_tif_seg = remove_small_components_cc3d(seg_tif[:cut_off, :, :])
            seg_tif[:cut_off, :, :] = filtered_seg_tif_seg
        else:
            seg_tif = remove_small_components_cc3d(seg_tif)
        save_bool_tif(seg_tif, output_path)
    return


class DualOutput:
    """Redirect output to both terminal and log file."""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


if __name__ == "__main__":
    script_name = Path(__file__).stem

    # Create output folder
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create log file inside output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_folder / f"{script_name}_{args.tag}.log"

    # Redirect all stdout to both terminal and log file
    original_stdout = sys.stdout
    dual_output = DualOutput(log_file)
    sys.stdout = dual_output

    try:
        print("Timestamp: ", timestamp)
        print("=" * 50)
        print(f"Running script: {script_name}")
        print("=" * 50)
        print()

        print(f"Output folder: {output_folder}")
        print(f"Log file: {log_file}")
        print("*" * 50)

        main(args)

        print()
        print("=" * 50)
        print("Script execution completed.")

    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        dual_output.close()
