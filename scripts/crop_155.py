""" # noqa: E501
Author: Chunyang Wang
Created on: 2025-10-14
Email: cw1722@ic.ac.uk
GitHub username: Chunyang Wang

Description:
Crop the 155 dataset to the first 1165 frames.
Example usage:
python script/crop_155.py \
    --input /Users/chunyang/projects/particle/data/Ketton155Data/155_segmented_cleaned_filtered/ \
    --output /Users/chunyang/projects/particle/data/Ketton155Data/155_segmented_cleaned_filtered_cropped/ \
    --crop_max 1165 \
    --crop_min 0 \
    --tag "" \

"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
import glob
import os
from natsort import natsorted
import tifffile
from tqdm import tqdm
import numpy as np
from skimage import io


parser = argparse.ArgumentParser(
    description="Template script for demonstration purposes.")

parser.add_argument(
    '--input', type=str,
    required=True,
    help='Input file path')
parser.add_argument(
    '--output', type=str,
    required=True,
    help='Output folder path')
parser.add_argument(
    '--crop_max', type=int,
    required=True,
    help='Maximum frame to crop to')
parser.add_argument(
    '--crop_min', type=int,
    required=True,
    help='Minimum frame to crop to')
parser.add_argument(
    '--tag', type=str,
    default="",
    help='Tag for the script')

args = parser.parse_args()


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
    for file in tqdm(input_files):
        print(file)
        file_name = os.path.basename(file)
        output_path = os.path.join(output_folder, file_name)
        seg_tif = io.imread(file)
        seg_tif = seg_tif[args.crop_min:args.crop_max, :, :]
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