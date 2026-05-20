"""
Extract slices from multipage TIFF files, normalize them, and plot side by side.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import click


@click.command()
@click.argument('tv_path')
@click.argument('cg_path')
@click.option('--slice', 'slice_num', type=int, default=-1, help='Slice number to extract (default: middle slice)')
@click.option('--output', default='comparison.png', help='Output filename for the plot')
def extract_and_compare_tiffs(tv_path, cg_path, slice_num, output):
    """
    Extract slices from two multipage TIFF files and plot them side by side.
    
    TV_PATH: Path to reconstruction with total-variation
    
    CG_PATH: Path to reconstruction with conjugate-gradient
    """
    # Open TIFF files
    tv_img = Image.open(tv_path)
    cg_img = Image.open(cg_path)
    
    # Count slices
    tv_slices = tv_img.n_frames
    cg_slices = cg_img.n_frames
    
    print(f"TV TIFF: {tv_slices} slices")
    print(f"CG TIFF: {cg_slices} slices")
    
    # Determine slice number
    if slice_num is None:
        tv_slice_num = tv_slices // 2
        cg_slice_num = cg_slices // 2
    else:
        tv_slice_num = min(slice_num, tv_slices - 1)
        cg_slice_num = min(slice_num, cg_slices - 1)
    
    print(f"\nExtracting slice {tv_slice_num} from TV and slice {cg_slice_num} from CG")
    
    # Extract slices
    tv_img.seek(tv_slice_num)
    tv_slice = np.array(tv_img)
    
    cg_img.seek(cg_slice_num)
    cg_slice = np.array(cg_img)
    
    print(f"\nTV slice shape: {tv_slice.shape}, dtype: {tv_slice.dtype}")
    print(f"CG slice shape: {cg_slice.shape}, dtype: {cg_slice.dtype}")
    
    # Normalize each slice to [0, 1]
    tv_normalized = (tv_slice - tv_slice.min()) / (tv_slice.max() - tv_slice.min())
    cg_normalized = (cg_slice - cg_slice.min()) / (cg_slice.max() - cg_slice.min())
    
    print(f"\nTV normalized range: [{tv_normalized.min():.3f}, {tv_normalized.max():.3f}]")
    print(f"CG normalized range: [{cg_normalized.min():.3f}, {cg_normalized.max():.3f}]")
    
    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(tv_normalized, cmap='gray')
    axes[0].set_title(f'TV - Slice {tv_slice_num}/{tv_slices-1}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    im2 = axes[1].imshow(cg_normalized, cmap='gray')
    axes[1].set_title(f'CG - Slice {cg_slice_num}/{cg_slices-1}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout(w_pad=0.5)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as '{output}'")
    
    plt.close()


if __name__ == '__main__':
    extract_and_compare_tiffs()
