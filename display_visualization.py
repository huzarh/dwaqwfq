#!/usr/bin/env python3
"""
Display visualization script.
This script shows the visualizations generated during training and evaluation.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def display_visualization(image_path):
    """
    Display a single visualization image.
    
    Args:
        image_path: Path to the image file
    """
    try:
        # Load and display the image
        img = mpimg.imread(image_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(image_path), fontsize=16)
        plt.tight_layout()
        plt.show()
        
        print(f"Displayed visualization: {os.path.basename(image_path)}")
        
    except Exception as e:
        print(f"Error displaying visualization: {e}")
        return False
    
    return True

def display_all_visualizations(run_dir, max_cols=3):
    """
    Display all visualizations for a specific run in a grid.
    
    Args:
        run_dir: Path to the run directory
        max_cols: Maximum number of columns in the grid
    """
    # Find all visualization directories
    model_viz_dir = os.path.join(run_dir, "model", "visualizations")
    eval_viz_dir = os.path.join(run_dir, "evaluation")
    
    # Find all png files
    all_images = []
    for viz_dir in [model_viz_dir, eval_viz_dir]:
        if os.path.exists(viz_dir):
            all_images.extend(glob.glob(os.path.join(viz_dir, "*.png")))
    
    if not all_images:
        print(f"No visualizations found in {run_dir}")
        return False
    
    # Sort images
    all_images.sort()
    
    # Display images in a grid
    n_images = len(all_images)
    n_cols = min(max_cols, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    print(f"Displaying {n_images} visualizations from {run_dir}")
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, img_path in enumerate(all_images):
        plt.subplot(n_rows, n_cols, i + 1)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(img_path), fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return True

def list_runs(base_dir="training_outputs"):
    """
    List all available runs in the base directory.
    
    Args:
        base_dir: Base directory containing runs
    """
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist")
        return []
    
    # Find all run directories
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))]
    run_dirs.sort(reverse=True)
    
    if not run_dirs:
        print(f"No runs found in {base_dir}")
        return []
    
    print("Available runs:")
    for i, run_dir in enumerate(run_dirs):
        print(f"  {i+1}. {run_dir}")
    
    return run_dirs

def get_latest_run(base_dir="training_outputs"):
    """
    Get the latest run directory.
    
    Args:
        base_dir: Base directory containing runs
        
    Returns:
        Path to the latest run directory, or None if not found
    """
    runs = list_runs(base_dir)
    if not runs:
        return None
    
    return os.path.join(base_dir, runs[0])

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Display visualizations from training and evaluation")
    parser.add_argument("--run_dir", type=str, help="Path to the run directory")
    parser.add_argument("--image", type=str, help="Path to a specific image to display")
    parser.add_argument("--list_runs", action="store_true", help="List all available runs")
    parser.add_argument("--base_dir", type=str, default="training_outputs", help="Base directory containing runs")
    
    args = parser.parse_args()
    
    if args.list_runs:
        list_runs(args.base_dir)
        return 0
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image file {args.image} does not exist")
            return 1
        
        display_visualization(args.image)
        return 0
    
    run_dir = args.run_dir
    if not run_dir:
        run_dir = get_latest_run(args.base_dir)
        if not run_dir:
            print("No run directory specified and no runs found")
            return 1
    
    if not os.path.exists(run_dir):
        print(f"Run directory {run_dir} does not exist")
        return 1
    
    display_all_visualizations(run_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 