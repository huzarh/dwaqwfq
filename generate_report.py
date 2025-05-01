#!/usr/bin/env python3
"""
Generate a comprehensive HTML report with all visualizations and metrics.
"""

import os
import sys
import argparse
import pickle
import glob
import json
import shutil
import datetime
from typing import Dict, List, Any

def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """
    Load metrics from a pickle file.
    
    Args:
        metrics_path: Path to the metrics file
        
    Returns:
        Dictionary with metrics
    """
    print(f"Loading metrics from: {metrics_path}")
    try:
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
        print(f"Loaded metrics: {list(metrics.keys())}")
        return metrics
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return {}

def copy_images(run_dir: str, output_dir: str) -> List[str]:
    """
    Copy all visualization images to the output directory.
    
    Args:
        run_dir: Path to the run directory
        output_dir: Path to the output directory
        
    Returns:
        List of image filenames
    """
    print(f"Looking for images in run directory: {run_dir}")
    
    # Find all visualization directories
    model_viz_dir = os.path.join(run_dir, "model", "visualizations")
    eval_viz_dir = os.path.join(run_dir, "evaluation")
    
    print(f"Model visualizations dir: {model_viz_dir} (exists: {os.path.exists(model_viz_dir)})")
    print(f"Evaluation dir: {eval_viz_dir} (exists: {os.path.exists(eval_viz_dir)})")
    
    # Create images directory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    print(f"Created images directory: {images_dir}")
    
    # Find all png files
    image_files = []
    for viz_dir in [model_viz_dir, eval_viz_dir]:
        if os.path.exists(viz_dir):
            png_files = glob.glob(os.path.join(viz_dir, "*.png"))
            print(f"Found {len(png_files)} PNG files in {viz_dir}")
            for img_path in png_files:
                img_name = os.path.basename(img_path)
                dest_path = os.path.join(images_dir, img_name)
                shutil.copy2(img_path, dest_path)
                print(f"Copied {img_path} -> {dest_path}")
                image_files.append(img_name)
    
    print(f"Total images copied: {len(image_files)}")
    return sorted(image_files)

def generate_html(run_dir: str, image_files: List[str], metrics: Dict[str, Any], output_path: str):
    """
    Generate an HTML report.
    
    Args:
        run_dir: Path to the run directory
        image_files: List of image filenames
        metrics: Dictionary with metrics
        output_path: Path to the output HTML file
    """
    print(f"Generating HTML report for run: {run_dir}")
    print(f"Number of images: {len(image_files)}")
    print(f"Metrics keys: {list(metrics.keys())}")
    
    # Basic metrics
    accuracy = metrics.get('accuracy', 'N/A')
    macro_f1 = metrics.get('macro_f1', 'N/A')
    
    print(f"Accuracy: {accuracy}")
    print(f"Macro F1: {macro_f1}")
    
    # Format accuracy and macro_f1 for display
    if isinstance(accuracy, float):
        accuracy_str = f"{accuracy * 100:.2f}%"
    else:
        accuracy_str = str(accuracy)
        
    if isinstance(macro_f1, float):
        macro_f1_str = f"{macro_f1 * 100:.2f}%"
    else:
        macro_f1_str = str(macro_f1)
    
    # Detailed class metrics
    class_metrics = {}
    if 'report' in metrics:
        for label, metrics_dict in metrics['report'].items():
            if isinstance(metrics_dict, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:
                class_metrics[label] = metrics_dict
    
    print(f"Class metrics: {list(class_metrics.keys())}")
    
    # Start generating HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speaker Classification Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }}
        .metrics {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
        }}
        .metric-card {{
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .visualizations {{
            margin-bottom: 30px;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 20px;
        }}
        .viz-item {{
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
        }}
        .viz-item img {{
            width: 100%;
            height: auto;
            border-radius: 3px;
        }}
        .viz-title {{
            margin-top: 10px;
            text-align: center;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Speaker Classification Report</h1>
        <p>Run: {os.path.basename(run_dir)}</p>
        <p>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <h2>Model Performance Metrics</h2>
    <div class="metrics">
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{accuracy_str}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{macro_f1_str}</div>
                <div class="metric-label">Macro F1 Score</div>
            </div>
        </div>
    </div>
    
    <h2>Per-Class Performance</h2>
    <table>
        <thead>
            <tr>
                <th>Speaker</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>Support</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # Add class metrics rows
    for label, metrics_dict in class_metrics.items():
        precision = metrics_dict.get('precision', 'N/A')
        recall = metrics_dict.get('recall', 'N/A')
        f1_score = metrics_dict.get('f1-score', 'N/A')
        support = metrics_dict.get('support', 'N/A')
        
        prec_str = f"{precision:.4f}" if isinstance(precision, float) else str(precision)
        rec_str = f"{recall:.4f}" if isinstance(recall, float) else str(recall)
        f1_str = f"{f1_score:.4f}" if isinstance(f1_score, float) else str(f1_score)
        
        html_content += f"""
            <tr>
                <td>{label}</td>
                <td>{prec_str}</td>
                <td>{rec_str}</td>
                <td>{f1_str}</td>
                <td>{support}</td>
            </tr>
"""
    
    html_content += """
        </tbody>
    </table>
    
    <h2>Visualizations</h2>
    <div class="visualizations">
        <div class="viz-grid">
"""
    
    # Add visualization items
    for img_file in image_files:
        title = img_file.replace('_', ' ').replace('.png', '')
        html_content += f"""
            <div class="viz-item">
                <img src="images/{img_file}" alt="{img_file}">
                <div class="viz-title">{title}</div>
            </div>
"""
    
    html_content += """
        </div>
    </div>
    
    <div class="footer">
        <p>Speaker Classification System - Generated Report</p>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated HTML report: {output_path}")

def get_latest_run(base_dir="training_outputs"):
    """
    Get the latest run directory.
    
    Args:
        base_dir: Base directory containing runs
        
    Returns:
        Path to the latest run directory, or None if not found
    """
    print(f"Looking for latest run in: {base_dir}")
    
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist")
        return None
    
    # Find all run directories
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))]
    run_dirs.sort(reverse=True)
    
    print(f"Found {len(run_dirs)} run directories")
    
    if not run_dirs:
        print(f"No runs found in {base_dir}")
        return None
    
    latest_run = os.path.join(base_dir, run_dirs[0])
    print(f"Latest run: {latest_run}")
    
    return latest_run

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate a comprehensive HTML report")
    parser.add_argument("--run_dir", type=str, help="Path to the run directory")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory to save the report")
    parser.add_argument("--base_dir", type=str, default="training_outputs", help="Base directory containing runs")
    
    args = parser.parse_args()
    print(f"Args: {args}")
    
    # Get run directory
    run_dir = args.run_dir
    if not run_dir:
        run_dir = get_latest_run(args.base_dir)
        if not run_dir:
            print("No run directory specified and no runs found")
            return 1
    
    print(f"Using run directory: {run_dir}")
    
    if not os.path.exists(run_dir):
        print(f"Run directory {run_dir} does not exist")
        return 1
    
    # Create output directory
    run_name = os.path.basename(run_dir)
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Load metrics
    metrics_path = os.path.join(run_dir, "evaluation", "metrics.pkl")
    print(f"Metrics path: {metrics_path} (exists: {os.path.exists(metrics_path)})")
    
    metrics = load_metrics(metrics_path) if os.path.exists(metrics_path) else {}
    
    # Copy images
    image_files = copy_images(run_dir, output_dir)
    
    if not image_files:
        print(f"No visualization images found in {run_dir}")
        return 1
    
    # Generate HTML
    output_path = os.path.join(output_dir, "report.html")
    generate_html(run_dir, image_files, metrics, output_path)
    
    print(f"Report generated successfully at {output_path}")
    print(f"You can open this file in your web browser to view the report.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 