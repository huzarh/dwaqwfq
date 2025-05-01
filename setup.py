import os
import argparse

def create_directory_structure():
    """Create the directory structure for the project."""
    directories = [
        "logs",
        "models",
        "plots",
        "test_output"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up the project directory structure")
    args = parser.parse_args()
    
    print("Setting up Speaker Classification project...")
    create_directory_structure()
    print("Setup complete!")
    print("\nTo install dependencies, run:")
    print("pip install -r requirements.txt")
    print("\nTo train the model, run:")
    print("python train.py --data_dir dataset --model_type attention")
    print("\nTo generate predictions, run:")
    print("python inference.py --test_dir path/to/test --model_dir path/to/model") 