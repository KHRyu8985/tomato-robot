import os
import sys

def setup_project_root():
    # Get the absolute path of the script
    script_path = os.path.abspath(__file__)
    # Get the directory containing the script
    script_dir = os.path.dirname(script_path)
    # Get the project root directory (one level up from script directory)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    if project_root not in sys.path:
        # Add the project root to the Python path
        sys.path.append(project_root)
        print(f"Added {project_root} to Python path")

