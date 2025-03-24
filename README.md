# Tomato Robot
Real-time tomato detection and tracking system for demonstration.

## Project Structure

The project is organized as follows:
- `data/pictures/`: Contains image data for training and testing
- `video/`: Video files for demonstration and testing
- `script/`: Scripts for running the system
- `src/`: Source code
  - `action_recognition/`: Code for action recognition functionality
  - `florence2_model/`: Implementation of the Florence2 model
  - `sam2_model/`: Implementation of the SAM2 model
- `Pipfile` and `Pipfile.lock`: Python dependency management

## Installation

### Prerequisites
- Python 3.10
- pipenv (for dependency management)

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/tomato-robot.git
   cd tomato-robot
   ```

2. Install dependencies using pipenv:
   ```
   pipenv install
   ```

3. Activate the virtual environment:
   ```
   pipenv shell
   ```

### Download Checkpoint
1. Download SAM2 model checkpoint:
   ```
   # git bash
   
   cd checkpoints
   ./download_ckpts.sh
   ```
2. Download Yolo v8 tomato detection model weight from https://drive.google.com/drive/folders/11IqbObmeL-HkpezpPxxycvjWyypTffwy?usp=drive_link
   - Place `yolo_v8_tomato.pt` in the `checkpoints/` path

### Zed Camera Settings (Window)
1. Download ZED SDK from https://www.stereolabs.com/developers/release/ 
2. Download ZED SDK Python API
   ```
   cd C:\Program Files (x86)\ZED SDK
   python get_python_api.py
   ```

### Environment Management

1. View installed packages and their dependencies:
   ```
   pipenv graph
   ```

2. Install new packages:
   ```
   pipenv install package_name
   ```
   
   For development dependencies:
   ```
   pipenv install package_name --dev
   ```
   
   Install a specific version:
   ```
   pipenv install package_name==1.2.3
   ```

3. Working with the environment:
   
   - In terminal:
     ```
     pipenv shell
     ```
   
   - In VS Code:
     - Open VS Code from within the activated environment:
       ```
       pipenv shell
       code .
       ```
     - Or select the Pipenv environment as the Python interpreter:
       1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
       2. Type "Python: Select Interpreter"
       3. Choose the interpreter from the pipenv virtual environment
   
   - In Cursor:
     - Similar to VS Code, open Cursor from within the activated environment or select the Pipenv interpreter in the settings

## Demo

### Tomato Picker
   Execute `app.py` in the `demo`
   ```
   python demo/app.py
   ```

## Git Workflow

### Basic Commands

1. Check status of your changes:
   ```
   git status
   ```

2. Stage changes:
   ```
   git add <filename>    # Add specific file
   git add .             # Add all changes
   ```

3. Commit changes:
   ```
   git commit -m "Your commit message"
   ```

4. Push to GitHub (current branch-name is main):
   ```
   git push origin <branch-name>
   ```

### Handling Git Conflicts

When Git cannot automatically merge changes, you'll encounter a conflict. Here's how to resolve it:

1. Git will mark the conflicted files. Check them with:
   ```
   git status
   ```

2. Open the conflicted files in your editor. You'll see sections marked like this:
   ```
   <<<<<<< HEAD
   Your local changes
   =======
   Incoming changes from the remote/branch
   >>>>>>> branch-name
   ```

3. Edit the files to resolve conflicts:
   - Decide which changes to keep (yours, theirs, or a combination)
   - Remove the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
   - Save the files

4. After resolving all conflicts, mark them as resolved:
   ```
   git add <resolved-file>
   ```

5. Complete the merge by committing:
   ```
   git commit -m "Resolve merge conflicts"
   ```

6. Push your changes:
   ```
   git push origin <branch-name>
   ```

#### Using Visual Tools for Conflict Resolution

Many editors and IDEs have built-in tools for resolving conflicts:

- VS Code: Open the file and use the "Accept Current Change", "Accept Incoming Change", "Accept Both Changes", or "Compare Changes" options
- Git GUI tools like GitKraken, Sourcetree, or GitHub Desktop provide visual interfaces for conflict resolution

#### Preventing Conflicts

- Pull changes frequently: `git pull origin <branch-name>`
- Communicate with team members about which files you're working on
- Use feature branches for isolated work
- Consider using Git hooks or workflows that encourage smaller, more frequent commits