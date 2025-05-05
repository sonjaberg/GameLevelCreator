import subprocess
import time
import keyboard  # For detecting key presses

def run_opencv(opencv_script_path):
    try:
        print(f"Running OpenCV script: {opencv_script_path}")
        opencv_process = subprocess.Popen(["python", opencv_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("OpenCV script started successfully.")
        return opencv_process
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def run_godot(godot_project_path, godot_executable_path, scene_path):
    try:
        print(f"Launching Godot with scene: {scene_path}")
        command = [godot_executable_path, "--path", godot_project_path, "--scene", scene_path]
        godot_process = subprocess.Popen(command)
        print("Godot launched successfully.")
        return godot_process
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Paths (Update accordingly)
opencv_script = r"SonjaBerg_GameLevelCreator\pythonScripts\genTSCN.py"
godot_project = r"SonjaBerg_GameLevelCreator"
godot_exe = r"C:\Users\sonja\OneDrive\Documents\Godot_v4.4-stable_win64.exe\Godot_v4.4-stable_win64.exe"
scene_file = r"Scenes\world.tscn"

# Start OpenCV and Godot
opencv_process = run_opencv(opencv_script)
time.sleep(2.5)
godot_process = run_godot(godot_project, godot_exe, scene_file)

# Main Loop for Key Presses
print("Press 'R' to restart Godot. Press 'Q' to quit.")

while True:
    if keyboard.is_pressed('r'):  # Restart Godot when 'R' is pressed
        print("Restarting Godot...")
        godot_process.terminate()  # Kill the existing Godot process
        godot_process = run_godot(godot_project, godot_exe, scene_file)  # Relaunch Godot
    if keyboard.is_pressed('q'):  # Quit the program when 'Q' is pressed
        print("Exiting...")
        godot_process.terminate()
        opencv_process.terminate()
        break