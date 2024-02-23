import subprocess

# Command to find the path of 'obabel' executable
find_path_command = ["which", "obabel"]
result = subprocess.run(find_path_command, capture_output=True, text=True)

if result.returncode == 0:
    print(f"Path to 'obabel': {result.stdout.strip()}")
else:
    print("obabel not found in PATH")
