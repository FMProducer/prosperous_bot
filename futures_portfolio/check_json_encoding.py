import os
import json

def check_json_files():
    for file in os.listdir('.'):
        if file.endswith('.json'):
            try:
                with open(file, 'rb') as f:
                    content = f.read()
                content.decode('utf-8')
            except UnicodeDecodeError as e:
                print(f"File {file} has invalid UTF-8 at position {e.start}: {e}")
            except Exception as e:
                print(f"Error reading {file}: {e}")

if __name__ == "__main__":
    check_json_files()
