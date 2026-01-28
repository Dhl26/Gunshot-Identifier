import os
import glob
import sys

MODEL_PATH = "gunshot_cnn_model.pth"
CHUNK_SIZE = 95 * 1024 * 1024  # 95 MB chunks (safe for GitHub 100MB limit)

def split_model(file_path=MODEL_PATH):
    """
    Splits the model file into smaller chunks.
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    part_num = 1
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            
            part_name = f"{file_path}.part{part_num:03d}"
            with open(part_name, 'wb') as chunk_file:
                chunk_file.write(chunk)
            
            print(f"Created {part_name} ({len(chunk)/1024/1024:.2f} MB)")
            part_num += 1
            
    print(f"Successfully split {file_path} into {part_num-1} parts.")
    print(f"You can now upload the .part files to GitHub.")
    print(f"Don't forget to add '{file_path}' to your .gitignore!")

def join_model(output_path=MODEL_PATH):
    """
    Joins the model chunks back into a single file.
    """
    # Find all parts
    parts = sorted(glob.glob(f"{output_path}.part*"))
    
    if not parts:
        print("No model parts found to join.")
        return False
        
    print(f"Found {len(parts)} parts. Joining...")
    
    with open(output_path, 'wb') as outfile:
        for part in parts:
            print(f"Reading {part}...")
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
                
    print(f"Successfully reconstructed {output_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--split":
            split_model()
        elif sys.argv[1] == "--join":
            join_model()
        else:
            print("Usage: python model_manager.py [--split | --join]")
    else:
        # Interactive mode
        print("--- Model File Manager ---")
        print("1. Split Model (Prepare for GitHub)")
        print("2. Join Model (Restore from Parts)")
        choice = input("Select option (1/2): ")
        
        if choice == "1":
            split_model()
        elif choice == "2":
            join_model()
