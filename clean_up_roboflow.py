from pathlib import Path

def process_label_file(file_path):
    """Process a single label file to update class IDs"""
    modified = False
    new_lines = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            new_lines.append(line)
            continue
            
        class_id = int(parts[0])
        
        # Skip the generic "bishop" class
        if class_id == 0:
            modified = True
            continue
            
        # Shift class IDs down by 1
        new_class_id = class_id - 1
        new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
        new_lines.append(new_line)
        modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        return True
    return False

def main():
    base_dir = Path("data/chess_pieces_roboflow")
    label_dirs = [
        base_dir / "train" / "labels",
        base_dir / "test" / "labels",
        base_dir / "valid" / "labels"
    ]
    
    # Process all label files
    files_processed = 0
    files_modified = 0
    
    for label_dir in label_dirs:
        if not label_dir.exists():
            print(f"Warning: Directory {label_dir} does not exist")
            continue
            
        for label_file in label_dir.glob("*.txt"):
            files_processed += 1
            if process_label_file(label_file):
                files_modified += 1
                
    print(f"Processed {files_processed} files, modified {files_modified} files")

if __name__ == "__main__":
    main()