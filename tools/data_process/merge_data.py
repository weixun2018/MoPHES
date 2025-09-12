import json

def merge_json_files(file_paths, output_file):
    """
    Merge multiple JSON files containing arrays into a single array.
    
    Args:
        file_paths: List of file paths to merge
        output_file: Output file name
    """
    merged_data = []
    
    print(f"Merging {len(file_paths)} files:")
    
    # Process each file
    for file_path in file_paths:
        print(f"Processing: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # If data is a list, extend merged_data with its contents
            if isinstance(data, list):
                merged_data.extend(data)
            # If data is a single object, append it
            else:
                merged_data.append(data)
                
        except json.JSONDecodeError as e:
            print(f"Error reading {file_path}: {e}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
    
    # Write merged data to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nMerged {len(merged_data)} items into {output_file}")
        
    except Exception as e:
        print(f"Error writing output file: {e}")

# Example usage
if __name__ == "__main__":
    input_files = [
        "EmoLLM_format.json",
        "CPQA_format.json"
    ]
    
    output_path = "train.json"
    
    merge_json_files(input_files, output_path)