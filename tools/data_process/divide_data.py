"""
Author: knull-cc
Date: 2025-01-20
Description: This script merges multiple JSON files containing conversation data 
from the data directory and directly splits them into training and validation sets 
in the divide directory.
"""

import json
import os
from pathlib import Path
import random

def merge_json_files(data_dir, output_dir):
    """
    Merges JSON files from the specified directory and splits the merged data into training and validation sets.

    Parameters:
    data_dir (str): The directory containing the JSON files to be merged.
    output_dir (str): The directory where the training and validation sets will be saved.
    """
    # Create an empty list to store all conversation data
    merged_data = []
    
    # Get all json files in the data directory
    data_path = Path(data_dir)
    json_files = list(data_path.glob('*.json'))
    
    # Iterate over all json files
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # If data is a list, extend merged_data with its contents
                if isinstance(data, list):
                    merged_data.extend(data)
                # If data is a single object, append it
                else:
                    merged_data.append(data)
        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}")
    
    print(f"Successfully loaded {len(json_files)} files with {len(merged_data)} conversations")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed to ensure reproducibility
    random.seed(42)
    
    # Shuffle the data randomly
    random.shuffle(merged_data)
    
    # Calculate the split point (90% train, 10% validation)
    total_samples = len(merged_data)
    train_size = int(total_samples * 0.9)
    
    # Split the data
    train_data = merged_data[:train_size]
    dev_data = merged_data[train_size:]
    
    try:
        # Save the training set
        train_file = os.path.join(output_dir, 'train.json')
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
            
        # Save the validation set
        dev_file = os.path.join(output_dir, 'dev.json')
        with open(dev_file, 'w', encoding='utf-8') as f:
            json.dump(dev_data, f, ensure_ascii=False, indent=2)
            
        print(f"Dataset split completed!")
        print(f"Training set size: {len(train_data)}, saved to: {train_file}")
        print(f"Validation set size: {len(dev_data)}, saved to: {dev_file}")
        print(f"Split ratio: {len(train_data)/total_samples:.2%} train / {len(dev_data)/total_samples:.2%} validation")
        
    except Exception as e:
        print(f"Error writing output files: {str(e)}")

if __name__ == "__main__":
    # Specify the data folder path and output directory
    data_directory = "data"
    divide_directory = "divide"
    merge_json_files(data_directory, divide_directory)