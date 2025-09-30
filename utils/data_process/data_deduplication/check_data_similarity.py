"""
Author: knull-cc
Date: 2025-04-02
Description: This script checks for duplicate or similar entries between
training and evaluation datasets using MinHash and LSH.
It removes the duplicate entries from the evaluation dataset and
saves the remaining unique entries to a new JSON file.
"""

import json
import numpy as np
from datasketch import MinHash, MinHashLSH
from typing import List, Dict
import re
from tqdm import tqdm

# Configuration parameters
INPUT_TRAIN_FILE = "deduplication_train_data.json"
INPUT_EVAL_FILE = "deduplication_eval_data.json"
OUTPUT_DEDUP_FILE = "deduplicated_eval_output.json"  # New output file name
SIMILARITY_THRESHOLD = 0.9

def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and extra whitespace.
    
    Args:
        text (str): The input string to clean.
        
    Returns:
        str: The cleaned string.
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

def get_document_signature(text: str, num_perm: int = 128) -> MinHash:
    """
    Generate MinHash signature for a document based on 3-grams.
    
    Args:
        text (str): The input text to create a signature for.
        num_perm (int): The number of permutations for the MinHash algorithm.
        
    Returns:
        MinHash: The MinHash signature object.
    """
    minhash = MinHash(num_perm=num_perm)
    # Split text into 3-grams
    words = text.split()
    for i in range(len(words) - 2):
        ngram = ' '.join(words[i:i+3])
        minhash.update(ngram.encode('utf-8'))
    return minhash

def calculate_similarity(data1: List[Dict], data2: List[Dict], threshold: float = 0.7) -> Dict:
    """
    Calculate similarity between two datasets using MinHash LSH to find duplicates.
    
    Args:
        data1 (List[Dict]): The first dataset (training data).
        data2 (List[Dict]): The second dataset (evaluation data).
        threshold (float): The similarity threshold for LSH.
        
    Returns:
        Dict: A dictionary containing details about the duplicate entries.
    """
    # Initialize LSH
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    
    # Store results
    results = {
        "duplicates": [],          # Duplicate data items
        "duplicate_pairs": [],     # Pairs of duplicate data
        "duplicate_indices": set() # Indices of duplicate data items in data2
    }
    
    print("Building LSH index for training data...")
    # Add training data to LSH
    for idx1, item1 in enumerate(tqdm(data1)):
        full_text1 = clean_text(item1['user'] + ' ' + item1['assistant'])
        signature1 = get_document_signature(full_text1)
        lsh.insert(f"data1_{idx1}", signature1)
    
    print("Checking for duplicates in evaluation data...")
    # Check each item in evaluation data against the LSH index
    for idx2, item2 in enumerate(tqdm(data2)):
        full_text2 = clean_text(item2['user'] + ' ' + item2['assistant'])
        signature2 = get_document_signature(full_text2)
        
        # Query similar documents
        similar_docs = lsh.query(signature2)
        if similar_docs:
            results["duplicates"].append(item2)
            results["duplicate_indices"].add(idx2)
            
            # Record duplicate pairs
            for doc_id in similar_docs:
                idx1 = int(doc_id.split('_')[1])
                results["duplicate_pairs"].append({
                    "eval_data": item2,
                    "train_data": data1[idx1]
                })
    
    return results

def main():
    """
    Main function to orchestrate the deduplication process.
    """
    # Read data
    print(f"Reading training data: {INPUT_TRAIN_FILE}")
    with open(INPUT_TRAIN_FILE, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    print(f"Reading evaluation data: {INPUT_EVAL_FILE}")
    with open(INPUT_EVAL_FILE, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # Calculate similarity to find the indices of duplicates
    results = calculate_similarity(train_data, eval_data, SIMILARITY_THRESHOLD)
    
    # Print statistics
    duplicate_count = len(results["duplicates"])
    total_eval_count = len(eval_data)
    
    print("\n=== Dataset Duplication Analysis Report ===")
    print(f"Training set size: {len(train_data)} items")
    print(f"Original evaluation set size: {total_eval_count} items")
    print(f"Duplicate count found in evaluation set: {duplicate_count} items")
    print("===========================================")

    # Create a new list with unique entries from the evaluation data
    deduplicated_eval_data = []
    duplicate_indices = results["duplicate_indices"]
    
    for idx, item in enumerate(eval_data):
        if idx not in duplicate_indices:
            deduplicated_eval_data.append(item)
            
    print(f"New deduplicated evaluation set size: {len(deduplicated_eval_data)} items")
    
    # Save the deduplicated data to a new file
    print(f"\nSaving deduplicated evaluation data to: {OUTPUT_DEDUP_FILE}")
    with open(OUTPUT_DEDUP_FILE, 'w', encoding='utf-8') as f:
       json.dump(deduplicated_eval_data, f, ensure_ascii=False, indent=2)
    
    print("Analysis and deduplication complete!")

if __name__ == "__main__":
    main()
