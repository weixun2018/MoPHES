import json
import re
from datasketch import MinHashLSH, MinHash
from collections import defaultdict
import jieba
import hashlib

def preprocess_text(text):
    """
    Preprocess Chinese text for better similarity detection
    """
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Use jieba for Chinese word segmentation
    words = jieba.cut(text)
    
    # Filter out very short words and create shingles
    words = [word.strip() for word in words if len(word.strip()) > 1]
    
    # Create character-level n-grams for better similarity detection
    char_ngrams = []
    for i in range(len(text) - 2):
        char_ngrams.append(text[i:i+3])
    
    # Combine word-level and character-level features
    features = words + char_ngrams
    
    return features

def create_minhash(features, num_perm=128):
    """
    Create MinHash signature from text features
    """
    minhash = MinHash(num_perm=num_perm)
    for feature in features:
        minhash.update(feature.encode('utf-8'))
    return minhash

def deduplicate_json_data(input_file, output_file, similarity_threshold=0.7, num_perm=128):
    """
    Deduplicate JSON data using MinHash and LSH
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output deduplicated JSON file
        similarity_threshold: Similarity threshold (0.7 = 70%)
        num_perm: Number of permutations for MinHash (higher = more accurate)
    """
    
    print("Loading data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} records")
    
    # Initialize LSH with threshold
    # LSH threshold calculation: threshold = (1/b)^(1/r) where b*r = num_perm
    # For 70% similarity, we need to tune b and r
    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)
    
    # Store MinHash signatures and track duplicates
    signatures = {}
    duplicate_groups = defaultdict(list)
    unique_data = []
    processed_ids = set()
    
    print("Creating MinHash signatures...")
    for i, record in enumerate(data):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(data)} records")
        
        # Combine user and assistant text for similarity comparison
        combined_text = record.get('user', '') + ' ' + record.get('assistant', '')
        
        # Preprocess and create features
        features = preprocess_text(combined_text)
        
        if not features:  # Skip empty records
            continue
            
        # Create MinHash signature
        minhash_sig = create_minhash(features, num_perm)
        signatures[i] = minhash_sig
        
        # Query LSH for similar items
        similar_items = lsh.query(minhash_sig)
        
        if similar_items:
            # Found similar items, group them
            group_id = min(similar_items)  # Use the smallest ID as group representative
            duplicate_groups[group_id].append(i)
        else:
            # No similar items found, this is unique
            lsh.insert(i, minhash_sig)
            duplicate_groups[i] = [i]
    
    print("\nDeduplication results:")
    total_duplicates = 0
    
    # Keep only one representative from each group
    for group_id, group_items in duplicate_groups.items():
        if len(group_items) > 1:
            print(f"Found duplicate group with {len(group_items)} items: {group_items}")
            total_duplicates += len(group_items) - 1
        
        # Keep the first item in each group
        representative_id = group_items[0]
        if representative_id not in processed_ids:
            unique_data.append(data[representative_id])
            processed_ids.add(representative_id)
    
    print(f"\nOriginal records: {len(data)}")
    print(f"Duplicate records removed: {total_duplicates}")
    print(f"Unique records remaining: {len(unique_data)}")
    print(f"Deduplication rate: {total_duplicates/len(data)*100:.2f}%")
    
    # Save deduplicated data
    print(f"\nSaving deduplicated data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)
    
    print("Deduplication completed!")
    
    return unique_data

def verify_similarity(data, signatures, sample_size=5):
    """
    Verify similarity calculation by showing some examples
    """
    print(f"\nVerifying similarity calculation (showing {sample_size} examples):")
    
    items = list(signatures.items())
    for i in range(min(sample_size, len(items)-1)):
        id1, sig1 = items[i]
        id2, sig2 = items[i+1]
        
        similarity = sig1.jaccard(sig2)
        print(f"\nSimilarity between record {id1} and {id2}: {similarity:.3f}")
        print(f"Record {id1} user: {data[id1]['user'][:100]}...")
        print(f"Record {id2} user: {data[id2]['user'][:100]}...")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "filter_train_data.json"
    OUTPUT_FILE = "deduplication_train_data.json"
    SIMILARITY_THRESHOLD = 0.7  # 70% similarity
    NUM_PERM = 128  # Number of hash functions (higher = more accurate but slower)
    
    try:
        # Perform deduplication
        unique_data = deduplicate_json_data(
            INPUT_FILE, 
            OUTPUT_FILE, 
            SIMILARITY_THRESHOLD, 
            NUM_PERM
        )
        
        print(f"\nDeduplication successful! Check {OUTPUT_FILE} for results.")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{INPUT_FILE}'")
        print("Please make sure the file exists in the current directory.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{INPUT_FILE}'")
        print("Please check that the file contains valid JSON data.")
    except Exception as e:
        print(f"Error during deduplication: {str(e)}")