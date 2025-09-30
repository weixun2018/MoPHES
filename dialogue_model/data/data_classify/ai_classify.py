"""
Author: knull-cc
Date: 2025-08-01
Description: AI classification script for psychological counseling data topic categorization.
This script processes single-turn psychological counseling data and classifies them into topic categories.

Input format for single_turn_data:
[
    {
        "user": "user message", 
        "assistant": "assistant reply"
    }, 
    ...
]

Output format:
[
    {
        "conversation": {
            "user": "user message", 
            "assistant": "assistant reply"
        }, 
        "topic_category": "E"
    },
    ...
]
"""

import json
import requests
import time
import concurrent.futures
import hashlib
from typing import Set
import os
from typing import List, Dict, Any, Union, Optional

# Configurable constants
INPUT_FILE = "deduplication_train_data.json"
MAX_WORKERS = 50  # Number of parallel processing threads
API_BASE_URL = "https://api.openai.com/v1/chat/completions"  # API base URL
API_KEY = "sk-x"  # API key

# System Prompt for Classification
SYSTEM_PROMPT = """You are a professional psychological counseling data classification expert responsible for categorizing psychological counseling-related content. Please evaluate according to the following criteria:

## 1.Classification Categories
Please classify the input text content into one of the following 7 topic categories:

**A - Emotional and Behavioral**
- Anxiety, depression, emotional regulation difficulties, procrastination, addiction, compulsive behaviors, sleep problems
**B - Academic and Career**
- Study pressure, exam anxiety, major selection, employment anxiety, workplace adaptation, career planning
**C - Interpersonal and Family**
- Romantic conflicts, family disputes, friendship issues, communication barriers, marital troubles
**D - Personal Growth**
- Self-awareness, goal setting, confidence building, identity formation, value conflicts
**E - Lifestyle and Habits**
- Irregular sleep schedules, dietary issues, internet addiction, life rhythm management, healthy habits
**F - Life Adjustment**
- School transitions, graduation, relocation, environmental changes, life milestone adaptation difficulties
**G - Therapy and Coping**
- Seeking psychological counseling, therapy selection, emotional regulation techniques, self-help resource access

## 2.Classification Requirements
1. **Single Classification**: Each text segment can only belong to one most matching topic category
2. **Core Theme**: Classify based on the core issue and main focus of the text
3. **Priority Rule**: If the text contains multiple themes, select the most prominent and primary theme

## 3.Input and Output Format
**Input text**: `{text_to_be_classified}`
**Output format**: `[CLASSIFICATION_LABEL]`
Where classification label is one of: [A], [B], [C], [D], [E], [F], [G]

---
**Note**: Please provide only the classification result without explanation or reasoning."""

def get_item_hash(item: Dict) -> str:
    """Generate hash for data item to track processing progress"""
    item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(item_str.encode('utf-8')).hexdigest()

def load_processed_hashes(progress_file: str) -> Set[str]:
    """Load processed item hashes from progress file"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                return set(progress_data.get('processed_hashes', []))
        except Exception as e:
            print(f"Warning: Cannot read progress file {progress_file}: {e}")
    return set()

def save_progress(progress_file: str, processed_hashes: Set[str], results: List[Dict]):
    """Save processing progress to file"""
    progress_data = {
        'processed_hashes': list(processed_hashes),
        'processed_count': len(processed_hashes),
        'classified_count': len(results),
        'timestamp': time.time()
    }
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Cannot save progress file: {e}")

def load_existing_results(output_file: str) -> List[Dict]:
    """Load existing classification results"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Cannot read existing result file {output_file}: {e}")
    return []

def call_openai_api(prompt_content: str, system_prompt: str) -> Optional[str]:
    """
    Call OpenAI API for topic classification
    
    Parameters:
    prompt_content (str): Prompt content submitted to API
    system_prompt (str): System prompt defining classification criteria
    
    Returns:
    str or None: Classification label (A-G) or None if failed
    """
    url = f"{API_BASE_URL}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        "model": "gpt-4o-mini",
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        # Extract classification label from response
        if content.startswith('[') and content.endswith(']'):
            label = content[1:-1].strip()
            if label in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                return label
        
        print(f"Invalid classification response: {content}")
        return None
        
    except Exception as e:
        print(f"API call error: {str(e)}")
        return None

def process_single_turn_data(item: Dict[str, str], system_prompt: str) -> Optional[Dict]:
    """
    Process single-turn data and classify
    
    Parameters:
    item (dict): Single-turn data item {"user": "...", "assistant": "..."}
    system_prompt (str): System prompt defining classification criteria
    
    Returns:
    dict or None: Classified item in new format or None if failed
    """
    user_message = item.get('user', '')
    assistant_message = item.get('assistant', '')
    
    # Skip empty messages
    if not user_message or not assistant_message:
        return None
    
    prompt_content = f"User: {user_message}\nAssistant: {assistant_message}"
    classification = call_openai_api(prompt_content, system_prompt)
    
    if classification:
        return {
            "conversation": {
                "user": user_message,
                "assistant": assistant_message
            },
            "topic_category": classification
        }
    return None

def process_data(input_file: str, output_file: str) -> List[Dict]:
    """Main data processing function"""
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            all_data = json.load(file)
        
        print(f"Total records to process: {len(all_data)}")
        
        # Setup progress tracking
        base_name = os.path.splitext(output_file)[0]
        progress_file = f"{base_name}_progress.json"
        
        processed_hashes = load_processed_hashes(progress_file)
        results = load_existing_results(output_file)
        
        print(f"Already processed items: {len(processed_hashes)}")
        print(f"Items already classified: {len(results)}")
        
        # Filter unprocessed items
        unprocessed_items = []
        for item in all_data:
            item_hash = get_item_hash(item)
            if item_hash not in processed_hashes:
                unprocessed_items.append((item, item_hash))
        
        print(f"Remaining items to process: {len(unprocessed_items)}")
        
        if not unprocessed_items:
            print("All items have been processed!")
            return results
        
        save_interval = 50
        current_processed = 0
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_item = {
                    executor.submit(process_single_turn_data, item, SYSTEM_PROMPT): (item, item_hash)
                    for item, item_hash in unprocessed_items
                }
                
                for future in concurrent.futures.as_completed(future_to_item):
                    item, item_hash = future_to_item[future]
                    current_processed += 1
                    total_processed = len(processed_hashes) + current_processed
                    
                    try:
                        classified_item = future.result()
                        if classified_item:
                            results.append(classified_item)
                            category = classified_item['topic_category']
                            print(f"Progress: {total_processed}/{len(all_data)} - Classified as [{category}]")
                        else:
                            print(f"Progress: {total_processed}/{len(all_data)} - Classification failed")
                        
                        processed_hashes.add(item_hash)
                        
                        if current_processed % save_interval == 0:
                            save_progress(progress_file, processed_hashes, results)
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(results, f, ensure_ascii=False, indent=4)
                            print(f"Saved intermediate results ({total_processed}/{len(all_data)})")
                            
                    except Exception as e:
                        print(f"Error processing data: {str(e)}")
                        processed_hashes.add(item_hash)
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user! Saving processed data...")
        except Exception as e:
            print(f"\nError occurred during processing: {str(e)}, saving processed data...")
        finally:
            save_progress(progress_file, processed_hashes, results)
            if results:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                print(f"\nProcessed {len(processed_hashes)}/{len(all_data)} records")
                print(f"{len(results)} records successfully classified and saved to {output_file}")
                print(f"Progress information saved to {progress_file}")
                
                # Print classification statistics
                category_stats = {}
                for item in results:
                    category = item['topic_category']
                    category_stats[category] = category_stats.get(category, 0) + 1
                
                print("\nClassification Statistics:")
                for category in sorted(category_stats.keys()):
                    print(f"  Category [{category}]: {category_stats[category]} items")
            else:
                print("\nNo records were successfully classified")
        
        return results
            
    except FileNotFoundError:
        print(f"Error: File not found {input_file}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return []

# Main program
if __name__ == "__main__":
    try:
        base_name = os.path.splitext(INPUT_FILE)[0]
        output_file = f"{base_name}_classified.json"
        print(f"\nStarting topic classification for file: {INPUT_FILE}")
        print(f"Results will be saved to: {output_file}")
        process_data(INPUT_FILE, output_file)
    except FileNotFoundError:
        print(f"Error: File not found {INPUT_FILE}")