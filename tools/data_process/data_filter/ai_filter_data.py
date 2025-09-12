"""
Author: knull-cc
Date: 2025-03-15
Description: Unified data processing script for evaluating and filtering single-turn and multi-turn psychological counseling data.
This script can automatically detect data formats:
Input format for single_turn_data:
[
    {
        "user": "user message", 
        "assistant": "assistant reply"
    }, 
    ...
]
Input Format for multi_turn_data:
[
    {
        "messages": [
            {
                "role": "user", 
                "content": "user message"
            }, 
            {   
                "role": "assistant", 
                "content": "assistant reply"
            }, 
            ...
        ]
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
INPUT_FILE = "train_data_by_length.json"
MAX_WORKERS = 70  # Number of parallel processing threads
API_BASE_URL = "https://api.openai.com/v1/chat/completions"  # API base URL
API_KEY = "sk-proj-x"  # API key


# System Prompt
SYSTEM_PROMPT_SINGLE = """
You are a professional psychological counseling data review expert responsible for evaluating conversation quality. Please assess according to the following criteria:

## 1. Classification Categories
- **ACCEPT**: Content meets psychological counseling standards
- **REJECT**: Content does not meet requirements

## 2. Content Relevance (must match at least one category)
**Topic Scope**: Psychological counseling, mental health, emotional management, and personal development

**Specific Categories**:
- **Emotional and Behavioral**: Anxiety, depression, emotional regulation difficulties, procrastination, addiction, compulsive behaviors, sleep problems
- **Academic and Career**: Study pressure, exam anxiety, major selection, employment anxiety, workplace adaptation, career planning
- **Interpersonal and Family**: Romantic conflicts, family disputes, friendship issues, communication barriers, marital troubles
- **Personal Growth**: Self-awareness, goal setting, confidence building, identity formation, value conflicts
- **Lifestyle and Habits**: Irregular sleep schedules, dietary issues, internet addiction, life rhythm management, healthy habits
- **Life Adjustment**: School transitions, graduation, relocation, environmental changes, life milestone adaptation difficulties
- **Therapy and Coping**: Seeking psychological counseling, therapy selection, emotional regulation techniques, self-help resource access

## 3. Quality Requirements (must meet at least 3 criteria)
- **Constructive**: Provides specific advice or solution-oriented thinking
- **Empathetic**: Shows understanding and care for the counselee's situation
- **Logical**: Response is well-structured and organized
- **Practical**: Suggestions are actionable and implementable
- **Substantial**: Not superficial or dismissive responses

## 4. Prohibited Content (strictly enforced)
- Self-harm, self-injury, or suicidal content
- Politically sensitive, violent, or sexual content
- Discriminatory language or illegal activities
- Obvious advertising or marketing content
- Specific medication recommendations
- Content promoting dangerous behaviors

## 5. Evaluation Process
1. Read the input text carefully
2. Verify if content matches psychological counseling topics
3. Check if quality requirements are met
4. Confirm basic length requirements
5. Identify any prohibited content
6. Provide only the classification label

## 6. Input and Output Format
**Input text**: `{your_chinese_text}`

**Output format**: `[CLASSIFICATION_LABEL]`

---
*Note: Provide only the classification result without explanation or reasoning.*
"""

SYSTEM_PROMPT_MULTI = ""


def get_item_hash(item: Dict) -> str:
    item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(item_str.encode('utf-8')).hexdigest()

def load_processed_hashes(progress_file: str) -> Set[str]:
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                return set(progress_data.get('processed_hashes', []))
        except Exception as e:
            print(f"Warning: Cannot read progress file {progress_file}: {e}")
    return set()

def save_progress(progress_file: str, processed_hashes: Set[str], results: List[Dict]):
    progress_data = {
        'processed_hashes': list(processed_hashes),
        'processed_count': len(processed_hashes),
        'passed_count': len(results),
        'timestamp': time.time()
    }
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Cannot save progress file: {e}")

def load_existing_results(output_file: str) -> List[Dict]:
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Cannot read existing result file {output_file}: {e}")
    return []

def call_openai_api(prompt_content: str, system_prompt: str, input_data: Union[str, List[Dict[str, str]]]) -> bool:
    """
    Call OpenAI API for data quality evaluation
    
    Parameters:
    prompt_content (str): Prompt content submitted to API
    system_prompt (str): System prompt defining evaluation criteria
    input_data (str or list): Input data, can be string or message list
    
    Returns:
    bool: Whether passed evaluation
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
        content = result["choices"][0]["message"]["content"]
        return content.strip() == "[ACCEPT]"
    except Exception as e:
        print(f"API call error: {str(e)}")
        return False

def process_single_turn_data(item: Dict[str, str], system_prompt: str) -> bool:
    """
    Process single-turn data and evaluate
    
    Parameters:
    item (dict): Single-turn data item {"user": "...", "assistant": "..."}
    system_prompt (str): System prompt defining evaluation criteria
    
    Returns:
    bool: Whether passed evaluation
    """
    user_message = item.get('user', '')
    assistant_message = item.get('assistant', '')
    
    # Skip empty messages
    if not user_message or not assistant_message:
        return False
    
    prompt_content = f"请求者：{user_message} 支持者：{assistant_message}"
    return call_openai_api(prompt_content, system_prompt, item)

def process_multi_turn_data(item: Dict[str, List[Dict[str, str]]], system_prompt: str) -> bool:
    """
    Process multi-turn data and evaluate
    
    Parameters:
    item (dict): Multi-turn data item {"messages": [...]}
    system_prompt (str): System prompt defining evaluation criteria
    
    Returns:
    bool: Whether passed evaluation
    """
    messages = item.get('messages', [])
    
    # Skip empty message list
    if len(messages) < 2:
        return False
    
    conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    prompt_content = f"对话内容：\n{conversation}"
    return call_openai_api(prompt_content, system_prompt, messages)

def process_data(input_file: str, output_file: str) -> List[Dict]:
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            all_data = json.load(file)
        
        data_type = 'multi' if all_data and len(all_data) > 0 and 'messages' in all_data[0] else 'single'
        system_prompt = SYSTEM_PROMPT_MULTI if data_type == 'multi' else SYSTEM_PROMPT_SINGLE
        
        print(f"Detected data type: {data_type}")
        print(f"Total records to process: {len(all_data)}")
        
        # Setup progress tracking
        base_name = os.path.splitext(output_file)[0]
        progress_file = f"{base_name}_progress.json"
        
        processed_hashes = load_processed_hashes(progress_file)
        results = load_existing_results(output_file)
        
        print(f"Already processed items: {len(processed_hashes)}")
        print(f"Items already passed evaluation: {len(results)}")
        
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
                if data_type == 'single':
                    future_to_item = {
                        executor.submit(process_single_turn_data, item, system_prompt): (item, item_hash)
                        for item, item_hash in unprocessed_items
                    }
                else:
                    future_to_item = {
                        executor.submit(process_multi_turn_data, item, system_prompt): (item, item_hash)
                        for item, item_hash in unprocessed_items
                    }
                
                for future in concurrent.futures.as_completed(future_to_item):
                    item, item_hash = future_to_item[future]
                    current_processed += 1
                    total_processed = len(processed_hashes) + current_processed
                    
                    try:
                        if future.result():
                            results.append(item)
                            print(f"Progress: {total_processed}/{len(all_data)} - Evaluation passed")
                        else:
                            print(f"Progress: {total_processed}/{len(all_data)} - Evaluation failed")
                        
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
                print(f"{len(results)} records passed AI evaluation and saved to {output_file}")
                print(f"Progress information saved to {progress_file}")
            else:
                print("\nNo qualifying records found")
        
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
        output_file = f"{base_name}_output.json"
        print(f"\nProcessing file: {INPUT_FILE}")
        process_data(INPUT_FILE, output_file)
    except FileNotFoundError:
        print(f"Error: File not found {INPUT_FILE}")