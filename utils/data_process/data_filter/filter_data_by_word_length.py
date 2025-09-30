import json

def filter_by_length(input_file, output_file, min_user_chars=50, min_assistant_chars=100):
    """
    Filter JSON data based on Chinese character length requirements.
    Both user and assistant must meet minimum length requirements.
    
    Args:
        input_file: Input JSON file path
        output_file: Output JSON file path
        min_user_chars: Minimum characters for user field (default: 50)
        min_assistant_chars: Minimum characters for assistant field (default: 100)
    """
    
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Original data count: {len(data)}")
        
        # Filter data
        filtered_data = []
        stats = {
            'total': len(data),
            'user_too_short': 0,
            'assistant_too_short': 0,
            'both_too_short': 0,
            'passed': 0
        }
        
        for item in data:
            user_text = item.get('user', '')
            assistant_text = item.get('assistant', '')
            
            user_len = len(user_text)
            assistant_len = len(assistant_text)
            
            user_valid = user_len >= min_user_chars
            assistant_valid = assistant_len >= min_assistant_chars
            
            # Track statistics
            if not user_valid and not assistant_valid:
                stats['both_too_short'] += 1
            elif not user_valid:
                stats['user_too_short'] += 1
            elif not assistant_valid:
                stats['assistant_too_short'] += 1
            else:
                stats['passed'] += 1
                filtered_data.append(item)
            
            # Print details for debugging (optional)
            if not (user_valid and assistant_valid):
                print(f"Filtered out - User: {user_len} chars, Assistant: {assistant_len} chars")
        
        # Save filtered data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        # Print results
        print(f"\n=== Filtering Results ===")
        print(f"Original records: {stats['total']}")
        print(f"Records passed: {stats['passed']}")
        print(f"User too short: {stats['user_too_short']}")
        print(f"Assistant too short: {stats['assistant_too_short']}")
        print(f"Both too short: {stats['both_too_short']}")
        print(f"Filtered out total: {stats['total'] - stats['passed']}")
        print(f"Pass rate: {stats['passed']/stats['total']*100:.1f}%")
        print(f"\nFiltered data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error: {e}")

def check_sample_lengths(input_file, sample_size=5):
    """
    Check lengths of first few samples to understand your data
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"=== Sample Length Check (First {sample_size} records) ===")
        for i, item in enumerate(data[:sample_size]):
            user_len = len(item.get('user', ''))
            assistant_len = len(item.get('assistant', ''))
            print(f"Record {i+1}: User={user_len} chars, Assistant={assistant_len} chars")
            
    except Exception as e:
        print(f"Error checking samples: {e}")

# Example usage
if __name__ == "__main__":
    input_file = "raw_train_data_113552.json"  # Replace with your file path
    output_file = "train_data_by_length_jusdgment.json"
    
    # First, check some samples to understand your data
    check_sample_lengths(input_file)
    
    # Then filter the data
    filter_by_length(input_file, output_file, min_user_chars=200, min_assistant_chars=500)