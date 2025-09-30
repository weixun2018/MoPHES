import json
import re

def clean_text(text):
    """
    清理文本中的多余换行符和空格
    """
    if not text:
        return ""
    
    # 去除首尾空白
    text = text.strip()
    
    # 将多个连续的换行符替换为单个空格
    text = re.sub(r'\n+', ' ', text)
    
    # 将多个连续的空格替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    return text

def format_emollm_data(input_file, output_file="EmoLLM_format.json"):
    """
    将EmoLLM格式数据转换为标准格式
    prompt -> user
    completion -> assistant
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是单个对象，转换为列表
        if isinstance(data, dict):
            data = [data]
        
        result = []
        processed_count = 0
        
        for item in data:
            try:
                # 提取prompt和completion
                prompt = item.get('prompt', '')
                completion = item.get('completion', '')
                
                # 清理文本
                user_message = clean_text(prompt)
                assistant_reply = clean_text(completion)
                
                # 转换格式
                result.append({
                    "user": user_message,
                    "assistant": assistant_reply
                })
                
                processed_count += 1
                
                # 每处理1000条数据打印进度
                if processed_count % 1000 == 0:
                    print(f"已处理: {processed_count} 条数据")
                    
            except Exception as e:
                print(f"处理单条数据时出错: {e}")
                continue
        
        # 保存结果到EmoLLM_format.json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成!")
        print(f"成功处理: {processed_count} 条数据")
        print(f"输出文件: {output_file}")
        
        return processed_count
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return 0
    except json.JSONDecodeError as e:
        print(f"错误: JSON格式不正确 - {e}")
        return 0
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return 0

if __name__ == "__main__":
    input_file = "single_turn_dataset.json"
    
    print("开始格式化EmoLLM数据...")
    total_count = format_emollm_data(input_file)
    
    print(f"\n最终结果: 共 {total_count} 条数据已保存到 EmoLLM_format.json")