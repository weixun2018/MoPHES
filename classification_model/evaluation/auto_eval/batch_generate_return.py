"""
Author: Modified from knull-cc
Date: 2025-08-08
Description: 心理健康诊断数据处理脚本，用于诊断对话数据中的抑郁和焦虑水平
输入格式：
[
    {
        "user": "用户消息", 
        "assistant": "助手回复"
    }, 
    ...
]
输出格式：
[
    {
        "messages": [
            {
                "role": "system", 
                "content": "system_prompt"
            },
            {
                "role": "user", 
                "content": "user_messages"
            },
            {
                "role": "assistant", 
                "content": "{\"depression\": X, \"anxiety\": Y}"
            }
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

# 可配置常量
INPUT_FILE = "eval_data_by_length_jusdgment.json"
MAX_WORKERS = 50  # 并行处理线程数
API_BASE_URL = "https://api.openai.com/v1/chat/completions"  # API基础URL
API_KEY = "sk-x"  # API密钥

# 系统提示词
SYSTEM_PROMPT = """

你是一个专业的心理健康诊断工具，需要基于用户文本内容诊断抑郁和焦虑水平。严格按照以下标准进行诊断。

## 诊断标准

### 抑郁水平 (0-3)
- **0级 - 无症状**: 情绪正常，积极表达，对活动有兴趣
- **1级 - 轻度**: 偶有负面情绪，轻微兴趣下降，整体功能正常
- **2级 - 中度**: 持续负面情绪，明显兴趣丧失，影响日常生活
- **3级 - 重度**: 严重绝望感、无价值感，功能严重受损，可能有自伤想法

### 焦虑水平 (0-3)
- **0级 - 无症状**: 放松平静，无过度担心
- **1级 - 轻度**: 轻微担心或紧张，可自我调节
- **2级 - 中度**: 明显焦虑症状，中度影响日常生活
- **3级 - 重度**: 严重焦虑症状，强烈回避行为，功能显著受损

## 诊断样例

### 样例1
**用户文本**: "今天天气不错，和朋友一起去了公园，心情很好。最近工作虽然忙但很充实，期待周末的聚会。"
**诊断**: {"depression": 0, "anxiety": 0}
**理由**: 积极情绪，社交正常，对活动期待

### 样例2
**用户文本**: "最近总是担心考试，晚上经常失眠，心跳很快。虽然复习了但还是很紧张，怕考不好。"
**诊断**: {"depression": 0, "anxiety": 2}
**理由**: 无抑郁症状，但有明显焦虑症状影响睡眠

### 样例3
**用户文本**: "感觉什么都没意思，不想见任何人。每天都很累，对以前喜欢的事情也提不起兴趣了。觉得自己很没用。"
**诊断**: {"depression": 2, "anxiety": 1}
**理由**: 明显抑郁症状（兴趣丧失、疲劳、自我价值感低），轻度焦虑

## 重要原则
1. 仅基于文本内容诊断，不做推测
2. 关注症状的严重程度和功能影响
3. 区分正常情绪波动和病理性症状
4. 输出必须为标准JSON格式，不添加任何解释

## 输出格式
严格输出JSON格式，不包含任何其他内容：
{"depression": [0-3], "anxiety": [0-3]}

"""

def get_item_hash(item: Dict) -> str:
    """生成数据项的哈希值"""
    item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(item_str.encode('utf-8')).hexdigest()

def load_processed_hashes(progress_file: str) -> Set[str]:
    """加载已处理的数据哈希"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                return set(progress_data.get('processed_hashes', []))
        except Exception as e:
            print(f"警告: 无法读取进度文件 {progress_file}: {e}")
    return set()

def save_progress(progress_file: str, processed_hashes: Set[str], results: List[Dict]):
    """保存处理进度"""
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
        print(f"警告: 无法保存进度文件: {e}")

def load_existing_results(output_file: str) -> List[Dict]:
    """加载已有的结果文件"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"警告: 无法读取现有结果文件 {output_file}: {e}")
    return []

def call_openai_api(user_content: str) -> Optional[str]:
    """
    调用OpenAI API进行心理健康诊断
    
    Parameters:
    user_content (str): 用户输入内容
    
    Returns:
    str: API返回的诊断结果，格式为 {"depression": X, "anxiety": Y}
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
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "model": "gpt-4o-mini",
        "max_tokens": 50,
        "temperature": 0
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        # 验证返回的内容是否为有效的JSON格式
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            print(f"API返回的内容不是有效的JSON: {content}")
            return None
            
    except Exception as e:
        print(f"API调用错误: {str(e)}")
        return None

def process_single_turn_data(item: Dict[str, str]) -> Optional[Dict]:
    """
    处理单轮对话数据并诊断
    
    Parameters:
    item (dict): 单轮数据项 {"user": "...", "assistant": "..."}
    
    Returns:
    dict: 转换后的多轮格式数据，如果处理失败返回None
    """
    user_message = item.get('user', '').strip()
    assistant_message = item.get('assistant', '').strip()
    
    # 跳过空消息
    if not user_message or not assistant_message:
        return None
    
    # 调用API诊断用户消息
    evaluation_result = call_openai_api(user_message)
    if evaluation_result is None:
        return None
    
    # 构建新的消息格式
    converted_item = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": user_message
            },
            {
                "role": "assistant",
                "content": evaluation_result
            }
        ]
    }
    
    return converted_item

def process_data(input_file: str, output_file: str) -> List[Dict]:
    """处理数据文件"""
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            all_data = json.load(file)
        
        print(f"总共需要处理的记录数: {len(all_data)}")
        
        # 设置进度跟踪
        base_name = os.path.splitext(output_file)[0]
        progress_file = f"{base_name}_progress.json"
        
        processed_hashes = load_processed_hashes(progress_file)
        results = load_existing_results(output_file)
        
        print(f"已处理的项目数: {len(processed_hashes)}")
        print(f"已通过诊断的项目数: {len(results)}")
        
        # 过滤未处理的项目
        unprocessed_items = []
        for item in all_data:
            item_hash = get_item_hash(item)
            if item_hash not in processed_hashes:
                unprocessed_items.append((item, item_hash))
        
        print(f"剩余需要处理的项目数: {len(unprocessed_items)}")
        
        if not unprocessed_items:
            print("所有项目都已处理完成！")
            return results
        
        save_interval = 50  # 每处理50个项目保存一次
        current_processed = 0
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_item = {
                    executor.submit(process_single_turn_data, item): (item, item_hash)
                    for item, item_hash in unprocessed_items
                }
                
                for future in concurrent.futures.as_completed(future_to_item):
                    item, item_hash = future_to_item[future]
                    current_processed += 1
                    total_processed = len(processed_hashes) + current_processed
                    
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                            print(f"进度: {total_processed}/{len(all_data)} - 诊断通过")
                        else:
                            print(f"进度: {total_processed}/{len(all_data)} - 诊断失败")
                        
                        processed_hashes.add(item_hash)
                        
                        # 定期保存进度
                        if current_processed % save_interval == 0:
                            save_progress(progress_file, processed_hashes, results)
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(results, f, ensure_ascii=False, indent=4)
                            print(f"保存中间结果 ({total_processed}/{len(all_data)})")
                            
                    except Exception as e:
                        print(f"处理数据时出现错误: {str(e)}")
                        processed_hashes.add(item_hash)
        
        except KeyboardInterrupt:
            print("\n用户中断处理！正在保存已处理的数据...")
        except Exception as e:
            print(f"\n处理过程中发生错误: {str(e)}，正在保存已处理的数据...")
        finally:
            # 最终保存
            save_progress(progress_file, processed_hashes, results)
            if results:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                print(f"\n已处理 {len(processed_hashes)}/{len(all_data)} 条记录")
                print(f"{len(results)} 条记录通过AI诊断并保存到 {output_file}")
                print(f"进度信息已保存到 {progress_file}")
            else:
                print("\n没有找到合格的记录")
        
        return results
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except json.JSONDecodeError:
        print("错误: JSON格式无效")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return []

# 主程序
if __name__ == "__main__":
    try:
        base_name = os.path.splitext(INPUT_FILE)[0]
        output_file = f"{base_name}_mental_health_output.json"
        print(f"\n正在处理文件: {INPUT_FILE}")
        process_data(INPUT_FILE, output_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {INPUT_FILE}")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")