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

def select_best_answer(answers):
    """
    根据策略选择最佳答案：字数优先（>50字符） + 点赞数辅助
    """
    # 过滤掉内容长度小于50字符的回复
    valid_answers = [ans for ans in answers if len(ans.get('content', '').strip()) >= 50]
    
    if not valid_answers:
        # 如果没有符合条件的长回复，则选择最长的那个
        if answers:
            valid_answers = [max(answers, key=lambda x: len(x.get('content', '')))]
        else:
            return None
    
    # 按策略排序：字数优先，点赞数辅助，评论数最后
    def sort_key(answer):
        content_length = len(answer.get('content', '').strip())
        zan_count = int(answer.get('zan', 0))
        comment_count = int(answer.get('comment_count', 0))
        
        return (content_length, zan_count, comment_count)
    
    # 降序排序
    sorted_answers = sorted(valid_answers, key=sort_key, reverse=True)
    
    return sorted_answers[0]

def process_qa_file(input_file, output_file):
    """
    处理问答JSON文件，转换为指定格式
    一个JSON文件包含多个问答对象
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是单个问答对象，转换为列表
        if isinstance(data, dict):
            data = [data]
        
        result = []
        processed_count = 0
        skipped_count = 0
        
        for item in data:
            try:
                # 提取问题内容 - user就是ques_info下的content
                ques_info = item.get('ques_info', {})
                user_message = ques_info.get('content', '')
                
                # 清理用户问题
                user_message = clean_text(user_message)
                
                # 获取回答列表
                answers = item.get('answers_info', [])
                
                if not answers:
                    print(f"跳过没有回答的问题")
                    skipped_count += 1
                    continue
                
                # 选择最佳答案 - assistant就是answers_info下最佳的content
                best_answer = select_best_answer(answers)
                
                if not best_answer:
                    print(f"跳过没有有效回答的问题")
                    skipped_count += 1
                    continue
                
                assistant_reply = clean_text(best_answer.get('content', ''))
                
                if not user_message or not assistant_reply:
                    print(f"跳过内容为空的问题")
                    skipped_count += 1
                    continue
                
                # 添加到结果
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
                skipped_count += 1
                continue
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成!")
        print(f"成功处理: {processed_count} 条数据")
        print(f"跳过: {skipped_count} 条数据")
        print(f"输出文件: {output_file}")
        
        return result
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: JSON格式不正确 - {e}")
        return None
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return None

def process_multiple_files(input_files, output_file):
    """
    处理多个JSON文件并合并结果
    """
    all_results = []
    
    for input_file in input_files:
        print(f"\n正在处理文件: {input_file}")
        result = process_qa_file(input_file, f"temp_{input_file}")
        if result:
            all_results.extend(result)
    
    # 保存合并结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有文件处理完成!")
    print(f"总计处理: {len(all_results)} 条数据")
    print(f"合并输出文件: {output_file}")

# 使用示例
if __name__ == "__main__":
    # 单文件处理
    input_file = "ques_ans1.json"
    output_file = "CPQA_format.json"
    
    print("开始处理问答数据...")
    process_qa_file(input_file, output_file)