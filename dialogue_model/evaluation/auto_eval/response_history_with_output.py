"""
心理咨询模型评估脚本 - 模拟历史模式
使用模型生成的输出作为对话历史进行评估
"""

import json
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import jieba
from rouge import Rouge
import gc

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_testset(test_path):
    """加载测试数据，为模拟历史模式准备数据结构"""
    processed_data = []
    system_prompt = "你是一位专业的心理咨询师，我有一些心理问题，请你用专业的知识帮我解决。请只生成一轮回复，不要继续对话。"

    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        for conv_idx, conversation in enumerate(data):
            messages = conversation["messages"]

            # 确保消息数量足够
            if len(messages) < 2:
                continue

            # 将每轮对话作为独立的评估项
            conversation_turns = []
            for i in range(0, len(messages), 2):
                if i + 1 >= len(messages):
                    break

                current_question = messages[i]["content"]
                current_answer = messages[i+1]["content"]

                turn_item = {
                    'turn_id': i // 2,
                    'question': current_question,
                    'reference_answer': current_answer,
                    'type': 'qa'
                }
                conversation_turns.append(turn_item)

            # 将完整对话添加到处理数据中
            processed_item = {
                'id': len(processed_data),
                'conversation_id': conv_idx,
                'system_prompt': system_prompt,
                'turns': conversation_turns,
                'original_messages': messages
            }
            processed_data.append(processed_item)

    return processed_data

def simulate_conversation_with_model(model, tokenizer, conversation_data, batch_size=80):
    """使用模型模拟完整对话过程，生成历史并评估"""
    results = []
    
    print(f"开始模拟 {len(conversation_data)} 个对话...")
    
    # 批量处理对话
    for batch_start in tqdm(range(0, len(conversation_data), batch_size), desc="模拟对话批次"):
        batch_conversations = conversation_data[batch_start:batch_start + batch_size]
        
        # 为当前批次的每个对话维护独立的历史
        batch_histories = {}
        for conv in batch_conversations:
            batch_histories[conv['id']] = []
        
        # 确定最大轮次数
        max_turns = max(len(conv['turns']) for conv in batch_conversations)
        
        # 逐轮次处理
        for turn_idx in range(max_turns):
            # 收集当前轮次的所有有效对话
            current_turn_data = []
            
            for conv in batch_conversations:
                if turn_idx < len(conv['turns']):
                    current_turn_data.append((conv, turn_idx))
            
            if not current_turn_data:
                continue
            
            # 准备批量messages
            batch_messages_list = []
            batch_turn_info = []
            
            for conv, turn_id in current_turn_data:
                turn = conv['turns'][turn_id]
                current_question = turn['question']
                
                # 构建包含历史的messages
                messages = [{"role": "system", "content": conv['system_prompt']}]
                
                # 添加模型生成的历史对话
                for prev_q, prev_a in batch_histories[conv['id']]:
                    messages.append({"role": "user", "content": prev_q})
                    messages.append({"role": "assistant", "content": prev_a})
                
                # 添加当前问题
                messages.append({"role": "user", "content": current_question})
                
                batch_messages_list.append(messages)
                batch_turn_info.append((conv, turn_id, turn))
            
            # 批量生成响应
            if batch_messages_list:
                batch_responses = batch_generate_responses(model, tokenizer, batch_messages_list)
                
                # 处理生成的响应并更新历史
                for i, ((conv, turn_id, turn), response) in enumerate(zip(batch_turn_info, batch_responses)):
                    current_question = turn['question']
                    
                    # 将当前轮次加入历史
                    batch_histories[conv['id']].append((current_question, response))
                    
                    # 保存结果
                    result = {
                        "id": f"{conv['id']}_{turn_id}",
                        "conversation_id": conv['id'],
                        "turn_id": turn_id,
                        "messages": batch_messages_list[i],
                        "generated_response": response,
                        "reference": turn['reference_answer'],
                        "type": turn['type']
                    }
                    results.append(result)
    
    return results

def batch_generate_responses(model, tokenizer, batch_messages_list):
    """批量生成响应"""
    model.eval()
    with torch.no_grad():
        # 使用官方chat template批量处理
        batch_prompts = []
        for messages in batch_messages_list:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_prompts.append(prompt_text)

        # 批量编码
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        # 批量生成
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.2,
            top_p=0.7,
            use_cache=False
        )

        # 批量解码和清理
        cleaned_responses = []
        for j, output in enumerate(outputs):
            input_length = len(inputs.input_ids[j])
            generated_tokens = output[input_length:]
            response_only = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            cleaned = clean_generated_response(response_only)
            cleaned_responses.append(cleaned)

        return cleaned_responses

def save_model_outputs(results, output_path):
    """保存模型输出为指定格式"""
    # 按对话重新组织结果
    conversations = defaultdict(list)
    for result in results:
        conversations[result['conversation_id']].append(result)
    
    # 构建输出格式
    output_conversations = []
    for conv_id, conv_results in conversations.items():
        # 按turn_id排序
        conv_results.sort(key=lambda x: x['turn_id'])
        
        # 构建完整的messages，包含模型生成的历史
        full_messages = []
        if conv_results:
            # 添加system prompt
            first_messages = conv_results[0]['messages']
            for msg in first_messages:
                if msg['role'] == 'system':
                    full_messages.append(msg)
                    break
            
            # 逐轮添加用户问题和助手回复
            for result in conv_results:
                # 找到最后的用户消息（当前问题）
                user_msg = None
                for msg in reversed(result['messages']):
                    if msg['role'] == 'user':
                        user_msg = msg
                        break
                
                if user_msg:
                    full_messages.append(user_msg)
                    full_messages.append({
                        "role": "assistant", 
                        "content": result['generated_response']
                    })
        
        if full_messages:
            output_conversations.append({"messages": full_messages})
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_conversations, f, ensure_ascii=False, indent=2)
    
    print(f"模型输出已保存至: {output_path}")

def calculate_bleu(prediction, reference):
    """计算BLEU和ROUGE分数"""
    prediction = str(prediction).strip().replace(" ", "")
    reference = str(reference).strip().replace(" ", "")

    prediction_text = " ".join(jieba.cut(prediction))
    reference_text = " ".join(jieba.cut(reference))

    try:
        weights = [
            (1.0, 0.0, 0.0, 0.0),      # BLEU-1
            (1./2., 1./2., 0.0, 0.0),  # BLEU-2
            (1./3., 1./3., 1./3., 0.0),# BLEU-3
            (1./4., 1./4., 1./4., 1./4.)# BLEU-4
        ]

        scores = {}
        for n, weight in enumerate(weights, start=1):
            score = sentence_bleu(
                references=[reference_text.split()],
                hypothesis=prediction_text.split(),
                weights=weight,
                smoothing_function=SmoothingFunction().method1
            ) * 100
            scores[f'bleu-{n}'] = score

        rouge = Rouge()
        rouge_scores = rouge.get_scores(prediction_text, reference_text, avg=True)

        for key, value in rouge_scores.items():
            scores[key] = value['f'] * 100

        return scores

    except Exception as e:
        print(f"Error calculating scores: {str(e)}")
        return {
            **{f'bleu-{n}': 0.0 for n in range(1, 5)},
            **{'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
        }

def evaluate_results(results):
    """评估生成结果"""
    metrics = defaultdict(float)
    valid_count = 0

    for item in results:
        if item['type'] != 'qa':
            continue

        valid_count += 1
        reference = item['reference']
        response = item['generated_response']

        scores = calculate_bleu(response, reference)
        for metric, value in scores.items():
            metrics[metric] += value

    # 计算平均值
    if valid_count > 0:
        total_score = 0.0
        for metric in metrics:
            metrics[metric] /= valid_count
            if 'bleu' in metric:
                total_score += metrics[metric] * 0.125
            else:
                total_score += metrics[metric] * 0.167
        metrics['total_score'] = total_score

    print(f"\nValid sample count: {valid_count}")
    print("Metrics:", dict(metrics))

    return dict(metrics)

def clean_generated_response(response):
    """清理生成的响应"""
    cut_markers = ["<user>", "<s>", "<AI>"]
    positions = []
    for marker in cut_markers:
        pos = response.find(marker)
        if pos != -1:
            positions.append(pos)

    if positions:
        first_cut = min(positions)
        return response[:first_cut].strip()

    return response.strip()

def main():
    """主函数"""
    # ========== 配置部分 ==========
    base_model_path = "/content/MiniCPM4-0.5B"
    test_path = "/content/MPchat_tools/eval_data.json"
    
    # 要评估的模型列表 (None表示基础模型)
    models_to_evaluate = [
        {"name": "MiniCPM4-0.5B", "adapter_path": None},
        {"name": "MiniCPM4-0.5b-35k", "adapter_path": "/content/drive/MyDrive/MPchat/MiniCPM4-0.5b-35k-0802-1810"}
    ]
    
    BATCH_SIZE = 80
    # ===============================

    print(f"=== 模拟历史评估配置 ===")
    print(f"批处理大小: {BATCH_SIZE}")
    print(f"基础模型: {base_model_path}")
    print(f"要评估的模型数量: {len(models_to_evaluate)}")
    print("=" * 50)

    # 加载测试数据
    test_data = load_testset(test_path)
    print(f"测试数据加载完成，共 {len(test_data)} 个对话")
    
    # 计算总轮次数
    total_turns = sum(len(conv['turns']) for conv in test_data)
    print(f"总评估轮次: {total_turns}")

    all_metrics = {}

    # 逐个评估模型
    for model_config in models_to_evaluate:
        model_name = model_config["name"]
        adapter_path = model_config["adapter_path"]
        
        print(f"\n评估模型 {model_name} (模拟历史模式)...")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if adapter_path:
            # 加载微调模型
            model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        else:
            # 使用基础模型
            model = base_model

        # 使用模拟历史模式生成结果
        results = simulate_conversation_with_model(model, tokenizer, test_data, batch_size=BATCH_SIZE)
        
        # 保存模型输出
        output_filename = f'simulate_history_outputs_{model_name}.json'
        save_model_outputs(results, output_filename)
        
        # 评估结果
        metrics = evaluate_results(results)
        all_metrics[model_name] = metrics

        # 释放内存
        del model
        del base_model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"{model_name} 模拟历史评估完成并释放")

    # 打印比较结果
    print("\n" + "=" * 80)
    print("模拟历史模式评估结果比较")
    print("=" * 80)
    
    metrics_list = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'rouge-1', 'rouge-2', 'rouge-l', 'total_score']
    
    for metric in metrics_list:
        print(f"\n{metric}:")
        for model_name, model_metrics in all_metrics.items():
            print(f"  {model_name}: {model_metrics[metric]:.2f}")

    # 保存评估结果
    eval_output_filename = 'simulate_history_evaluation_results.json'
    with open(eval_output_filename, 'w', encoding='utf-8') as f:
        output_data = {
            "evaluation_mode": "simulate_history",
            "batch_size": BATCH_SIZE,
            "total_conversations": len(test_data),
            "total_turns": total_turns,
            "models_evaluated": list(all_metrics.keys()),
            "metrics": all_metrics
        }
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n模拟历史评估结果已保存至: {eval_output_filename}")
    print("=" * 80)

if __name__ == "__main__":
    main()