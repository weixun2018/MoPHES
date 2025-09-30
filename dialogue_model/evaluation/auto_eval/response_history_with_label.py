"""
心理咨询模型评估脚本 - 使用官方历史携带方式
生成模型输出并进行BLEU/ROUGE评估
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
    """加载测试数据并构造messages格式"""
    processed_data = []
    system_prompt = "你是一位专业的心理咨询师，我有一些心理问题，请你用专业的知识帮我解决。请只生成一轮回复，不要继续对话。"

    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        for conv_idx, conversation in enumerate(data):
            messages = conversation["messages"]

            # 确保消息数量足够
            if len(messages) < 2:
                continue

            # 构建完整的messages格式
            for i in range(0, len(messages), 2):
                if i + 1 >= len(messages):
                    break

                current_question = messages[i]["content"]
                current_answer = messages[i+1]["content"]

                # 构造包含历史的messages
                chat_messages = [{"role": "system", "content": system_prompt}]
                
                # 添加历史对话
                for j in range(0, i, 2):
                    if j + 1 < len(messages):
                        chat_messages.append({"role": "user", "content": messages[j]["content"]})
                        chat_messages.append({"role": "assistant", "content": messages[j+1]["content"]})
                
                # 添加当前问题
                chat_messages.append({"role": "user", "content": current_question})

                processed_item = {
                    'id': len(processed_data),
                    'conversation_id': conv_idx,
                    'turn_id': i // 2,
                    'messages': chat_messages,
                    'reference_answer': current_answer,
                    'type': 'qa'
                }
                processed_data.append(processed_item)

    return processed_data

def batch_generate(model, tokenizer, test_data, batch_size=150):
    """批量生成响应"""
    results = []
    
    for i in tqdm(range(0, len(test_data), batch_size), desc="Processing batches"):
        batch = test_data[i:i+batch_size]
        
        # 构建messages列表
        batch_messages_list = [item['messages'] for item in batch]
        
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

        # 组装结果
        for idx in range(len(batch_messages_list)):
            result = {
                "id": batch[idx]['id'],
                "messages": batch_messages_list[idx],
                "generated_response": cleaned_responses[idx],
                "reference": batch[idx]['reference_answer'],
                "type": batch[idx].get('type', 'unknown')
            }
            results.append(result)

    return results

def save_model_outputs(results, output_path):
    """保存模型输出为指定格式"""
    output_conversations = []
    
    # 直接为每个结果创建一个对话条目
    for result in results:
        # 构建完整的对话messages，包含模型输出
        full_messages = []
        
        # 添加除了system消息之外的所有历史消息
        for msg in result['messages']:
            if msg['role'] != 'system':  # 跳过system消息
                full_messages.append(msg)
        
        # 添加模型生成的回复
        full_messages.append({
            "role": "assistant",
            "content": result['generated_response']
        })
        
        # 为每个结果创建一个对话条目
        output_conversations.append({
            "messages": full_messages
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_conversations, f, ensure_ascii=False, indent=2)
    
    print(f"模型输出已保存至: {output_path}")
    print(f"共保存了 {len(output_conversations)} 个对话")

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
    
    BATCH_SIZE = 150
    # ===============================

    print(f"=== 评估配置 ===")
    print(f"批处理大小: {BATCH_SIZE}")
    print(f"基础模型: {base_model_path}")
    print(f"要评估的模型数量: {len(models_to_evaluate)}")
    print("=" * 50)

    # 加载测试数据
    test_data = load_testset(test_path)
    print(f"测试数据加载完成，共 {len(test_data)} 条")

    all_metrics = {}

    # 逐个评估模型
    for model_config in models_to_evaluate:
        model_name = model_config["name"]
        adapter_path = model_config["adapter_path"]
        
        print(f"\n评估模型 {model_name}...")
        
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

        # 生成结果
        results = batch_generate(model, tokenizer, test_data, batch_size=BATCH_SIZE)
        
        # 保存模型输出
        output_filename = f'model_outputs_{model_name}.json'
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
        print(f"{model_name} 评估完成并释放")

    # 打印比较结果
    print("\n" + "=" * 80)
    print("评估结果比较")
    print("=" * 80)
    
    metrics_list = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'rouge-1', 'rouge-2', 'rouge-l', 'total_score']
    
    for metric in metrics_list:
        print(f"\n{metric}:")
        for model_name, model_metrics in all_metrics.items():
            print(f"  {model_name}: {model_metrics[metric]:.2f}")

    # 保存评估结果
    eval_output_filename = 'evaluation_results.json'
    with open(eval_output_filename, 'w', encoding='utf-8') as f:
        output_data = {
            "batch_size": BATCH_SIZE,
            "total_samples": len(test_data),
            "models_evaluated": list(all_metrics.keys()),
            "metrics": all_metrics
        }
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n评估结果已保存至: {eval_output_filename}")
    print("=" * 80)

if __name__ == "__main__":
    main()