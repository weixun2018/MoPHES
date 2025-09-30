# -*- coding: utf-8 -*-
"""
心理咨询模型评估脚本（仅接受严格 JSON 输出；评测可复现）
输出：
1) all_model_outputs.json —— 所有模型的输出放在一个文件（含对话与标准答案）
2) 控制台直接 print 每个模型的分数（不再写 evaluation_metrics_summary.json）

评估分三部分：
1. 粗略准确率（严格相等）：分别计算抑郁/焦虑准确率（预测值==真值才算对）
2. P/R/F1：
   - 4 类（只看抑郁）weighted
   - 4 类（只看焦虑）weighted
   - 16 类（抑郁0-3 × 焦虑0-3）weighted
3. 归一化绝对误差评分（Dep/Anx/Total）
"""

import json
import torch
import re
import gc
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime
from transformers import StoppingCriteria, StoppingCriteriaList

# ============== 可调参数 ==============
APPEND_INSTRUCTION = (
    '\n\n请只输出一个JSON对象，格式为：{"depression": x, "anxiety": y}。'
    '其中 x 和 y 必须是整数，范围在 0 到 3 之间。'
    '不要输出示例范围，不要解释，不要添加代码块或额外符号。'
)
BASE_MAX_NEW_TOKENS = 30     # 严格JSON输出足够；也有助于抑制跑题
MAX_INPUT_LEN = 1024
GLOBAL_SEED = 1234           # 评测可复现
# ====================================


# ===============================
# 随机种子
# ===============================
def set_global_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===============================
# 数据加载
# ===============================
def load_testset(test_path):
    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 统一为 judgment 类型
    return [{"id": idx, "messages": item["messages"], "type": "judgment"} for idx, item in enumerate(data)]


# ===============================
# 仅严格 JSON 解析
# ===============================
def _normalize_keys(d):
    """允许中文键，统一返回英文键；只接受0-3的整数。"""
    def pick(dct, *keys):
        for k in keys:
            if k in dct:
                return dct[k]
        return None

    dep = pick(d, "depression", "抑郁")
    anx = pick(d, "anxiety", "焦虑")
    if dep is None or anx is None:
        return None

    # 必须是 int 且落在 0-3；允许字符串数字但要能转成 int
    try:
        dep = int(dep); anx = int(anx)
    except Exception:
        return None
    if not (0 <= dep <= 3 and 0 <= anx <= 3):
        return None
    return {"depression": dep, "anxiety": anx}


def _extract_first_balanced_json(s):
    """
    在字符串中扫描第一个配平的大括号 {...} 片段，尝试 json.loads。
    成功则返回 dict；否则返回 None。
    """
    if not s:
        return None
    # 快速路径：若本身就是纯 JSON
    st = s.strip()
    if st.startswith("{") and st.endswith("}"):
        try:
            obj = json.loads(st)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 扫描寻找第一个配平的 {...}
    start = None
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = s[start:i+1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        # 若此段不是合法 JSON，继续往后找
                        pass
    return None


def extract_json_from_response(response):
    """
    只接受严格 JSON：
      - 允许中文键（抑郁/焦虑），会规范化为英文键（depression/anxiety）
      - 只要能在文本中找到首个配平的大括号片段，并且能被 json.loads 就接受，否则 None
    """
    if not response:
        return None
    obj = _extract_first_balanced_json(response)
    if obj is None:
        return None
    return _normalize_keys(obj)


def get_last_assistant_content(messages):
    for msg in reversed(messages):
        if msg.get('role') == 'assistant':
            return msg.get('content')
    return None


def get_ground_truth_from_messages(messages):
    last = get_last_assistant_content(messages)
    return extract_json_from_response(last)


# ===============================
# 生成输入构造 / 清理
# ===============================
def clean_generated_response(response):
    # 这里不再做正则提取；保留原文，交给 extract_json_from_response 只认 JSON
    return response.strip() if response else ""


def build_messages_with_optional_system(messages, use_system=True):
    """
    构造输入：若 use_system=True 且样本含 system，则保留 system；总是保留首个 user。
    若 system 已经明确约束输出格式，则不再在 user 尾部附加 APPEND_INSTRUCTION。
    """
    sys_msg = next((m for m in messages if m.get("role") == "system"), None)
    usr_msg = next((m for m in messages if m.get("role") == "user"), None)
    out = []
    if use_system and sys_msg is not None:
        out.append({"role": "system", "content": sys_msg.get("content", "")})
    if usr_msg is not None:
        content = (usr_msg.get("content", "") or "")
        need_append = True
        if use_system and sys_msg is not None:
            sc = sys_msg.get("content", "")
            if ("输出格式" in sc) or ("只输出JSON" in sc) or ("JSON格式" in sc):
                need_append = False
        if need_append:
            content = content.rstrip() + APPEND_INSTRUCTION
        out.append({"role": "user", "content": content})
    return out


# ===============================
# 停止条件：JSON 大括号配平即停
# ===============================
class JsonBraceStopping(StoppingCriteria):
    """当新增文本中第一次出现 '{' 与 '}' 数量相等时停止。"""
    def __init__(self, tokenizer, prompt_lens):
        self.tokenizer = tokenizer
        self.prompt_lens = prompt_lens  # list[int]

    def __call__(self, input_ids, scores, **kwargs):
        for b in range(len(input_ids)):
            gen_part = input_ids[b][self.prompt_lens[b]:]
            if len(gen_part) == 0:
                continue
            text = self.tokenizer.decode(gen_part, skip_special_tokens=True)
            # 简单判定：出现至少一个 '{'，并且大括号配平
            if "{" in text and text.count("{") >= 1 and text.count("{") == text.count("}"):
                return True
        return False


# ===============================
# 批量生成（按需传入 system）
# ===============================
def batch_generate(model, tokenizer, test_data, batch_size=64, use_system=True):
    """
    评测用：支持是否传入 system；关闭采样；只接受严格 JSON。
    若首轮非 JSON，将进行一次极短“修复重试”（仍只解析 JSON）。
    """
    results = []
    for i in tqdm(range(0, len(test_data), batch_size), desc="Processing batches"):
        batch = test_data[i:i+batch_size]
        batch_messages_list = [
            build_messages_with_optional_system(item['messages'], use_system=use_system)
            for item in batch
        ]

        model.eval()
        with torch.no_grad():
            batch_prompts = [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in batch_messages_list
            ]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt",
                padding=True, truncation=True, max_length=MAX_INPUT_LEN
            ).to(model.device)

            prompt_lens = [len(ids) for ids in inputs.input_ids]
            stop = StoppingCriteriaList([JsonBraceStopping(tokenizer, prompt_lens)])

            # —— 第一次生成 —— #
            outputs = model.generate(
                **inputs,
                max_new_tokens=BASE_MAX_NEW_TOKENS,
                do_sample=False,                 # ★ 评测：禁用采样，稳定复现
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stop
            )

            first_pass = []
            for j, output in enumerate(outputs):
                input_len = len(inputs.input_ids[j])
                gen_tokens = output[input_len:]
                resp_only = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                first_pass.append(clean_generated_response(resp_only))

        # —— 尝试解析；若非 JSON，进行一次极短修复重试 —— #
        finalized = []
        for j, resp in enumerate(first_pass):
            parsed = extract_json_from_response(resp)
            if parsed is not None:
                finalized.append(resp)
                continue

            # 修复重试：极短、极硬的二次提示
            repair_prompt = (
                batch_prompts[j] +
                "\n只输出一个JSON对象，且仅包含两个整数键：{\"depression\": 0..3, \"anxiety\": 0..3}。"
                "不能有任何解释或多余字符。现在给出JSON："
            )
            repair_inputs = tokenizer(
                repair_prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN
            ).to(model.device)

            prompt_len2 = [len(repair_inputs.input_ids[0])]
            stop2 = StoppingCriteriaList([JsonBraceStopping(tokenizer, prompt_len2)])
            repair_out = model.generate(
                **repair_inputs,
                max_new_tokens=BASE_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stop2
            )
            gen2 = repair_out[0][prompt_len2[0]:]
            resp2 = tokenizer.decode(gen2, skip_special_tokens=True)
            finalized.append(clean_generated_response(resp2))

        # 组装结果
        for idx in range(len(batch_messages_list)):
            full_msgs = batch[idx]['messages']
            gt_json = get_ground_truth_from_messages(full_msgs)
            results.append({
                "id": batch[idx]['id'],
                "dialogue": batch_messages_list[idx],
                "generated_response": finalized[idx],
                "ground_truth_json": gt_json,
                "ground_truth_raw": get_last_assistant_content(full_msgs),
                "type": batch[idx].get('type', 'judgment')
            })
    return results


# ===============================
# 工具函数
# ===============================
def to_int_safe(d, key):
    try:
        return int(d[key]) if d and key in d else None
    except Exception:
        return None


def is_valid_prediction(predicted):
    return (
        predicted
        and all(k in predicted for k in ["depression", "anxiety"])
        and all(isinstance(predicted[k], int) and 0 <= predicted[k] <= 3 for k in ["depression", "anxiety"])
    )


def _safe_div(a, b):
    return a / b if b else 0.0


def build_confusion_matrix(indices_true, indices_pred, num_classes):
    cm = [[0]*num_classes for _ in range(num_classes)]
    for t, p in zip(indices_true, indices_pred):
        if t is None or p is None:
            continue
        cm[t][p] += 1
    return cm


# ===============================
# 粗略准确率：严格相等
# ===============================
def evaluate_coarse_accuracy_strict_equal(results):
    dep_total = dep_correct = 0
    anx_total = anx_correct = 0

    for item in results:
        if item.get('type') != 'judgment':
            continue
        gt = item["ground_truth_json"]
        pr = extract_json_from_response(item["generated_response"])

        gt_dep = to_int_safe(gt, "depression")
        gt_anx = to_int_safe(gt, "anxiety")
        pr_dep = to_int_safe(pr, "depression")
        pr_anx = to_int_safe(pr, "anxiety")

        if isinstance(gt_dep, int) and 0 <= gt_dep <= 3:
            dep_total += 1
            dep_correct += 1 if (isinstance(pr_dep, int) and pr_dep == gt_dep) else 0
        if isinstance(gt_anx, int) and 0 <= gt_anx <= 3:
            anx_total += 1
            anx_correct += 1 if (isinstance(pr_anx, int) and pr_anx == gt_anx) else 0

    dep_acc = dep_correct / dep_total if dep_total > 0 else 0.0
    anx_acc = anx_correct / anx_total if anx_total > 0 else 0.0
    return {
        "depression_accuracy": dep_acc,
        "anxiety_accuracy": anx_acc,
        "num_samples": {"depression": dep_total, "anxiety": anx_total}
    }


# ===============================
# P/R/F1（16 类，weighted）
# ===============================
LABELS_16 = [f"dep{d}_anx{a}" for d in range(4) for a in range(4)]
LABEL2IDX_16 = {lab: i for i, lab in enumerate(LABELS_16)}


def prf1_weighted_from_confusion(cm):
    K = len(cm)
    tp = [cm[i][i] for i in range(K)]
    fp = [sum(cm[r][i] for r in range(K)) - cm[i][i] for i in range(K)]
    fn = [sum(cm[i]) - cm[i][i] for i in range(K)]
    support = [sum(cm[i]) for i in range(K)]
    total = sum(support)

    per_class = []
    for i in range(K):
        p_i = _safe_div(tp[i], tp[i] + fp[i])
        r_i = _safe_div(tp[i], tp[i] + fn[i])
        f1_i = _safe_div(2*p_i*r_i, p_i + r_i) if (p_i + r_i) > 0 else 0.0
        per_class.append({"precision": p_i, "recall": r_i, "f1": f1_i, "support": support[i]})

    if total == 0:
        p = r = f1 = 0.0
    else:
        p = sum(per_class[i]["precision"] * support[i] for i in range(K)) / total
        r = sum(per_class[i]["recall"]    * support[i] for i in range(K)) / total
        f1 = sum(per_class[i]["f1"]       * support[i] for i in range(K)) / total

    return {"precision": p, "recall": r, "f1": f1, "support_total": total}


def evaluate_prf1_16class(results):
    sev_true_idx, sev_pred_idx = [], []

    for item in results:
        if item.get('type') != 'judgment':
            continue
        gt = item["ground_truth_json"]
        pr = extract_json_from_response(item["generated_response"])

        gt_dep = to_int_safe(gt, "depression")
        gt_anx = to_int_safe(gt, "anxiety")
        pr_dep = to_int_safe(pr, "depression")
        pr_anx = to_int_safe(pr, "anxiety")

        if (gt_dep in range(4)) and (gt_anx in range(4)) and (pr_dep in range(4)) and (pr_anx in range(4)):
            gt_label = f"dep{gt_dep}_anx{gt_anx}"
            pr_label = f"dep{pr_dep}_anx{pr_anx}"
            sev_true_idx.append(LABEL2IDX_16[gt_label])
            sev_pred_idx.append(LABEL2IDX_16[pr_label])

    cm = build_confusion_matrix(sev_true_idx, sev_pred_idx, num_classes=16)
    return prf1_weighted_from_confusion(cm)


# ===============================
# 新增：P/R/F1（单维 4 类，weighted）
# ===============================
def evaluate_prf1_4class(results, dim="depression"):
    """
    对单一维度（'depression' 或 'anxiety'）做 4 类 weighted P/R/F1
    """
    assert dim in ("depression", "anxiety")
    true_idx, pred_idx = [], []

    for item in results:
        if item.get('type') != 'judgment':
            continue
        gt = item["ground_truth_json"]
        pr = extract_json_from_response(item["generated_response"])

        gt_v = to_int_safe(gt, dim)
        pr_v = to_int_safe(pr, dim)
        if (gt_v in range(4)) and (pr_v in range(4)):
            true_idx.append(int(gt_v))
            pred_idx.append(int(pr_v))

    cm = build_confusion_matrix(true_idx, pred_idx, num_classes=4)
    return prf1_weighted_from_confusion(cm)


# ===============================
# 归一化绝对误差
# ===============================
def calculate_normalized_score(prediction_dict, ground_truth_dict, scale_range=(0, 3), weights=None):
    if prediction_dict is None or ground_truth_dict is None:
        return {"depression_score": 0.0, "anxiety_score": 0.0, "total_score": 0.0}
    min_val, max_val = scale_range
    M = max_val - min_val
    if weights is None:
        weights = {"depression": 0.5, "anxiety": 0.5}
    scores = {}
    for dim in ["depression", "anxiety"]:
        if dim in prediction_dict and dim in ground_truth_dict:
            pred, true = prediction_dict[dim], ground_truth_dict[dim]
            if not isinstance(pred, int) or not isinstance(true, int):
                scores[f"{dim}_score"] = 0.0
                continue
            pred = max(min_val, min(max_val, pred))
            true = max(min_val, min(max_val, true))
            scores[f"{dim}_score"] = 1.0 - abs(pred - true) / M
        else:
            scores[f"{dim}_score"] = 0.0
    scores["total_score"] = sum(weights.get(dim, 0) * scores.get(f"{dim}_score", 0) for dim in ["depression", "anxiety"])
    return scores


def evaluate_normalized_scores(results, scale_range=(0, 3)):
    all_scores = []
    for item in results:
        if item.get('type') != 'judgment':
            continue
        pred = extract_json_from_response(item["generated_response"])
        gt   = item["ground_truth_json"]
        if is_valid_prediction(pred):
            scores = calculate_normalized_score(pred, gt, scale_range)
        else:
            scores = {"depression_score": 0.0, "anxiety_score": 0.0, "total_score": 0.0}
        all_scores.append(scores)

    total = len(all_scores)
    avg_scores = {
        "depression_score": sum(s["depression_score"] for s in all_scores)/total if total else 0.0,
        "anxiety_score":    sum(s["anxiety_score"]    for s in all_scores)/total if total else 0.0,
        "total_score":      sum(s["total_score"]      for s in all_scores)/total if total else 0.0
    }
    return avg_scores


# ===============================
# 调试：确认 PEFT 适配器是否挂载
# ===============================
def debug_adapter(model):
    print("== PEFT debug ==")
    print("isinstance PeftModel:", isinstance(model, PeftModel))
    try:
        print("active_adapter:", getattr(model, "active_adapter", None))
        peft_cfg = getattr(model, "peft_config", None)
        print("peft_config keys:", list(peft_cfg.keys()) if peft_cfg else None)
    except Exception as e:
        print("peft introspect err:", e)


# ===============================
# 主流程
# ===============================
def main():
    set_global_seed(GLOBAL_SEED)

    base_model_path="/content/Qwen2.5-7B"  # 你的底座
    test_path="/content/MPchat_tools/eval_data_judgment.json"
    models_to_evaluate=[
        {"name":"Qwen2.5-7B", "adapter_path":None, "use_system":False},
        # {"name":"MiniCPM4-0.5b-5k-with-system", "adapter_path":"/content/drive/MyDrive/MPchat/MiniCPM4-0.5B-5k-with-system", "use_system":True},
        # {"name":"MiniCPM4-0.5b-5k-without-system", "adapter_path":"/content/drive/MyDrive/MPchat/MiniCPM4-0.5B-5k-without-system", "use_system":False}
    ]
    BATCH_SIZE=64
    SCALE_RANGE=(0,3)

    print(f"=== 配置 ===\n批处理大小: {BATCH_SIZE}\n量表范围: {SCALE_RANGE}\n基础模型: {base_model_path}\n模型数: {len(models_to_evaluate)}\n{'='*50}")
    test_data=load_testset(test_path)
    print(f"测试数据加载完成，共 {len(test_data)} 条")

    all_samples_dict = {}

    for mc in models_to_evaluate:
        model_name, adapter_path, use_system = mc["name"], mc["adapter_path"], mc["use_system"]
        print(f"\n=== 开始评估模型 {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False) if adapter_path else base_model
        debug_adapter(model)

        results = batch_generate(model, tokenizer, test_data, batch_size=BATCH_SIZE, use_system=use_system)

        # 解析成功率统计（成功=严格JSON解析成功）
        parsed_total = len([r for r in results if r.get('type')=='judgment'])
        parsed_ok = sum(1 for r in results if extract_json_from_response(r["generated_response"]) is not None)
        print(f"[{model_name}] 解析成功 {parsed_ok}/{parsed_total} 条（成功率 {parsed_ok/parsed_total if parsed_total else 0:.3f}）")

        # 评估
        coarse_acc = evaluate_coarse_accuracy_strict_equal(results)
        prf1_dep  = evaluate_prf1_4class(results, dim="depression")
        prf1_anx  = evaluate_prf1_4class(results, dim="anxiety")
        prf1_both = evaluate_prf1_16class(results)
        nrm_res   = evaluate_normalized_scores(results, SCALE_RANGE)

        # —— 打印 —— #
        print(f"[{model_name}] 粗略准确率 | 抑郁={coarse_acc['depression_accuracy']:.3f} "
              f"(样本数 {coarse_acc['num_samples']['depression']}) | "
              f"焦虑={coarse_acc['anxiety_accuracy']:.3f} "
              f"(样本数 {coarse_acc['num_samples']['anxiety']})")

        print(f"[{model_name}] P/R/F1 —— 列：抑郁(4类) | 焦虑(4类) | 合并(16类)")
        print("Precision  | {0:.3f} | {1:.3f} | {2:.3f}".format(
            prf1_dep['precision'], prf1_anx['precision'], prf1_both['precision']))
        print("Recall     | {0:.3f} | {1:.3f} | {2:.3f}".format(
            prf1_dep['recall'], prf1_anx['recall'], prf1_both['recall']))
        print("F1         | {0:.3f} | {1:.3f} | {2:.3f}".format(
            prf1_dep['f1'], prf1_anx['f1'], prf1_both['f1']))

        print(f"[{model_name}] Norm —— 列：抑郁 | 焦虑 | 总分")
        print("Norm       | {0:.3f} | {1:.3f} | {2:.3f}".format(
            nrm_res['depression_score'], nrm_res['anxiety_score'], nrm_res['total_score']))

        # 保存所有模型输出
        for r in results:
            sid = r["id"]
            if sid not in all_samples_dict:
                all_samples_dict[sid] = {
                    "id": sid,
                    "dialogue": r["dialogue"],  # 实际喂给模型的对话（包含/不包含 system 取决于 use_system）
                    "ground_truth": {
                        "label": r["ground_truth_json"],
                        "true_label": r["ground_truth_raw"]
                    },
                    "models": {}
                }
            all_samples_dict[sid]["models"][model_name] = {
                "model_output": r["generated_response"],
                "predicted": extract_json_from_response(r["generated_response"])
            }

        # 释放显存
        del model, base_model, tokenizer
        torch.cuda.empty_cache(); gc.collect()

    # 写出：所有模型输出（一个文件）
    all_outputs = {
        "timestamp": datetime.now().isoformat(),
        "samples": sorted(all_samples_dict.values(), key=lambda x: x["id"])
    }
    with open('all_model_outputs.json', 'w', encoding='utf-8') as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print("\n已保存所有模型输出至: all_model_outputs.json")


if __name__=="__main__":
    main()
