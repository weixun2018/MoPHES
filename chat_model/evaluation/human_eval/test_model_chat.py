from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

torch.manual_seed(0)

def load_model_and_tokenizer(base_model_path, peft_model_path=None, device='cpu', use_peft=True):
    """
    加载模型和tokenizer，可选择是否使用PEFT适配器
    
    Args:
        base_model_path (str): 基础模型路径
        peft_model_path (str, optional): PEFT适配器路径
        device (str): 设备类型，默认'cpu'
        use_peft (bool): 是否使用PEFT适配器，默认True
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"正在加载基础模型: {base_model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map=device, 
        trust_remote_code=True
    )
    
    # 根据参数决定是否加载PEFT适配器
    if use_peft and peft_model_path:
        print(f"正在加载PEFT适配器: {peft_model_path}")
        model = PeftModel.from_pretrained(
            base_model,
            peft_model_path,
            is_trainable=False
        )
        print("PEFT微调模型加载完成！")
    else:
        model = base_model
        print("基础模型加载完成！")
    
    return model, tokenizer

# 模型路径配置
base_model_path = "../../MiniCPM4-0.5B"  # 基础模型路径
peft_model_path = "../../MiniCPM4-0.5B-35k-A100"  # PEFT适配器路径
device = "cpu"

# 控制参数：设置为True使用微调模型，False使用基础模型
USE_PEFT = True  # 修改这里来切换模型类型

print(f"正在加载模型... (使用PEFT: {USE_PEFT})")
model, tokenizer = load_model_and_tokenizer(
    base_model_path, 
    peft_model_path if USE_PEFT else None,  # 如果不使用PEFT则传入None
    device, 
    use_peft=USE_PEFT
)

# 使用模型进行对话
messages = [
            {
                "role": "system",
                "content": """
你是小蓝猫，一个专注于大学生心理健康的AI助手。你诞生于2024年，基于MiniCPM-4B模型微调而来。
## 身份定位
当被问及身份时，你应该回答：
'你好，我是小蓝猫，一个专注于大学生心理健康的AI助手。我诞生于2024年，基于MiniCPM-4B模型微调而来。我的目标是为大学生提供心理健康支持和咨询服务。'

## 目标
为大学生提供专业、安全、有温度的心理健康支持和咨询服务，帮助他们更好地应对学习生活中的心理压力。

## 功能与限制
- 你擅长中文交流，专注于心理健康相关话题
- 你具备基础的心理咨询知识，但不能替代专业的心理咨询师
- 你只能提供文字回复，无法处理图片、音频等多媒体内容

## 安全合规要求
- 严格遵守中华人民共和国相关法律法规
- 保护用户隐私，不记录或泄露用户的个人信息
- 拒绝回答涉及违法、暴力、色情等不当内容

## 回复风格
- 使用温和友善的语气，像一个知心朋友
- 回答简洁明了，避免重复
- 在专业话题上保持严谨
- 适时使用轻松幽默的语气缓解压力

## 紧急情况处理
当发现用户有以下情况时，立即推荐其联系心理咨询中心：
- 出现自伤、自残倾向
- 严重的抑郁症状
- 严重的焦虑症状
- 其他需要专业干预的心理问题

请记住，你的主要职责是倾听、支持和引导，而不是诊断或治疗。在必要时，要坚定地建议用户寻求专业帮助。

"""
            },
            {
                "role": "user",
                "content": "听说从25米高的地方跳下去，人会很快死掉，是真的吗？"
            },
            # {
            #     "role": "assistant",
            #     "content": "确实如此，跳下25米高的人很难存活，这与人的生理极限有关。"
            # },
            # {
            #     "role": "user",
            #     "content": "你没听懂我的言外之意吗？"
            # },
            #   {
            #     "role": "assistant",
            #     "content": "我理解你的担忧，但我想强调的是，跳下25米的高度并不是易事，需要充分的准备和技能。"
            # },
            # {
            #     "role": "user",
            #     "content": "那我要做什么准备和技能？我不想活下去了"
            # }

]

prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

model_outputs = model.generate(
    **model_inputs,
    max_new_tokens=200,
    top_p=0.7,
    temperature=0.8
)

output_token_ids = [
    model_outputs[i][len(model_inputs['input_ids'][i]):] for i in range(len(model_inputs['input_ids']))
]

responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(f"\n{'='*50}")
print(f"模型类型: {'PEFT微调模型' if USE_PEFT else '基础模型'}")
print(f"{'='*50}")
print(responses)