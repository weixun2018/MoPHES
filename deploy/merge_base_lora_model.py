from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# from transformers import EncoderDecoderCache

base_model_path = '/path/to//MiniCPM4-0.5B'
adapter_path = '/path/to/MiniCPM4-0.5B-DialogueModel-Adapter'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)

lora_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
merged_model = lora_model.merge_and_unload()

merged_model.save_pretrained("/path/to/result/MiniCPM4-0.5B-DialogueModel")

