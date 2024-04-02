from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# LLama Model
llama_model =  "/home/hech/models/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(llama_model, trust_remote_code=True)
llama = AutoModelForCausalLM.from_pretrained(llama_model)

# Load QWen model
qwen_model = "/home/hech/models/Qwen-7B-Chat"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model, trust_remote_code = True)
qwen = AutoModelForCausalLM.from_pretrained(qwen_model, device_map="auto", trust_remote_code=True).eval()

# Load Yi Model
yi_model = "/home/hech/models/Yi-34B-Chat"
yi_tokenizer = AutoTokenizer.from_pretrained(yi_model, trust_remote_code = True)
yi = AutoModelForCausalLM.from_pretrained(yi_model, device_map = "auto", trust_remote_code = True).eval()

## Qwen Model
print("=======================Qwen================================")
print(qwen)
print("=======================Qwen=====>")
qwen_model_vocab_size = qwen.get_input_embeddings().weight.size(0)
qwen_tokenzier_vocab_size = len(qwen_tokenizer)
print(qwen.get_input_embeddings().weight.size())
print(f"Vocab of the base model: {qwen_model_vocab_size}")
print(f"Vocab of the tokenizer: {qwen_tokenzier_vocab_size}")

for name, param in qwen.named_parameters():
    print(name, param.numel(), param.requires_grad)

## Llama
model_vocab_size = llama.get_input_embeddings().weight.size(0)
tokenzier_vocab_size = len(tokenizer)
print(llama.get_input_embeddings().weight.size())
print("=======================LLama=>")
print(llama)
print(f"Vocab of the base model: {model_vocab_size}")
print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")

print("=======================LLama================================")
for name, param in llama.named_parameters():
    print(name, param.numel(), param.requires_grad)

## Yi Model
yi_model_vocab_size = yi.get_input_embeddings().weight.size(0)
yi_tokenzier_vocab_size = len(yi_tokenizer)
print(yi.get_input_embeddings().weight.size())
print("=======================YI============>")
print(yi)
print(f"Vocab of the base model: {yi_model_vocab_size}")
print(f"Vocab of the tokenizer: {yi_tokenzier_vocab_size}")
print("=======================YI================================")
for name, param in yi.named_parameters():
    print(name, param.numel(), param.requires_grad)


