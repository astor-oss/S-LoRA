import argparse
import os

# base_model = "dummy-llama-7b"
# base_model = "huggyllama/llama-7b"
# adapter_dirs = ["tloen/alpaca-lora-7b", "MBZUAI/bactrian-x-llama-7b-lora"]

#base_model = "huggyllama/llama-7b"
#adapter_dirs = ["/slurmhome/huzx/Code/huzx_llama_factory.git/checkpoint_lora_ft_llama-7b-kv/checkpoint10/", "/slurmhome/huzx/Code/huzx_llama_factory.git/checkpoint_lora_ft_llama-7b-kv/checkpoint20"]

#base_model = "meta-llama/Llama-2-13b-hf"
#adapter_dirs = ["/slurmhome/huzx/Code/huzx_llama_factory.git/checkpoint_lora_ft_llama2-13b_ko/checkpoint1000/", "/slurmhome/huzx/Code/huzx_llama_factory.git/checkpoint_lora_ft_llama2-13b_ko/checkpoint2000"]

base_model = "Qwen/Qwen-7B-Chat"
adapter_dirs = ["/slurmhome/huzx/Code/huzx_llama_factory.git/checkpoint_lora_ft_qwen-chat-1/checkpoint1000/", "/slurmhome/huzx/Code/huzx_llama_factory.git/checkpoint_lora_ft_qwen-chat-1/checkpoint2000"]

#base_model = "01-ai/Yi-34B-Chat"
#adapter_dirs = ["/slurmhome/huzx/Code/huzx_llama_factory.git/checkpoint_lora_ft_yi-34b-chat-3-ko/checkpoint1000/", "/slurmhome/huzx/Code/huzx_llama_factory.git/checkpoint_lora_ft_yi-34b-chat-3-ko/checkpoint2000"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-adapter", type=int)
    parser.add_argument("--num-token", type=int)
    parser.add_argument("--pool-size-lora", type=int)

    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--no-mem-pool", action="store_true")
    args = parser.parse_args()

    
    if args.num_adapter is None: args.num_adapter = 4
    if args.num_token is None: args.num_token = 10000
    if args.pool_size_lora is None: args.pool_size_lora = 0
 
    cmd = f"python -m slora.server.api_server --max_total_token_num {args.num_token}"
    cmd += f" --model {base_model}"
    cmd += f" --tokenizer_mode auto"
    cmd += f" --pool-size-lora {args.pool_size_lora}"

    num_iter = args.num_adapter // len(adapter_dirs) + 1
    for adapter_dir in adapter_dirs:
        cmd += f" --lora {adapter_dir}"

    cmd += " --swap"
    # cmd += " --scheduler pets"
    # cmd += " --profile"
    if args.no_lora_compute:
        cmd += " --no-lora-compute"
    if args.no_prefetch:
        cmd += " --prefetch False"
    if args.no_mem_pool:
        cmd += " --no-mem-pool"

    print(f"Final run command is: {cmd}")
    os.system(cmd)
