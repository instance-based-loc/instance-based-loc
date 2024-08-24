import logging
import torch
import psutil

BLUE = "\033[94m"
RESET = "\033[0m"

logging.basicConfig(level=logging.INFO)

def conditional_log(message, should_log):
    if should_log:
        logging.info(f"{BLUE}> {message}{RESET}")

def get_mem_stats():
    pid = psutil.Process()
    memory_info = pid.memory_info()
    memory_info_GBs = memory_info.rss / (1e3 ** 3)

    cuda_memory_stats = torch.cuda.memory_stats()
    max_cuda_memory_GBs = int(cuda_memory_stats["allocated_bytes.all.peak"]) / (1e3 ** 3)
    # TODO subtract initial memory and GPU usuage from before the run 
    return f"{memory_info_GBs:.3f}", f"{max_cuda_memory_GBs:.3f}"
