import subprocess
import os
import json
import logging

from utils.extract_task_code import file_to_string


import os
import json
import subprocess

def set_freest_gpu():
    target_gpu = get_target_gpu()
    print(f"Setting CUDA_VISIBLE_DEVICES to: {target_gpu}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(target_gpu)

def get_target_gpu():
    try:
        sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str, _ = sp.communicate()
        gpustats = json.loads(out_str.decode('utf-8'))
    except Exception as e:
        print(f"Error checking gpustat: {e}")
        return 3 

    gpus_dict = {g['index']: g for g in gpustats['gpus']}
    

    target_candidates = [0, 1, 2, 3]
    
    best_gpu = target_candidates[0]
    max_free_mem = -1

    for idx in target_candidates:
        if idx in gpus_dict:
            gpu_info = gpus_dict[idx]
            free_mem = gpu_info['memory.total'] - gpu_info['memory.used']
            if free_mem > max_free_mem:
                max_free_mem = free_mem
                best_gpu = idx
        else:
            print(f"Warning: GPU {idx} not found in gpustat output.")

    print(f"Selected GPU {best_gpu} with {max_free_mem}MB free memory.")
    return best_gpu



def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if "fps step:" in rl_log or "Traceback" in rl_log:
            if log_status and "fps step:" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break
