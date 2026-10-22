import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
from openai import OpenAI
import re
import subprocess
from pathlib import Path
import shutil
import time 

from utils.misc import * 
from utils.file_utils import load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *


IDEA_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ISAAC_ROOT_DIR = os.path.abspath(os.path.join(IDEA_ROOT_DIR, "../IsaacGymEnvs/isaacgymenvs"))


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):

    workspace_dir = Path.cwd()
    logging.info(f"Idea Root Dir: {IDEA_ROOT_DIR}")
    logging.info(f"IsaacGymEnvs Root Dir: {ISAAC_ROOT_DIR}")
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {IDEA_ROOT_DIR}")

    info_dir = workspace_dir / "info"
    prompts_history_dir = info_dir / "prompts_history"
    reward_archive_dir = info_dir / "reward_archive"
    best_functions_dir = info_dir / "best_functions"
    experience_pool_dir = info_dir / "experience_pool"
    
    info_dir.mkdir(exist_ok=True)
    prompts_history_dir.mkdir(exist_ok=True)
    reward_archive_dir.mkdir(exist_ok=True)
    best_functions_dir.mkdir(exist_ok=True)
    experience_pool_dir.mkdir(exist_ok=True)
    
    logging.info(f"Info directory created at: {info_dir}")

    # chatGPT = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
    qwen = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    role_temperature = {
        "INNOVATOR": cfg.innovator_temperature,
        "GUARDIAN": cfg.guardian_temperature,
        "EVOLVER": cfg.evolver_temperature,
        "PROGENITOR": cfg.progenitor_temperature
    }
    
    logging.info(f"Using LLM: {model}")
    logging.info(f"Task: {task}")
    logging.info(f"Task Description: {task_description}")
    env_name = cfg.env.env_name.lower()
    env_parent = 'isaac' if f'{env_name}.py' in os.listdir(f'{IDEA_ROOT_DIR}/envs/isaac') else 'bidex'
    logging.info(f"Env Parent: {env_parent}")
    
    task_file = f'{IDEA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    task_obs_file = f'{IDEA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_obs_code_string  = file_to_string(task_obs_file)
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"
    roles = ["INNOVATOR", "GUARDIAN", "EVOLVER", "PROGENITOR"]
    prompt_dir = f'{IDEA_ROOT_DIR}/utils/prompts'
    

    role_files = {
        "INNOVATOR_SYSTEM": f"{prompt_dir}/innovator_system.txt",
        "GUARDIAN_SYSTEM": f"{prompt_dir}/guardian_system.txt",
        "EVOLVER_SYSTEM": f"{prompt_dir}/evolver_system.txt",
        "INITIAL_SYSTEM": f"{prompt_dir}/initial_system.txt",
        "INNOVATOR_ASSISTANT": f"{prompt_dir}/innovator_assistant.txt",
        "GUARDIAN_ASSISTANT": f"{prompt_dir}/guardian_assistant.txt",
    }


    initial_system = file_to_string(role_files["INITIAL_SYSTEM"])
    innovator_system= file_to_string(role_files["INNOVATOR_SYSTEM"])   
    guardian_system = file_to_string(role_files["GUARDIAN_SYSTEM"])
    evolver_system = file_to_string(role_files["EVOLVER_SYSTEM"])

    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
    
    initial_system = initial_system.format(task_reward_signature_string=reward_signature)
    innovator_system = innovator_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    guardian_system = guardian_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    
    innovator_assistant = file_to_string(role_files["INNOVATOR_ASSISTANT"])
    guardian_assistant = file_to_string(role_files["GUARDIAN_ASSISTANT"])
    task_code_string = task_code_string.replace(task, task+suffix)
    create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)
    DUMMY_FAILURE = -10000.

    experience_pool = []
    role_information = {}
    for role in roles:
        role_information[role] = {
            "code_runs": [],
            "rl_runs": [],
            "code_paths": [],
            "rl_paths": [],
            "contents": [],
            "success": [],
            "reward_correlations": [],
            "responses": [],
        }
    
    global_best = {
        "role": None,
        "response_id": None,
        "iteration": None,
        "success": -float("inf"),
        "reward_corr": DUMMY_FAILURE,
        "content": None,
        "code_path": None
    }
    
    for iter in range(cfg.iteration):
        logging.info(f"=== Iteration {iter} ===")
        
        iter_prompts_dir = prompts_history_dir / f"iteration_{iter}"
        iter_prompts_dir.mkdir(exist_ok=True)
        
        sample_budget = 16
        if iter == 0:
            role_sample_counts = {
                "PROGENITOR": sample_budget,
                "INNOVATOR": 0,
                "GUARDIAN": 0,
                "EVOLVER": 0
            }
        else:
            # t is 1-indexed. From iteration 2, (Explorer, Guardian, Evolver)
            # follows ceil(N * (1 - (t-1)/(T-1))), then split the remainder.
            t = iter + 1
            n_explorer = int(np.ceil(sample_budget * (1 - (t - 1) / (cfg.iteration - 1))))
            n_guardian = int(np.floor((sample_budget - n_explorer) / 2))
            n_evolver = sample_budget - n_explorer - n_guardian
            role_sample_counts = {
                "PROGENITOR": 0,
                "INNOVATOR": n_explorer,
                "GUARDIAN": n_guardian,
                "EVOLVER": n_evolver
            }
        
        active_roles = [role for role in roles if role_sample_counts[role] > 0]
        logging.info(f"Active roles for iteration {iter}: {active_roles}")
        logging.info(f"Sample counts: {role_sample_counts}")
        
        role_messages = {}
        
        for role in active_roles:
            if iter == 0:
                if role == "PROGENITOR":
                    role_messages[role] = [
                        {"role": "system", "content": initial_system},
                        {"role": "user", "content": initial_user}
                    ]

            else:
                sorted_pool = sorted(experience_pool, key=lambda x: x["success"], reverse=True)
                
                if role == "INNOVATOR":
                    if len(sorted_pool) >= 2:
                        first_best = sorted_pool[0]
                        second_candidates = sorted_pool[1:min(5, len(sorted_pool))]
                        if second_candidates:
                            second_best = second_candidates[np.random.randint(len(second_candidates))]
                        else:
                            second_best = sorted_pool[1] if len(sorted_pool) > 1 else first_best
                        
                        
                        innovator_assistant_content = innovator_assistant.format(
                            first_function=first_best['reward_code'],
                            the_first_function_metrics=first_best['feedback_content'],
                            second_function=second_best['reward_code'],
                            the_second_function_metrics=second_best['feedback_content']
                        )
                        
                        role_messages[role] = [
                            {"role": "system", "content": innovator_system},
                            {"role": "user", "content": initial_user},
                            {"role": "assistant", "content": innovator_assistant_content}
                        ]
                    else:
                        logging.warning(f"Iteration {iter}: Not enough entries in experience pool for INNOVATOR")
                        continue
                
                elif role == "GUARDIAN":
                    if len(sorted_pool) >= 1:
                        best_function = sorted_pool[0]
                        guardian_assistant_content = guardian_assistant.format(
                            best_function=best_function['reward_code'],
                            the_best_function_metrics=best_function['feedback_content']
                        )
                        
                        role_messages[role] = [
                            {"role": "system", "content": guardian_system},
                            {"role": "user", "content": initial_user},
                            {"role": "assistant", "content": guardian_assistant_content}
                        ]
                    else:
                        logging.warning(f"Iteration {iter}: Not enough entries in experience pool for GUARDIAN")
                        continue
                
                elif role == "EVOLVER":
                    sorted_pool = sorted(experience_pool, key=lambda x: x["success"], reverse=True)
                    
                    if len(sorted_pool) >= 2:
                        first_best = sorted_pool[0]
                        second_candidates = sorted_pool[1:min(5, len(sorted_pool))]
                        if second_candidates:
                            second_best = second_candidates[np.random.randint(len(second_candidates))]
                        else:
                            second_best = sorted_pool[1]

                        evolver_system_iter = evolver_system.format(
                            task_reward_signature_string=reward_signature,
                            first_function=first_best['reward_code'],
                            the_first_function_metrics=first_best['feedback_content'],
                            second_function=second_best['reward_code'],
                            the_second_function_metrics=second_best['feedback_content']
                        )

                        evolver_system_iter = evolver_system_iter + code_output_tip
                        
                        role_messages[role] = [
                            {"role": "system", "content": evolver_system_iter},
                            {"role": "user", "content": initial_user}
                        ]
                    else:
                        logging.warning(f"Iteration {iter}: Not enough entries in experience pool for EVOLVER")
                        continue

        for role in role_messages:
            role_prompt_content = ""
            for msg_idx, msg in enumerate(role_messages[role]):
                role_prompt_content += f"=== {msg['role'].upper()} ===\n{msg['content']}\n\n"
            
            prompt_file = iter_prompts_dir / f"{role}_prompt.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(role_prompt_content)
            logging.info(f"Saved prompt for {role} to {prompt_file}")
        
        for role in role_messages:
            logging.info(f"Role: {role}")
            role_information[role]["responses"] = []
            response_cur = None
            total_samples = 0
            total_token = 0
            total_completion_token = 0
            
            sample_count = role_sample_counts[role]
            chunk_size = min(sample_count, 4)  
            role_temp = role_temperature[role]
            logging.info(f"Generating {sample_count} samples with {cfg.model} and temperature {role_temp} for role {role}")

            while total_samples < sample_count:
                remaining = sample_count - total_samples
                current_chunk = min(chunk_size, remaining)
                
                for attempt in range(1000):
                    try:
                        if cfg.model.startswith("gpt-"):
                            response_cur = chatGPT.chat.completion.create(
                                model=model,
                                messages=role_messages[role],
                                temperature=role_temp,
                                n=current_chunk
                            )
                        elif cfg.model.startswith("qwen"):
                            response_cur = qwen.chat.completions.create(
                                model=model,
                                messages=role_messages[role],
                                temperature=role_temp,
                                n=current_chunk
                            )
                        total_samples += current_chunk
                        break
                    except Exception as e:
                        if attempt >= 10:
                            chunk_size = max(int(chunk_size / 2), 1)
                            logging.info(f"Reduced chunk size to {chunk_size}")
                        logging.info(f"Attempt {attempt+1} failed with error: {e}")
                        time.sleep(1)
                        
                if response_cur is None:
                    logging.info("Code terminated due to too many failed attempts!")
                    exit()

                role_information[role]["responses"].extend(response_cur.choices)
                prompt_tokens = response_cur.usage.prompt_tokens
                total_completion_token += response_cur.usage.completion_tokens
                total_token += response_cur.usage.total_tokens

            logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

        for role in role_messages:
            role_information[role]["code_runs"] = []
            role_information[role]["rl_runs"] = []

            sample_count = role_sample_counts[role]
            
            for response_id in range(sample_count):
                
                response_cur = role_information[role]["responses"][response_id].message.content
                logging.info(f"Iteration {iter}: role_{role} Processing Code Run {response_id}")

                patterns = [
                    r'```python(.*?)```',
                    r'```(.*?)```',
                    r'"""(.*?)"""',
                    r'""(.*?)""',
                    r'"(.*?)"',
                ]
                for pattern in patterns:
                    code_string = re.search(pattern, response_cur, re.DOTALL)
                    if code_string is not None:
                        code_string = code_string.group(1).strip()
                        break
                code_string = response_cur if not code_string else code_string
                code_string = normalize_total_reward(code_string)
                
                try:
                    gpt_reward_signature, input_lst = get_function_signature(code_string)
                except Exception as e:
                    logging.info(f"Iteration {iter}: Role {role} - Code Run {response_id} cannot parse function signature!")
                    continue

                role_information[role]["code_runs"].append(code_string)
                reward_signature_lines = [
                    f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                    f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                    f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
                ]
                indent = " " * 8
                reward_signature_code = "\n".join([indent + line for line in reward_signature_lines])

                if "def compute_reward(self)" in task_code_string:
                    task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature_code)
                elif "def compute_reward(self, actions)" in task_code_string:
                    task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature_code)
                else:
                    raise NotImplementedError

                with open(output_file, 'w') as file:
                    file.writelines(task_code_string_iter + '\n')
                    file.writelines("from typing import Tuple, Dict" + '\n')
                    file.writelines("import math" + '\n')
                    file.writelines("import torch" + '\n')
                    file.writelines("from torch import Tensor" + '\n')
                    if "@torch.jit.script" not in code_string:
                        code_string = "@torch.jit.script\n" + code_string
                    file.writelines(code_string + '\n')
                    
                with open(f"env_iter{iter}_role_{role}_response{response_id}_rewardonly.py", 'w') as file:
                    file.writelines("from typing import Tuple, Dict" + '\n')
                    file.writelines("import math" + '\n')
                    file.writelines("import torch" + '\n')
                    file.writelines("from torch import Tensor" + '\n')
                    file.writelines(code_string + '\n')

                shutil.copy(output_file, f"env_iter{iter}_role_{role}_response{response_id}.py")
                
                set_freest_gpu()

                rl_filepath = f"env_iter{iter}_role_{role}_response{response_id}.txt"
                role_output_dir = f"./Iteration_{iter}/{role}_{response_id}"
                with open(rl_filepath, 'w') as f:
                    process = subprocess.Popen([
                        'python', '-u', f'{ISAAC_ROOT_DIR}/train.py',
                        f'hydra.run.dir={role_output_dir}',
                        f'hydra.output_subdir=null',
                        f'task={task}{suffix}',
                        f'wandb_activate={cfg.use_wandb}',
                        f'wandb_entity={cfg.wandb_username}',
                        f'wandb_project={cfg.wandb_project}',
                        f'headless={not cfg.capture_video}',
                        f'capture_video={cfg.capture_video}',
                        'force_render=False',
                        f'max_iterations={cfg.max_iterations}',
                        f'seed={cfg.seed}'
                    ], stdout=f, stderr=f)
                block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
                role_information[role]["rl_runs"].append(process)
        

        for role in role_messages:
            logging.info(f"Role: {role} - Processing RL Results")
            role_information[role]["contents"] = []
            role_information[role]["success"] = []
            role_information[role]["reward_correlations"] = []
            role_information[role]["code_paths"] = []

            sample_count = role_sample_counts[role]

            for response_id in range(sample_count):
                if response_id >= len(role_information[role]["code_runs"]):
                    continue
                    
                code_run = role_information[role]["code_runs"][response_id]
                rl_run = role_information[role]["rl_runs"][response_id]
                
                rl_run.communicate()
                rl_filepath = f"env_iter{iter}_role_{role}_response{response_id}.txt"
                code_path = f"env_iter{iter}_role_{role}_response{response_id}.py"
                role_information[role]["code_paths"].append(code_path)
                
                try:
                    with open(rl_filepath, 'r') as f:
                        stdout_str = f.read() 
                except: 
                    content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                    content += code_output_tip
                    role_information[role]["contents"].append(content) 
                    role_information[role]["success"].append(DUMMY_FAILURE)
                    role_information[role]["reward_correlations"].append(DUMMY_FAILURE)
                    continue

                content = ''
                traceback_msg = filter_traceback(stdout_str)

                if traceback_msg == '':
                    lines = stdout_str.split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith('Tensorboard Directory:'):
                            break 
                    tensorboard_logdir = line.split(':')[-1].strip() 
                    tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                    max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                    epoch_freq = max(int(max_iterations // 10), 1)
                    
                    content += policy_feedback.format(epoch_freq=epoch_freq)
                    
                    if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                        gt_reward = np.array(tensorboard_logs["gt_reward"])
                        gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                        reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                        role_information[role]["reward_correlations"].append(reward_correlation)

                    successes_rate = DUMMY_FAILURE
                    for metric in tensorboard_logs:
                        if "/" not in metric:
                            metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                            metric_cur_max = max(tensorboard_logs[metric])
                            metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                            metric_cur_min = min(tensorboard_logs[metric])
                            
                            if "consecutive_successes" == metric:
                                role_information[role]["success"].append(metric_cur_max)
                                successes_rate = metric_cur_max
                            
                            if metric != "gt_reward" and metric != "gpt_reward":
                                if metric != "consecutive_successes":
                                    metric_name = metric 
                                else:
                                    metric_name = "task_score"
                                content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f}\n"
                            else:
                                # Provide ground-truth score when success rate not applicable
                                if "consecutive_successes" not in tensorboard_logs:
                                    content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f}\n"
                    
                    
                    if successes_rate != DUMMY_FAILURE:
                        experience_entry = {
                            "iteration": iter,
                            "role": role,
                            "response_id": response_id,
                            "reward_code": code_run,
                            "success": successes_rate,
                            "reward_correlation": role_information[role]["reward_correlations"][-1] if role_information[role]["reward_correlations"] else DUMMY_FAILURE,
                            "feedback_content": content,  
                            "code_path": code_path
                        }
                        
                        MAX_POOL_SIZE = 5
                        should_add = False
                        removed_entry = None
                        
                        if len(experience_pool) < MAX_POOL_SIZE:
                            experience_pool.append(experience_entry)
                            should_add = True
                            logging.info(f"Added to experience pool: iter={iter}, role={role}, response={response_id}, success={successes_rate:.4f} (Pool size: {len(experience_pool)}/{MAX_POOL_SIZE})")
                        else:
                            min_entry = min(experience_pool, key=lambda x: x["success"])
                            if successes_rate > min_entry["success"]:
                                experience_pool.remove(min_entry)
                                experience_pool.append(experience_entry)
                                removed_entry = min_entry
                                should_add = True
                                logging.info(f"Replaced worst entry in experience pool: iter={iter}, role={role}, response={response_id}, success={successes_rate:.4f}")
                                logging.info(f"  Removed: iter={min_entry['iteration']}, role={min_entry['role']}, response={min_entry['response_id']}, success={min_entry['success']:.4f}")
                            else:
                                logging.info(f"Entry rejected (success={successes_rate:.4f} <= min={min_entry['success']:.4f}): iter={iter}, role={role}, response={response_id}")
                        
                        if should_add:
                            reward_filename = f"iter_{iter}_{role}_{response_id}.py"
                            reward_filepath = reward_archive_dir / reward_filename
                            with open(reward_filepath, 'w', encoding='utf-8') as f:
                                f.write(f"# Iteration: {iter}\n")
                                f.write(f"# Role: {role}\n")
                                f.write(f"# Response ID: {response_id}\n")
                                f.write(f"# Success Rate: {successes_rate:.4f}\n")
                                f.write(f"# Reward Correlation: {experience_entry['reward_correlation']:.4f}\n\n")
                                f.write(code_run)
                            
                            logging.info(f"Saved reward function to {reward_filepath}")
                            
                        
                        if successes_rate > global_best["success"]:
                            global_best.update({
                                "role": role,
                                "response_id": response_id,
                                "iteration": iter,
                                "success": successes_rate,
                                "reward_corr": experience_entry["reward_correlation"],
                                "content": content,
                                "code_path": code_path
                            })
                    
                    content += code_feedback  
                else:
                    # Otherwise, provide execution traceback error feedback
                    role_information[role]["success"].append(DUMMY_FAILURE)
                    role_information[role]["reward_correlations"].append(DUMMY_FAILURE)
                    content += execution_error_feedback.format(traceback_msg=traceback_msg)

                content += code_output_tip
                role_information[role]["contents"].append(content)


        iter_experience_file = experience_pool_dir / f"experience_pool_iter_{iter}.json"
        with open(iter_experience_file, 'w', encoding='utf-8') as f:
            sorted_pool = sorted(experience_pool, key=lambda x: x["success"], reverse=True)

            serializable_pool = []
            for entry in sorted_pool:
                serializable_entry = {
                    "iteration": entry["iteration"],
                    "role": entry["role"],
                    "response_id": entry["response_id"],
                    "success": float(entry["success"]),
                    "reward_correlation": float(entry["reward_correlation"]),
                    "code_path": entry["code_path"]
                }
                serializable_pool.append(serializable_entry)
            json.dump(serializable_pool, f, indent=4, ensure_ascii=False)
        
        logging.info(f"Saved experience pool snapshot to {iter_experience_file}")
        

        if global_best["role"] is not None:
            logging.info(
                f"Iteration {iter} - "
                f"🏆 Global Best Reward Function from Role [{global_best['role']}] - "
                f"Iteration: {global_best['iteration']}, "
                f"Response ID: {global_best['response_id']}, "
                f"Success: {global_best['success']:.4f}"
            )
            

            best_function_file = best_functions_dir / f"global_best_iter_{iter}.py"
            with open(best_function_file, 'w', encoding='utf-8') as f:
                f.write(f"# Global Best Function up to Iteration {iter}\n")
                f.write(f"# Role: {global_best['role']}\n")
                f.write(f"# Original Iteration: {global_best['iteration']}\n")
                f.write(f"# Response ID: {global_best['response_id']}\n")
                f.write(f"# Success Rate: {global_best['success']:.4f}\n")
                f.write(f"# Reward Correlation: {global_best['reward_corr']:.4f}\n\n")
                
                with open(global_best['code_path'], 'r') as code_file:
                    best_code = code_file.read()
                f.write(best_code)
            
            logging.info(f"Saved global best function to {best_function_file}")
    
    final_experience_file = experience_pool_dir / "experience_pool_final.json"
    with open(final_experience_file, 'w', encoding='utf-8') as f:
        sorted_pool = sorted(experience_pool, key=lambda x: x["success"], reverse=True)
        serializable_pool = []
        for entry in sorted_pool:
            serializable_entry = {
                "iteration": entry["iteration"],
                "role": entry["role"],
                "response_id": entry["response_id"],
                "success": float(entry["success"]),
                "reward_correlation": float(entry["reward_correlation"]),
                "code_path": entry["code_path"]
            }
            serializable_pool.append(serializable_entry)
        json.dump(serializable_pool, f, indent=4, ensure_ascii=False)
    
    logging.info(f"Saved final experience pool to {final_experience_file}")
    
    global_best_metadata = {
        "role": global_best["role"],
        "iteration": global_best["iteration"],
        "response_id": global_best["response_id"],
        "success": float(global_best["success"]),
        "reward_correlation": float(global_best["reward_corr"]),
        "code_path": global_best["code_path"]
    }
    
    metadata_file = best_functions_dir / "global_best_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(global_best_metadata, f, indent=4, ensure_ascii=False)
    
    logging.info(f"Saved global best metadata to {metadata_file}")


    if global_best["code_path"] is None:
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    
    max_reward_code_path = global_best["code_path"]
    logging.info(f"Task: {task}, Max Training Success {global_best['success']:.4f}, Correlation {global_best['reward_corr']:.4f}, Best Reward Code Path: {max_reward_code_path}")
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    
    shutil.copy(max_reward_code_path, output_file)
    
    eval_dir = info_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    eval_runs = []
    for i in range(cfg.num_eval):
        
        rl_filepath = eval_dir / f"reward_code_eval{i}.txt"
        eval_output_dir = f"./Evaluation/eval_{i}"
        
        set_freest_gpu()
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen([
                'python', '-u', f'{ISAAC_ROOT_DIR}/train.py',
                f'hydra.run.dir={eval_output_dir}',
                f'hydra.output_subdir=null',
                f'task={task}{suffix}',
                f'wandb_activate={cfg.use_wandb}',
                f'wandb_entity={cfg.wandb_username}',
                f'wandb_project={cfg.wandb_project}',
                f'headless={not cfg.capture_video}',
                f'capture_video={cfg.capture_video}',
                'force_render=False',
                f'seed={i}',
                f'max_iterations={cfg.test_max_iterations}'
            ], stdout=f, stderr=f)
        
        block_until_training(str(rl_filepath), log_status=True, iter_num=-1, response_id=i)
        eval_runs.append(process)
    
    reward_code_final_successes = []
    reward_code_correlations_final = []
    
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        rl_filepath = eval_dir / f"reward_code_eval{i}.txt"
        
        try:
            with open(rl_filepath, 'r') as f:
                stdout_str = f.read()
            
            lines = stdout_str.split('\n')
            tensorboard_logdir = None
            for line in lines:
                if line.startswith('Tensorboard Directory:'):
                    tensorboard_logdir = line.split(':')[-1].strip()
                    break
            
            if tensorboard_logdir is None:
                logging.warning(f"Evaluation {i}: Could not find tensorboard directory")
                continue
            
            tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
            

            if 'consecutive_successes' in tensorboard_logs:
                max_success = max(tensorboard_logs['consecutive_successes'])
                reward_code_final_successes.append(max_success)
            

            if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                gt_reward = np.array(tensorboard_logs["gt_reward"])
                gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                reward_code_correlations_final.append(reward_correlation)
        
        except Exception as e:
            logging.error(f"Evaluation {i}: Error processing results - {e}")
            continue
    

    if reward_code_final_successes:
        logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes):.4f}, "
                    f"Std: {np.std(reward_code_final_successes):.4f}, "
                    f"Raw: {reward_code_final_successes}")
    else:
        logging.warning("No successful evaluations recorded")
    
    if reward_code_correlations_final:
        logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final):.4f}, "
                    f"Std: {np.std(reward_code_correlations_final):.4f}, "
                    f"Raw: {reward_code_correlations_final}")
    

    evaluation_results = {
        "best_code_path": max_reward_code_path,
        "training_success": float(global_best['success']),
        "training_correlation": float(global_best['reward_corr']),
        "eval_successes": [float(s) for s in reward_code_final_successes],
        "eval_correlations": [float(c) for c in reward_code_correlations_final],
        "eval_success_mean": float(np.mean(reward_code_final_successes)) if reward_code_final_successes else 0.0,
        "eval_success_std": float(np.std(reward_code_final_successes)) if reward_code_final_successes else 0.0,
        "eval_correlation_mean": float(np.mean(reward_code_correlations_final)) if reward_code_correlations_final else 0.0,
        "eval_correlation_std": float(np.std(reward_code_correlations_final)) if reward_code_correlations_final else 0.0
    }
    
    eval_results_file = eval_dir / "evaluation_results.json"
    with open(eval_results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
    
    logging.info(f"Saved evaluation results to {eval_results_file}")

if __name__ == "__main__":
    main()