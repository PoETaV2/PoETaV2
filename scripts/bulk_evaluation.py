# Script to download, convert and evaluate checkpoints.
# Assumes you are logged in with the gcloud cli in an account that is able to access any necessary bucket.

import argparse
import inspect
import json
import os
from collections import defaultdict
from pathlib import Path
from subprocess import run
from typing import List

import pandas as pd
import wandb

def eval_hf_checkpoint_or_api(args, model_config, model_name_or_path=None, checkpoint_number=-1, results_save_dir=None, wandb_run=None):
    

    # Evaluate
    model = model_config.get("model", "gpt")
    if model == "gpt":
        model_args = f"pretrained={model_name_or_path}"
    elif model in ["anthropic", "azure", "chatgpt", "cohere", "deepinfra", "deepseek", "fireworks", "gemini", "google", "maritalk", "maritalk_local", "mistral", "tgi", "together", "vllm"]:
        # If model_name_or_path (i.e., engine) was not passed as an argument, use the one in model config.
        engine = model_name_or_path
        if not engine:
            engine = model_config['engine']
        model_args = f"engine={engine}"

        if model == "maritalk_local":
            chat_mode = model_config.get('chat_mode', True)
            model_args += f",chat_mode={chat_mode}"

        if model_config.get("base_url", None):
            model_args += f",base_url={model_config.get('base_url')}"

    tokenizer_for_lm_eval = model_config.get("tokenizer_for_lm_eval", None)
    revision_for_lm_eval = model_config.get("revision_for_lm_eval", None)
    use_adapters = model_config.get("use_adapters", False)
    device = model_config.get('device', None)
    description_path = model_config.get('description_path', None)
    conversation_template = model_config.get('conversation_template', None)
    prompt_as_single_user_message = model_config.get('prompt_as_single_user_message', False)
    batch_size = model_config.get("batch_size", None)

    dtype_in_config = model_config.get("dtype", None)
    dtype_for_eval = ""
    if dtype_in_config and dtype_in_config in ["fp32", "fp16", "bf16", "int8", "int4"]:
        dtype_for_eval = f",dtype={dtype_in_config}"
    else:
        print(f"WARNING: Invalid dtype {dtype_in_config}, no dtype will be specified for lm-eval")

    task_configs = json.load(open(args.task_configs))
    for task_config in task_configs["tasks"]:
        task_name = task_config['lm_eval_task']
        limit = task_config.get("limit", None)
        response_format = task_config.get("response_format", None)
        num_fewshot = task_config["num_fewshot"]
        prompt_mode = task_configs["prompt_mode"]

        print(f"**** Running evaluation on {task_name} ****")

        tasks = task_name.split(',')
        fname = f"{tasks[0]},{tasks[-1]}" if len(tasks) > 1 else task_name
        results_save_file = Path(results_save_dir, f"{fname}.json")
        if os.path.isfile(results_save_file):
            print(f"INFO: the file {results_save_file} already exists, skipping evaluation and using the one that already exists")
        else:
            os.makedirs(results_save_dir, exist_ok=True)

            eval_command = (
                f"python3 main.py "
                f"--model {model} "
                f"--model_args {model_args}"
                f"{dtype_for_eval}"
                f"{',tokenizer=' + tokenizer_for_lm_eval if tokenizer_for_lm_eval else ''}"
                f"{',adapter=' + hf_checkpoint_save_path if use_adapters else ''}"
                f"{',revision=' + revision_for_lm_eval if revision_for_lm_eval else ''} "
                f"{f'--device {device}' if device else ''} "
                f"--task {task_name} "
                f"--num_fewshot {num_fewshot} "
                f"--prompt_mode {prompt_mode} "
                f"{f'--limit {limit}' if limit else ''} "
                f"--description_dict_path {description_path} "
                f"{f'--conversation_template {conversation_template}' if conversation_template else ''} "
                f"{'--prompt_as_single_user_message' if prompt_as_single_user_message else ''} "
                f"{f'--batch_size {batch_size}' if batch_size else ''} "
                f"--output_path {results_save_file} "
                f"{f'--response_format {response_format}' if response_format else ''} "
                f"--no_cache "
            )
            run(eval_command, shell=True, check=True)

        # Send results to wandb, even if it was already there.
        if model_config.get("log_to_wandb", False):
            # retrieve metrics
            results = json.load(open(Path(results_save_dir, f"{fname}.json")))
            data = defaultdict(lambda: {})
            if task_config["metrics"] == ["all"]:
                for metric_name in results["results"][task_name][prompt_mode]:
                    data[task_name][metric_name] = results["results"][task_name][prompt_mode][metric_name]
            else:
                for metric_name in task_config["metrics"]:
                    for task in task_name.split(','):
                        metric = float(results["results"][task][prompt_mode][metric_name])
                        data[task][metric_name] = metric

            # specify checkpoint step as a value
            data["checkpoint_step"] = checkpoint_number
            wandb.log(data, commit=True)

    # Compute NPM
    wandb_run_name = ""
    if model_config.get("log_to_wandb", False):
        wandb_run_name = f"{model_config['wandb_entity']}/{model_config['wandb_project']}/{args.experiment_name}"

    npm_command = (
        f"python3 scripts/calculate_npm.py "
        f"--results_folder {results_save_dir} "
        f"--task_configs {args.task_configs} "
        f"--checkpoint_number {checkpoint_number} "
    )
    run(npm_command, shell=True, check=True)
    if model_config.get("log_to_wandb", False):
        # log npm results
        with open(Path(results_save_dir, "npm.json")) as f:
            npm_results = json.load(f)
            npm_results["checkpoint_step"] = checkpoint_number
            wandb.log(npm_results, commit=True)

    if wandb_run is not None:
        # Add result files for future audits
        artifact = wandb.Artifact(
            f'result_{checkpoint_number}_{args.experiment_name}',
            type="results",
            metadata={"checkpoint_number": checkpoint_number},
        )
        artifact.add_dir(results_save_dir)
        wandb_run.log_artifact(artifact)


if __name__ == "__main__":

    script_path = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        required=True,
        help="Name of the experiment. E.g. 'exp123_ckpt_2000_poeta'",
    )
    parser.add_argument(
        "--model_config",
        required=True,
        help="Path for a config file containing model parameters. See an example in config/model_hf_hub.json",
    )
    parser.add_argument(
        "--task_configs",
        required=True,
        help="Path for a config file containing tasks parameters. See an example in config/poeta_limited.json",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Name of the model (e.g. sabia-3-2024-09-09) when using API calls.",
    )
    
    args = parser.parse_args()

    model_config = json.load(open(args.model_config))
    task_configs = json.load(open(args.task_configs))

    wandb_run = None
    if model_config.get("log_to_wandb", False):
        # If you are not logged in with the wandb cli, the API key will be prompted by the script.
        wandb_run = wandb.init(
            project=model_config["wandb_project"],
            entity=model_config["wandb_entity"],
            id=args.experiment_name,
            resume="allow")

        # Log configuration files being used
        artifact = wandb.Artifact(f"model_config", type="configFile")
        artifact.add_file(args.model_config)
        wandb_run.log_artifact(artifact)

        artifact = wandb.Artifact(f"task_configs", type="configFile")
        artifact.add_file(args.task_configs)
        wandb_run.log_artifact(artifact)

    results_base_dir = model_config.get("results_save_dir", None) or Path(script_path, "results", args.experiment_name)
    execution_mode = "EXECUTE"
    checkpoint_type = model_config.get("checkpoint_type", "").lower()

    if model_config.get("model_for_inference", None):
        eval_hf_checkpoint_or_api(
            args=args,
            model_config=model_config,
            model_name_or_path=model_config["model_for_inference"],
            results_save_dir=results_base_dir,
            wandb_run=wandb_run)
    else:
        eval_hf_checkpoint_or_api(
            args=args,
            model_config=model_config,
            results_save_dir=results_base_dir,
            wandb_run=wandb_run)
    

    
