"""Script to calculate the Normalized Preferred Metric (NPM) on the Poeta Benchmark, separated in 3 categories:
- "All" - contains the NPM metric for all the datasets
- "Native" - contains the NPM metric for all the datasets except the translated ones
- "Translated" - contains the NPM metric for all the translated datasets

Example usage:

python calculate_poeta_npm.py \
    --results_folder ~/t5x/t5x/scripts/results/npm_exp133/checkpoint_500 \
    --prompt_type fixed \
    --greedy \
    --wandb_run rodrigonogueira/poeta-eval/npm_exp133 \
    --checkpoint_number 500
"""
import argparse
import json
import numpy as np
import os
import wandb


parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_folder", type=str, required=True,
    help="Folder containing subdiretories for each dataset in the Poeta benchmark."
         "Inside these folders, it is expected a file named {prompt_type}.json.")
parser.add_argument("--task_configs", type=str, required=True, help="If set, use greedy mapping.")
parser.add_argument("--wandb_run", default=None, type=str, help="Wandb run name (optional), specified as <entity/project-name/run-name>")
parser.add_argument("--checkpoint_number", default=-1, type=int, help="Checkpoint number.")

args = parser.parse_args()

task_configs = json.load(open(args.task_configs))

if args.wandb_run:
    entity, project, run_name = args.wandb_run.split("/")
    wandb_run = wandb.init(
            entity=entity,
            project=project,
            id=run_name,
            resume="allow")

normalized_metrics = []
raw_metrics = []
task_examples_count = [] 

normalized_metric_per_task = {}

prompt_mode = task_configs['prompt_mode']
for task_config in task_configs["tasks"]:
    task_name = task_config["lm_eval_task"]
        
    tasks = task_name.split(',')
    fname = f"{tasks[0]},{tasks[-1]}" if len(tasks) > 1 else task_name
    
    with open(os.path.join(args.results_folder, f"{fname}.json")) as f:
        data = json.load(f)
        for task in task_name.split(','):
            raw_metric = data["results"][task][prompt_mode][task_config["preferred_metric"]]
            normalized_metric = 100 * (raw_metric - task_config["random_score"]) / (task_config["max_score"] - task_config["random_score"])
            normalized_metrics.append((normalized_metric, task_config["translated"]))
            normalized_metric_per_task[task_name] = normalized_metric
            raw_metrics.append((100 * raw_metric, task_config["translated"]))
            
            # Get the number of examples from the results
            task_examples_count.append(data["results"][task][prompt_mode].get("num_examples", 1))

# Aggregate
npm = np.mean([metric for metric, _ in normalized_metrics])
npm_translated = np.mean([metric for metric, translated in normalized_metrics if translated])
npm_native = np.mean([metric for metric, translated in normalized_metrics if not translated])


# Weighted average
total_examples = sum(task_examples_count)
print("Total examples: ", total_examples)

print("           All | Translated | Native")
print(f"NPM:     {npm:.2f} | {npm_translated:.2f} | {npm_native:.2f}")

results = {
    "NPM.All": npm, 
    "NPM.Translated": npm_translated, 
    "NPM.Native": npm_native,
    "NPMPerTask": normalized_metric_per_task
}

with open(f"{args.results_folder}/npm.json", "w") as fout:
    json.dump(results, fout, indent=4)

if args.wandb_run:
    results["checkpoint_step"] = args.checkpoint_number
    wandb.log(results, commit=True)