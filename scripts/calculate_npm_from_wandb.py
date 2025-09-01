import wandb
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os


mapping = {
    "faquad-f1": {"random_score": 0, "max_score": 100, "translated": False},
    "smallmkqa-best_f1": {"random_score": 0, "max_score": 100, "translated": True},
    "tweetsentbr-f1-macro": {"random_score": 32.4, "max_score": 100, "translated": False},
    "assin_rte-f1": {"random_score": 50, "max_score": 100, "translated": False},
    "enem-acc": {"random_score": 0.20, "max_score": 1, "translated": False},
    "agnews_pt-acc": {"random_score": 25, "max_score": 100, "translated": True},
    "imdb_pt-acc": {"random_score": 0.5, "max_score": 1, "translated": True},
    "smallmassive-f1-macro": {
        # 18 classes
        "random_score": 0.5847,
        "max_score": 100,
        "translated": True,
    },
    "sst2_pt-acc": {"random_score": 0.5, "max_score": 1, "translated": True},
    "boolq_pt-acc": {"random_score": 0.5, "max_score": 1, "translated": True},
    "bat-acc": {"random_score": 0.25, "max_score": 1, "translated": False},
    "enem_2022-acc": {"random_score": 0.20, "max_score": 1, "translated": False},
    "wsc285_pt-acc": {"random_score": 0.50, "max_score": 1, "translated": True},
    "assin_sts-pearson": {"random_score": 0.0, "max_score": 1, "translated": False}

}

mapping_greedy = {
    "faquad-f1": {"random_score": 0, "max_score": 100, "translated": False},
    "smallmkqa_greedy-best_f1": {"random_score": 0, "max_score": 100, "translated": True},
    "tweetsentbr_greedy-f1-macro": {"random_score": 32.4, "max_score": 100, "translated": False},
    "assin_rte_greedy-f1": {"random_score": 50, "max_score": 100, "translated": False},
    "enem_greedy-acc": {"random_score": 0.20, "max_score": 1, "translated": False},
    "agnews_pt_greedy-acc": {"random_score": 25, "max_score": 100, "translated": True},
    "imdb_pt_greedy-acc": {"random_score": 0.5, "max_score": 1, "translated": True},
    "smallmassive_greedy-f1-macro": {
        # 18 classes
        "random_score": 0.5847,
        "max_score": 100,
        "translated": True,
    },
    "sst2_pt_greedy-acc": {"random_score": 0.5, "max_score": 1, "translated": True},
    "boolq_pt_greedy-acc": {"random_score": 0.5, "max_score": 1, "translated": True},
    "bat_greedy-acc": {"random_score": 0.25, "max_score": 1, "translated": False},
    "enem_2022_greedy-acc": {"random_score": 0.20, "max_score": 1, "translated": False},
    "wsc285_pt_greedy-acc": {"random_score": 0.50, "max_score": 1, "translated": True},
    "assin_sts_greedy-pearson": {"random_score": 0.0, "max_score": 1, "translated": False}

}


def generate_csv_from_accumulator_dict(metric_per_prompt_mode, checkpoint_ids, output_dir):
    """
    Generate a csv file with the NPM for each prompt mode.

    Args:
        metric_per_prompt_mode (dict): Dictionary with the accuracy for each prompt mode.
        checkpoint_ids (list): List with the checkpoint ids.
        output_dir (str): Path to the output directory.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    data_for_avg_df = [checkpoint_ids]
    columns_for_avg_df = ["step"]
    num_datasets = []
    columns_num_datasets = []
    # for each prompt mode
    for prompt_mode in metric_per_prompt_mode:
        data_for_prompt_mode = []
        columns = list(metric_per_prompt_mode[prompt_mode].keys())
        num_datasets.append(len(columns))
        for feature in columns:
            data_for_prompt_mode.append(acc[prompt_mode][feature])

        a = np.array(data_for_prompt_mode)
        if a.size == 0:
            average_metric = np.zeros(len(checkpoint_ids))
        else:
            average_metric = a.mean(axis=0)

        data_for_avg_df.append(average_metric)
        columns_for_avg_df.append(f"NPM-{prompt_mode}")
        columns_num_datasets.append(f"{prompt_mode}-Datasets")

        # aggregate data in dataframe
        columns = ["step"] + columns
        data_for_prompt_mode = [steps] + data_for_prompt_mode
        data_for_prompt_mode = np.array(data_for_prompt_mode).T
        prompt_mode_df = pd.DataFrame(data_for_prompt_mode, columns=columns)
        # sort by step
        prompt_mode_df = prompt_mode_df.sort_values(by=["step"])
        output_csv_path = Path(output_dir, f"{prompt_mode}_data.csv")
        prompt_mode_df.to_csv(output_csv_path, index=False)

    # aggregate data in dataframe
    data_for_avg_df = np.array(data_for_avg_df).T
    avg_df = pd.DataFrame(data_for_avg_df, columns=columns_for_avg_df)
    output_csv_path = Path(output_dir, f"NPM_data.csv")
    avg_df.to_csv(output_csv_path, index=False)

    # Export dataframe with number of datasets used in NPM calculation
    num_datasets = np.array(num_datasets).reshape(1, -1)
    num_datasets_df = pd.DataFrame(num_datasets, columns=columns_num_datasets)
    output_csv_path = Path(output_dir, f"num_datasets_data.csv")
    num_datasets_df.to_csv(output_csv_path, index=False)


parser = argparse.ArgumentParser()
parser.add_argument("--run_name", default=None, type=str, required=True, help="wandb run name")
parser.add_argument("--entity", default="Zanez", type=str, help="Wandb entity.")
parser.add_argument("--project", default="evaluations", type=str, required=False, help="Wandb project.")
parser.add_argument("--greedy", action="store_true", help="If set, use greedy mapping.")

parser.add_argument(
    "--consider_prompts",
    default="dynamic_similar",
    type=str,
    choices=["dynamic_similar", "dynamic_random", "fixed", "manual", "all"],
    required=False,
    help="Types of prompt to consider.",
)

# add mode as arg
parser.add_argument("--run_format",
    default="recommended",
    type=str,
    choices=["recommended", "legacy"],
    required=False,
    help="Naming convetion used in the run, usually older runs used the legacy format while newer ones use the recommended",)

args = parser.parse_args()

api = wandb.Api()
# Project is specified by <entity/project-name>
run = api.run(f"{args.entity}/{args.project}/{args.run_name}")
run_history = run.history()

if args.greedy:
    mapping=mapping_greedy

output_dir = f"agg_results/{args.run_name}"
os.makedirs(output_dir, exist_ok=True)

acc = {}
acc_translated = {}
acc_non_translated = {}

if args.consider_prompts == "all":
    prompt_modes = ["fixed", "dynamic_random", "dynamic_similar", "manual"]
else:
    prompt_modes = [args.consider_prompts]

# initialize accumulator dictionaries
for prompt_mode in prompt_modes:
    acc[prompt_mode] = {}
    acc_translated[prompt_mode] = {}
    acc_non_translated[prompt_mode] = {}

metric_separator = "."
if args.run_format == "recommended":
    dataset_separator = "/"
    steps = run_history["checkpoint_step"].to_numpy()
    composed_prefixes = []
else:
    dataset_separator = "_"
    steps = run_history["_step"].to_numpy()
    composed_prefixes = ["assin", "imdb", "agnews"]

summary_list = []
config_list = []
name_list = []

for feature_name in run_history.columns:
    dataset = feature_name.split(dataset_separator)[0]
    prompt_and_metric_string = "_".join(feature_name.split(dataset_separator)[1:])

    if dataset in composed_prefixes:
        # get correct dataset names such as "assin_rte" and "imdb_pt"
        dataset = "_".join(feature_name.split(dataset_separator)[:2])

        prompt_and_metric_string = "_".join(feature_name.split(dataset_separator)[2:])

    # <prompt_mode>.<metric>
    metric = prompt_and_metric_string.split(metric_separator)[-1]
    prompt_mode = prompt_and_metric_string.split(metric_separator)[0]

    # due to inconsistency in naming, some runs have "dinamic" in the name instead of "dynamic"
    prompt_mode = prompt_mode.replace("dinamic", "dynamic")
    key = f"{dataset}-{metric}"

    if key in mapping:
        random_score = mapping[key]["random_score"]
        max_score = mapping[key]["max_score"]
        feature_data = run_history[feature_name].to_numpy()

        # normalize data
        adjusted_feature_data = (feature_data - random_score) / (max_score - random_score)

        acc[prompt_mode][key] = adjusted_feature_data

        # accumulate separately for translated and non translated datasets
        if mapping[key]["translated"] == True:
            acc_translated[prompt_mode][key] = adjusted_feature_data
        else:
            acc_non_translated[prompt_mode][key] = adjusted_feature_data


generate_csv_from_accumulator_dict(acc, steps, os.path.join(output_dir, "NPM_all"))
generate_csv_from_accumulator_dict(acc_translated, steps, os.path.join(output_dir, "NPM_translated"))
generate_csv_from_accumulator_dict(acc_non_translated, steps, os.path.join(output_dir, "NPM_non_translated"))