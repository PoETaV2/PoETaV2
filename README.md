

# <img src="docs/images/poeta_v2_logo.png" alt="PoETa V2" width="200"/> 
# PoETa V2 Evaluation

This repository contains the necessary code to run POETA V2.

Poeta V2 is a benchmark for evaluating the performance of models in portuguese on a wide range of tasks. Containing more than 40 tasks, and evaluating over 12k examples, we perform the largest evaluation of pretrained LLMs in portuguese to date.


## Results

# POETA v2 Leaderboard

| Model           | Pretrain size | Model size | Compute Cost | Poeta V2 |
|-----------------|---------------|------------|--------------|----------|
| **Curió**       |               |            |              |          |
| Curio 1.1B      | 1T + 100B     | 1.1B       | 1.07         | 14.7     |
| Curio 7B        | 2T + 100B     | 7B         | 14.07        | 34.8     |
| **Sabiá**       |               |            |              |          |
| Sabia           | 1T + 10B      | 7B         | 7            | 32.4     |
| **Llama**       |               |            |              |          |
| Tinyllama 1T    | 1T            | 1.1B       | 1.0          | 10.2     |
| Llama 1 7B      | 1T            | 7B         | 7            | 19.9     |
| Llama 2 7B      | 2T            | 7B         | 14           | 29.5     |
| Llama 2 13B     | 2T            | 13B        | 26           | 41.5     |
| Llama 3.1 8B    | 15T           | 8B         | 120          | 53.5     |
| **Falcon 3**    |               |            |              |          |
| Falcon 3 1B     | 14T           | 1B         | 14           | 18.6     |
| Falcon 3 3B     | 14T           | 3B         | 42           | 38.6     |
| Falcon 3 7B     | 14T           | 7B         | 98           | 58.5     |
| Falcon 3 10B    | 14T           | 10B        | 140          | 63.5     |
| **Qwen 1 & 2**  |               |            |              |          |
| Qwen 1 1.8B     | 2.4T          | 1.8B       | 4.32         | 19.0     |
| Qwen 1 7B       | 2.4T          | 7B         | 16.8         | 41.1     |
| Qwen 2 1.5B     | 7T            | 1.5B       | 10.5         | 38.4     |
| Qwen 2 7B       | 7T            | 7B         | 49           | 58.8     |
| **Qwen 2.5**    |               |            |              |          |
| Qwen 2.5 1.5B   | 18T           | 1.5B       | 27           | 43.0     |
| Qwen 2.5 3B     | 18T           | 3B         | 54           | 53.1     |
| Qwen 2.5 7B     | 18T           | 7B         | 126          | 63.7     |
| Qwen 2.5 14B    | 18T           | 14B        | 252          | 71.0     |
| **Qwen 3**      |               |            |              |          |
| Qwen 3 1.7B     | 36T           | 1.7B       | 61.2         | 45.1     |
| Qwen 3 4B       | 36T           | 4B         | 144          | 59.2     |
| Qwen 3 8B       | 36T           | 8B         | 288          | 64.7     |
| Qwen 3 14B      | 36T           | 14B        | 503          | 70.5     |
| **Commercial**  |               |            |              |          |
| GPT-4.1         | -             | -          | -            | 76.2     |
| GPT-4o          | -             | -          | -            | 75.2     |
| Sabia 3         | -             | -          | -            | 72.2     |

## Setup

### Clone repository

```bash
git clone https://github.com/maritaca-ai/poeta-v2-evaluation.git
```

### Install requeriments

Needs Python 3.10. We suggest using conda:
```bash
# Download
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install conda
chmod 777 Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment named lm_eval with python 3.10
conda create -n poeta_v2 python=3.10

# Activate environment
conda activate poeta_v2

# Install lm-eval requirements
cd ~/poeta-v2-evaluation
python3 -m pip install -e .
```


## Run a single task

Poeta V2 is a collection fo more than 40 tasks, if you want to run a single task, you can use the following command:

```bash
YOUR_MODEL_PATH=path/to/your/model # could be a local path or a HuggingFace path
OUTPUT_PATH=path/to/your/output # could be a local path or a HuggingFace path

python main.py --model gpt --model_args pretrained=$YOUR_MODEL_PATH --tasks assin_rte_greedy --num_fewshot 2 --prompt_modes dynamic-random --output_path $OUTPUT_PATH --description_dict_path description.json --no_cache
```

You can find the names for each task in the `configs/poeta_v2_full.json` file.

## Running all tasks

We provide a script to run all poeta v2 tasks. To use it, first, create a config for your model. 

We provide a template config in `configs/hf_model_template.json`, You can edit it's fields to run any HF compatible model.

Additionaly, we provide model configs for some Maritaca AI models, like `configs/model_sabiazinho3.json`, and OpenAI models, like `configs/model_gpt-4.1-mini.json`.

You can run all tasks with the following command. The results will be written in the specified "results_save_dir" in the model config, additionally, if you set "log_to_wandb" to true, the results will be logged to wandb.


```bash
export MARITALK_API_SECRET_KEY='1234_3498_3498_3498' # necessary if running a Maritaca AI model, like sabia-3.1 or sabiazinho-3
export OPENAI_API_SECRET_KEY='sk-proj-1234567890' # necessary if running a OpenAI model, like gpt-4.1

experiment_label="poeta_v2_full_model_X"    
model_config="configs/hf_model_template.json"
task_config="configs/poeta_v2_full.json"

python scripts/bulk_evaluation.py --model_config $model_config --task_config $task_config --experiment_name $experiment_label
```


