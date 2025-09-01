import collections
import itertools
import pathlib
import random
import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
import numpy as np
import os
import json
from lm_eval.utils import positional_deprecated, run_task_tests
from tqdm import tqdm

@positional_deprecated
def simple_evaluate(model, model_args=None, tasks=[],
                    num_fewshot=0, prompt_modes=['dynamic-random'], 
                    batch_size=None, device=None, no_cache=False,
                    limit=None, bootstrap_iters=100000,
                    description_dict=None, conversation_template=None,
                    prompt_as_single_user_message=False,
                    check_integrity=False, output_dir=None):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string. 
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int, optional
        Batch size for model
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description` 
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param output_dir: str
        Directory to save results to
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"

   
    if isinstance(model, str):
        if model_args is None: model_args = ""
        
        # no response format
        lm = lm_eval.models.get_model(model).create_from_arg_string(model_args, {
            'batch_size': batch_size, 'device': device
        })
    else:
        assert isinstance(model, lm_eval.base.LM)
        lm = model

    if not no_cache:
        lm = lm_eval.base.CachingLM(
            lm, 'lm_cache/' + model + '_' + model_args.replace('=', '-').replace(',', '_').replace('/', '-') + '.db'
        )
    
    if isinstance(lm, lm_eval.base.CachingLM):
        model_category = lm.lm.MODEL_CATEGORY
    else:
        model_category = lm.MODEL_CATEGORY
    
    print(f"Selected Model is a {model_category}")

    task_dict = lm_eval.tasks.get_task_dict(tasks)
    
    # let each task know what kind of model we are using for inference
    for task in task_dict.values():
        task.set_inference_model_category(model_category)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        prompt_modes=prompt_modes,
        limit=limit,
        description_dict=description_dict,
        conversation_template=conversation_template,
        prompt_as_single_user_message=prompt_as_single_user_message,
        output_dir=output_dir,
    )

    # add info about the model and few shot config
    results["config"] = {
        "model": model,
        "model_args": model_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "device": device,
        "no_cache": no_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
    }

    return results


def write_sample(output_dir, task_name, prompt_mode, doc_id, pred, req_obj, metrics):
    with open(os.path.join(output_dir, f"{task_name}_{prompt_mode}_samples.txt"), "a", encoding="utf-8") as f:
        f.write("*" * 50)
        f.write(f"doc_id: {doc_id}\n")
        f.write("------------------------prompt------------------------------------\n")

        f.write(f"{req_obj.args[0]}\n")
        f.write("--" * 10 + "Prediction" + "--" * 10 + "\n")

        f.write(f"{pred}\n")
        f.write("##" * 50 + "\n")
        pretty_print_metrics=json.dumps(metrics, indent=4, ensure_ascii=False)
        f.write(f"Metrics: {pretty_print_metrics}\n")


@positional_deprecated
def evaluate(lm, task_dict, provide_description=None, num_fewshot=0, 
        prompt_modes=['dynamic-random'], limit=None, bootstrap_iters=100000, 
        description_dict=None, conversation_template=None, prompt_as_single_user_message=False, output_dir=None
    ):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param provide_description: bool
        Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description` 
    :return
        Dictionary of results
    """
    # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces

    # TODO: todo: implement proper description-providing system
    assert not provide_description  # not implemented.
    if provide_description is not None:
        # nudge people to not specify it at all
        print("WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for task_name in task_dict:
            for prompt_mode in prompt_modes:
                # remove old samples
                if os.path.isfile(os.path.join(output_dir, f"{task_name}_{prompt_mode}_samples.txt")):
                    os.remove(os.path.join(output_dir, f"{task_name}_{prompt_mode}_samples.txt"))

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if(task.has_validation_docs() or task.has_test_docs())
    ]

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    # If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
    # memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
    # over-engineering is bad (or we could make it write the requests to disk and then read them back out again
    #  - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}

    # get lists of each type of request
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
        elif task.has_validation_docs():
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        description = ""
        if description_dict:
            if task_name in description_dict:
                description = description_dict[task_name]
            elif task_name.replace("_greedy", "") in description_dict:
                description = description_dict[task_name.replace("_greedy", "")]

        results[task_name] = collections.defaultdict(dict)

        # the requests will be separated for each prompt_mode
        for prompt_mode in prompt_modes:

            # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
            task_docs = list(task_doc_func())
            rnd = random.Random()
            rnd.seed(42)
            rnd.shuffle(task_docs)

            for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
                docs[(task_name, prompt_mode, doc_id)] = doc
                ctx = task.fewshot_context(
                    doc=doc,
                    num_fewshot=num_fewshot,
                    prompt_mode=prompt_mode,
                    rnd=rnd,
                    description=description,
                    conversation_template=conversation_template,
                    prompt_as_single_user_message=prompt_as_single_user_message
                )
                reqs = task.construct_requests(doc, ctx)
                if not isinstance(reqs, (list, tuple)):
                    reqs = [reqs]
                for i, req in enumerate(reqs):
                    requests[req.request_type].append(req)
                    # i: index in requests for a single task instance
                    # doc_id: unique id that we can get back to a doc using `docs`
                    requests_origin[req.request_type].append((i, task_name, prompt_mode, doc, doc_id))

    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)
    
    # execute each type of request
    for reqtype, reqs in requests.items():
        # TODO: right now, this code runs multiple separate LM requests for multiple Requests differing
        #       only in index. We could implement some kind of caching, but that would be more of a band-aid
        #       solution. we could also implement some kind of auto-grouping here;
        #       they should end up next to each other.

        print("Running", reqtype, "requests")
        resps = getattr(lm, reqtype)([req.args for req in reqs])
        resps = [x if req.index is None else x[req.index] for x, req in zip(resps, reqs)]
        

        for (resp, (i, task_name, prompt_mode, doc, doc_id), req) in zip(resps, requests_origin[reqtype], reqs):
            if output_dir and reqtype == "greedy_until":
                process_res_queue[(task_name, prompt_mode, doc_id)].append((i, resp, req, reqtype))
            else:
                # if output_dir is not defined or if the request is not greedy_until, we don't need to record the request object, since we are not logging it
                process_res_queue[(task_name, prompt_mode, doc_id)].append((i, resp, None, reqtype))
    
    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    # group by task_name, prompt_mode
    
    grouped_task_prompt_mode = collections.defaultdict(list)
    
    # grouping by task is convenient to show evaluation progress, when evaluating with llm
    # this may take a while
    for (task_name, prompt_mode, doc_id), requests in process_res_queue.items():
        grouped_task_prompt_mode[(task_name, prompt_mode)].append((requests, doc_id))
        
    for (task_name, prompt_mode), requests in grouped_task_prompt_mode.items():
        
        task=task_dict[task_name]
        
        
        llm_judged_task = False
        
        # show tqdm only for tasks that are evaluated with llm as judge
        for request_list, doc_id in tqdm(requests, desc=f"Evaluating {task_name} {prompt_mode}", disable=not llm_judged_task):
            
            request_list.sort(key=lambda x: x[0])
            
            preds = [x[1] for x in request_list]         
            doc = docs[(task_name, prompt_mode, doc_id)]
            metrics = task.process_results(doc, preds)
            
            _, pred, req_obj, reqtype = request_list[0]
            
            if output_dir and reqtype=="greedy_until":
                write_sample(output_dir, task_name, prompt_mode, doc_id, pred=pred, req_obj=req_obj, metrics=metrics)

            for metric, value in metrics.items():
                vals[(task_name, prompt_mode, metric)].append(value)
        
    
    # aggregate results
    for (task_name, prompt_mode, metric), items in vals.items():
        task = task_dict[task_name]
        try:
            results[task_name][prompt_mode][metric] = task.aggregation()[metric](items)
        except KeyError:
            # allows tasks to not define an aggregation method for a metric
            # useful for logging extra metrics that don't need aggregation
            continue
        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this
        stderr = lm_eval.metrics.stderr_for_metric(
            metric=task.aggregation()[metric],
            bootstrap_iters=min(bootstrap_iters, 1000) if metric in ["bleu", "chrf", "ter"] else bootstrap_iters,
        )
        if stderr is not None:
            results[task_name][prompt_mode][metric + "_stderr"] = stderr(items)
    
    for task_name in task_dict:
        # Count how many docs were actually evaluated
        task_doc_count = len([1 for key in process_res_queue.keys() if key[0] == task_name])
        for prompt_mode in prompt_modes:
            # Save the count in the results
            results[task_name][prompt_mode]["num_examples"] = task_doc_count
    
    return {
        "results": dict(results),
        "versions": dict(versions)
    }


def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Prompt", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Prompt", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for task, dic_ in result_dict["results"].items():
        version = result_dict["versions"][task]
        for prompt, dic in dic_.items():
            for m, v in dic.items():
                if m.endswith("_stderr"):
                    continue

                if m + "_stderr" in dic:
                    se = dic[m + "_stderr"]
                    values.append([task, prompt, version, m, '%.4f' % v, '±', '%.4f' % se])
                else:
                    values.append([task, prompt, version, m, '%.4f' % v, '', ''])
                task = ""
                prompt = ""
                version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    return md_writer.dumps()
