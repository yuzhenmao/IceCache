import argparse
import json
import os
import logging
import re
import sys
import time
import torch
import numpy as np
import datasets
import transformers
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from datasets import load_dataset
from typing import Any, Callable, Dict, Sequence, cast, List
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from transformers import LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import random


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(cmd_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="longchat-v1.5-7b-32k",
        choices=[
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "vicuna-v1.5-7b-16k",
            "mistral-7b-inst",
            "llama-3-8b-inst-64k",
            "llama-3-8b-inst-1048k",
            "llama-3.1",
            "qwen32",
            "qwen3-4b"
        ],
    )
    ap.add_argument("--name", type=str, default="default")
    ap.add_argument("--prompt_file", type=str, default="gsm8k_prompt_formal.txt", help="")
    ap.add_argument("--icecache", action="store_true", help="Enable IceCache")
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--page-budgets", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=None, help="")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="")
    ap.add_argument("--exp_name", type=str, default="dafault_exp")
    ap.add_argument("--n-unlimited-layers", type=int, default=2)
    ap.add_argument("--n-max-bytes", type=int, default=40 * (1 << 28))
    ap.add_argument("--n-max-cpu-bytes", type=int, default=80 * (1 << 28))
    ap.add_argument("--page-topks", type=int, default=0)
    ap.add_argument("--n-win-pages", type=int, default=2)
    ap.add_argument("--n-sink-pages", type=int, default=2)
    ap.add_argument("--use-3-stages-gen", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--n_prefetch_layers", type=int, default=0)
    ap.add_argument("--n_reuse_layers", type=int, default=0)
    ap.add_argument("--do_sample", action="store_true", default=False, help="")
    ap.add_argument("--temperature", type=float, default=0.0, help="")
    ap.add_argument("--top_k", type=int, default=50, help="")
    ap.add_argument("--top_p", type=float, default=1.0, help="")
    ap.add_argument("--generation_split", type=str, default=MODEL_GENERATION_SPLIT, help="")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "gov_report",
            "qmsum",
            "multi_news",
            "vcsum",
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
            "lcc",
            "repobench-p",
        ],
    )
    args = ap.parse_args(cmd_args)
    args.e = False
    if args.page_budgets < 0:
        args.page_budgets = None
    return args


def load_model_and_tokenizer(path, model_name, device, args):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    if args.icecache:
        from icecache import adapter

        model = AutoModelForCausalLM.from_pretrained(
            path, device_map=device, torch_dtype=torch.bfloat16
        )
        adapter.enable_icecache(
            model, dtype=torch.bfloat16, device=device, **args.__dict__
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    model = model.eval()
    return model, tokenizer



IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

MODEL_GENERATION_SPLIT = "\nQuestion: "
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationSample:
    """Wrapper around format evaluation sample."""

    question: str
    generation: str
    answer: str
    list_from_pred: List[str]
    list_from_answer: List[str]
    pred: float
    label: float
    is_pred_true: bool


@dataclass(frozen=True)
class EvaluationMetrics(DataClassJsonMixin):
    """Wrapper around aggregated evaluation metrics."""

    accuracy: float


@dataclass(frozen=True)
class EvaluationResults(DataClassJsonMixin):
    """Wrapper around evaluation results"""

    samples: List[EvaluationSample]
    metrics: EvaluationMetrics


def evaluate_pred_answer(pred_str, ans_str):
    pattern = "\d*\.?\d+"
    pred_str, ans_str = pred_str.replace(",", ""), ans_str.replace(",", "")
    pred_list = re.findall(pattern, pred_str)
    gold_list = re.findall(pattern, ans_str)
    if len(pred_list) >= 1:
        pred = float(pred_list[-1])
        gold = float(gold_list[-1])
        is_pred_true = pred == gold
    else:
        is_pred_true = False
        pred = None
        gold = float(gold_list[-1])
    return (
        is_pred_true,
        pred,
        pred_list,
        gold,
        gold_list,
    )


def test_answer(pred_str, ans_str):
    pattern = "\d*\.?\d+"
    pred = re.findall(pattern, pred_str)
    if len(pred) >= 1:
        print("#####\n Pred string:", pred_str, "\n pred_list", pred)
        pred = float(pred[-1].replace(",", ""))
        gold = re.findall(pattern, ans_str)
        print("\n Gold_answer", ans_str, "\n gold_list", gold)
        gold = float(gold[-1].replace(",", ""))
        print("\n result", gold, pred, gold == pred)
        return pred == gold
    else:
        return False


def parse_pred_ans(filename):
    with open(filename) as fd:
        lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = "none"
    questions = []
    ans_pred = []
    ans_gold = []
    am_others = []
    for l in lines:
        if l.startswith("Q: "):
            if am is not None and a is not None:
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if test_answer(am, a):
                    acc += 1
            current_mode = "q"
            q = l
            num_q += 1
        elif l.startswith("A_model:"):
            current_mode = "am"
            am = l
        elif l.startswith("A:"):
            current_mode = "a"
            a = l
        # TODO
        elif current_mode == "am" and l.startswith("Question: "):
            current_mode = "am_other"
            am_other = l
        else:
            if current_mode == "q":
                q += l
            elif current_mode == "am":
                am += l
            elif current_mode == "a":
                a += l
            elif current_mode == "am_other":
                am_other += l
            else:
                raise ValueError(current_mode)

    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    am_others.append(am_other)
    if test_answer(am, a):
        acc += 1
    print("######\n num_q %d correct %d ratio %.4f" % (num_q, acc, float(acc / num_q)))
    return questions, ans_pred, ans_gold


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("longbench_config/model2path.json", "r"))
    model2maxlen = json.load(open("longbench_config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name], model_name, device, args
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = "gsm8k_test"
    model_name = args.model
    exp_name = args.exp_name
    eval_dataset = load_dataset('json', data_files=dataset+'.jsonl', split='train')
    
    if not os.path.exists(f"pred/{model_name}/{dataset}"):
        os.makedirs(f"pred/{model_name}/{dataset}")
    if not os.path.exists(f"pred/{model_name}/{dataset}/{exp_name}"):
        os.makedirs(f"pred/{model_name}/{dataset}/{exp_name}")
    if not os.path.exists(f"pred/{model_name}/{dataset}/{exp_name}/generate"):
        os.makedirs(f"pred/{model_name}/{dataset}/{exp_name}/generate")
    if not os.path.exists(f"pred/{model_name}/{dataset}/{exp_name}/evaluate"):
        os.makedirs(f"pred/{model_name}/{dataset}/{exp_name}/evaluate")

    output_dir = Path(f"pred/{model_name}/{dataset}/{exp_name}")
    generation_file = Path(f"pred/{model_name}/{dataset}/{exp_name}/generate/g.jsonl")
    evaluation_result_file = Path(f"pred/{model_name}/{dataset}/{exp_name}/evaluate/e.json")
    
    logging.basicConfig(
        filename=os.path.join(output_dir.resolve(), "log.txt"),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("Preprocessing the dataset.")
    with open(f"{args.prompt_file}", "r") as handle:
        prompt_cot = handle.read()

    dataloader = torch.utils.data.DataLoader(
        cast(torch.utils.data.Dataset, eval_dataset),
        batch_size=1,
    )

    acc_list = []
    all_samples = []
    all_question, all_generation, all_answer = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate GSM8K"):
            questions = batch["question"]
            answers = batch["answer"]
          
            prompts = [
                prompt_cot + "\nQuestion: " + question + "\n"
                for question in questions
            ]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )
            inputs = inputs.to("cuda")

            generate_kwargs = dict(
                return_dict_in_generate=True,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            
            if args.do_sample:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = args.temperature
                generate_kwargs["top_k"] = args.top_k
                generate_kwargs["top_p"] = args.top_p
            else:
                generate_kwargs["do_sample"] = False
                generate_kwargs["temperature"] = 0

            from transformers import StoppingCriteria, StoppingCriteriaList
            import time
            class TimingCriteria(StoppingCriteria):
                def __init__(self):
                    self.timings = [time.time()]

                def __call__(self, input_ids, scores, **kwargs):
                    self.timings.append(time.time())
                    return False  # Never actually stop early

            timer = TimingCriteria()
            
            outputs = model.generate(                
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask, 
                stopping_criteria=StoppingCriteriaList([timer]),
                **generate_kwargs
            )

            token_latencies = [t2 - t1 for t1, t2 in zip(timer.timings, timer.timings[1:])]
            print(f"Prefill latencies: {token_latencies[0]+token_latencies[1]}, Decode latencies: {np.mean(token_latencies[2:])}")

            generations = tokenizer.batch_decode(
                outputs.sequences[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            all_question += questions
            all_generation += generations
            all_answer += answers

            for question, generation, answer in zip(questions, generations, answers):
                is_pred_true, pred, pred_list, gold, gold_list = evaluate_pred_answer(
                    generation.split(args.generation_split)[0], answer
                )
                sample = EvaluationSample(
                    question=question,
                    generation=generation,
                    answer=answer,
                    list_from_pred=pred_list,
                    list_from_answer=gold_list,
                    pred=pred,
                    label=gold,
                    is_pred_true=is_pred_true,
                )
                all_samples.append(sample)

            acc_list.append(sample.is_pred_true)
            print('acc: {:.5f}'.format(np.mean(acc_list)))

        accuracy = sum([sample.is_pred_true for sample in all_samples]) / len(all_samples)
        evaluation_metric = EvaluationMetrics(accuracy=accuracy)
        evaluation_result = EvaluationResults(samples=all_samples, metrics=evaluation_metric)

    logging.info(f"Accuracy: {accuracy}")

    with evaluation_result_file.open("w") as handle:
        json.dump(evaluation_result.to_dict(), handle)

    with generation_file.open("w") as handle:
        for question, generation, answer in zip(all_question, all_generation, all_answer):
            handle.write("Q: %s\nA_model:\n%s\nA:\n%s\n\n" % (question, generation, answer))
