import argparse
import json
import logging
import os
import re
import sys
import csv
from model import GPT, CodeLLAMA, StarChat

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)


MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
tokenizer_name = RobertaTokenizer.from_pretrained("/XXX/XXX/XXX/", local_files_only=True )

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def write_formatted_txt(results, language, eval_model):
    file_path = f'./result/{language}/{eval_model}-read.txt'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as fw:
        for i, item in enumerate(results):
            formatted_str = f"-----------------样本：{i} -----------------\n"
            formatted_str += f"idx: {item['idx']}\n"
            formatted_str += f"code:\n{item['code']}\n"

            formatted_str += f"reasons:\n{item['reasons']}\n\n"

            formatted_str += "\n"
            fw.write(formatted_str)

def evaluate_all(input, model):
    prompt = (
        "You are an experienced software developer. I will provide you with a  code snippet and four candidate  names. Your task is to evaluate each name based on the code snippet and the following criteria (C): "
        "### Instructions: "
        "For each candidate method name: "
        "1. Carefully read the method body."
        "2. Determine what the method does (write a one-sentence summary). "
        "3. Judge whether the candidate name:"
        "- accurately describes the method's functionality (semantic match),"
        "- is clear and readable (naming style and conventions),"
        "4. Assign a rating from 1 to 5:"
        "- 1 = Very poor (completely unrelated, misleading)"
        "- 2 = Poor (vague or generic)"
        "- 3 = Fair (somewhat descriptive, but not ideal)"
        "- 4 = Good (clear and mostly accurate)"
        "- 5 = Excellent (precise, clear, and idiomatic)"
        "5. Briefly explain your score for each name.\n"       
        "Once your evaluation is complete, present the four results and explanations for each  name in the following format:"
        "**Name 1: <name1>**"
        "......"
        "The following is a code snippet and four candidate  names under evaluation:"
        "The code snippet:"
        "<code>"
        "The  names under evaluation:"
        "<name 1>"
        "<name 2>"
        "<name 3>"
        "<name 4>"
    ) + input
    message = model.ask(prompt)
    TOTAL_FUNCTIONS = 4

    method_names = ['N/A'] * TOTAL_FUNCTIONS
    c1_scores = [0] * TOTAL_FUNCTIONS

    name_pattern = re.findall(r'[*#\s]* Name\s*(\d+)\s*:\s*([^\n*#]+)', message)
    for index_str, name in name_pattern:
        try:
            index = int(index_str) - 1
            if 0 <= index < TOTAL_FUNCTIONS:
                method_names[index] = name.strip()
            else:
                logger.warning(f' index {index + 1} out of expected range')
        except ValueError:
            logger.error(f'Invalid  index: {index_str}')


    c1_only_scores = re.findall(r'C1\s*[:：]?\s*(\d+)', message, re.IGNORECASE)
    for i, c1 in enumerate(c1_only_scores):
        if i < TOTAL_FUNCTIONS:
            c1_scores[i] = int(c1)

    return list(zip(method_names, c1_scores)), message

def evaluate(code, names, file_path, cnt=0, model=None):
    if not file_path.endswith(".jsonl"):
        file_path = file_path.rsplit(".", 1)[0] + ".jsonl"
    mode = 'a' if cnt > 0 else 'w'
    csv_path = file_path.replace(".jsonl", ".csv")
    write_csv_header = not os.path.exists(csv_path) or cnt == 0

    with open(file_path, mode, encoding="utf-8") as f_jsonl, \
         open(csv_path, mode, encoding="utf-8", newline='') as f_csv:
        csv_writer = csv.writer(f_csv)
        if write_csv_header:
            header = ['idx']
            for j in range(len(names[0])):
                header.append(f'c1_{j}')
            csv_writer.writerow(header)

        for i in range(len(code)):
            if i < cnt:
                continue

            prompt_lines = [f"Code:\n{code[i]}"]
            prompt_lines += [f" Name {j + 1}: {names[i][j]}" for j in range(len(names[i]))]
            input_prompt = '\n'.join(prompt_lines)

            try:
                method_data, reasons = evaluate_all(input_prompt, len(names[i]), model)
                method_names, scores_c1 = zip(*method_data)
            except Exception as e:
                logger.error(f"Error at idx {i}: {e}")
                continue

            result = {
                "idx": i,
                "code": code[i],
                "reasons": reasons
            }
            for j in range(len(method_names)):
                result[f"names[{j}]"] = method_names[j]
                result[f"scores_c1[{j}]"] = scores_c1[j]

            f_jsonl.write(json.dumps(result, ensure_ascii=False) + '\n')

            csv_row = [i]
            for j in range(len(method_names)):
                csv_row.append(scores_c1[j])
            csv_writer.writerow(csv_row)

            print(f"[{i}] Done")
def compare_human_eval_and_gpt_eval(language, eval_model='',model=None):
    code = []
    names = []

    with open('/XXX/XXX/XXX/XXX/Python_examples.jsonl', "r",
              encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)

            code_item = js.get("code", "").strip()
            name1 = js.get("Name 1", "").replace('\n', ' ').split('.')[-1].strip()
            name2 = js.get("Name 2", "").replace('\n', ' ').split('.')[-1].strip()
            name3 = js.get("Name 3", "").replace('\n', ' ').split('.')[-1].strip()
            name4 = js.get("Name 4", "").replace('\n', ' ').split('.')[-1].strip()

            code.append(code_item)
            names.append([name1, name2, name3, name4])

    evaluate(code=code, names=names, file_path=f'./result/{language}/{eval_model}.jsonl', cnt=0, model=model)

    file_path = f"./result/{language}/{eval_model}.jsonl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


    with open(file_path, 'r', encoding='utf-8') as f:
        results = [json.loads(line.strip()) for line in f]

    write_formatted_txt(results, language, eval_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--openai_key", default="", type=str)
    parser.add_argument("--max_new_tokens", default=0, type=int)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--log_filename", default='log-eval-llms.txt', type=str)
    parser.add_argument("--n_reasoning_paths", default=1, type=int)
    parser.add_argument("--frequency_penalty", default=0, type=int)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    args.logger = logging.getLogger(__name__)
    fh = logging.FileHandler(args.log_filename)
    args.logger.addHandler(fh)

    MODEL_NAME_OR_PATH = {'gpt-3.5': 'gpt-3.5-turbo',
                          'codellama': '/XXX/XXX/XXX/XXX/XXX/codellama-I',
                          'starchat': '/XXX/XXX/XXX/XXX/XXX/starchat-bert',
                          }
    args.model_name_or_path = MODEL_NAME_OR_PATH[args.model]
    if args.model == 'gpt-3.5':
        model = GPT(args=args)
    elif args.model == 'codellama':
        model = CodeLLAMA(args=args)
    elif args.model == 'starchat':
        model = StarChat(args=args)
    else:
        print('Model not found!')
        sys.exit(0)

    for language in ['python']:
        compare_human_eval_and_gpt_eval(language=language, eval_model=args.model, model=model)
