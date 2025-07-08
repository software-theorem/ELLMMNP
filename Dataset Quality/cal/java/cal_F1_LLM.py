import csv
import argparse
import json
import re
import logging
import os
import sys
from tqdm import tqdm
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
tokenizer_name = RobertaTokenizer.from_pretrained("/XXX/XXX/XXX/", local_files_only=True )

def extract_names(file_path, output_path):
    names = []
    no_match_count = 0

    with open(file_path, 'r', encoding='utf-8') as fg:
        for idx, line in enumerate(fg):
            js = json.loads(line.strip())
            match = re.search(r"`([^`]+)`", js.get("prediction", ""))
            if match:
                raw_name = match.group(1)
                filtered_name = re.sub(r'[^a-zA-Z]', '', raw_name)
                name_tokens = tokenizer_name.tokenize(filtered_name.lower())

            else:
                name_tokens = []
                no_match_count += 1
                print(f"[Warning] No method name found at line {idx}")
            names.append({
                "idx": js.get("idx", idx),
                "name": name_tokens
            })

    print(f"[Info] Total no-match count: {no_match_count}")

    with open(output_path, 'w', encoding='utf-8') as out:
        for entry in names:
            out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"[Info] Extracted names written to: {output_path}")
    return names
def extract_names_fewshot(file_path, output_path):
    names = []
    no_match_count = 0

    with open(file_path, 'r', encoding='utf-8') as fg:
        for idx, line in enumerate(fg):
            js = json.loads(line.strip())
            prediction_str = js.get("prediction", "")
            pred_str = js.get("prediction", "").strip()

            filtered_name = re.sub(r'[^a-zA-Z]', '', pred_str)
            if filtered_name:
                name_tokens = tokenizer_name.tokenize(filtered_name.lower())
            else:
                name_tokens = []
                no_match_count += 1
                print(f"[Warning] No valid alphabetic content found at line {idx}")
                print(f"[Warning] Prediction content: {prediction_str}")

            names.append({
                "idx": js.get("idx", idx),
                "name": name_tokens
            })

    print(f"[Info] Total no-match count: {no_match_count}")

    with open(output_path, 'w', encoding='utf-8') as out:
        for entry in names:
            out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"[Info] Extracted names written to: {output_path}")
    return names


def extract_names_fewshot_starcoder(file_path, output_path):
    names = []
    no_match_count = 0

    with open(file_path, 'r', encoding='utf-8') as fg:
        for idx, line in enumerate(fg):
            js = json.loads(line.strip())
            prediction_str = js.get("prediction", "")


            pred_str = prediction_str.split('\n')[0].strip()

            filtered_name = re.sub(r'[^a-zA-Z]', '', pred_str)

            if filtered_name:
                name_tokens = tokenizer_name.tokenize(filtered_name.lower())
            else:
                name_tokens = []
                no_match_count += 1
                print(f"[Warning] No valid alphabetic content found at line {idx}")
                print(f"[Warning] Prediction content: {prediction_str}")

            names.append({
                "idx": js.get("idx", idx),
                "name": name_tokens
            })

    print(f"[Info] Total no-match count: {no_match_count}")

    with open(output_path, 'w', encoding='utf-8') as out:
        for entry in names:
            out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"[Info] Extracted names written to: {output_path}")
    return names
def gold_names(file_path):
    goldnames = []
    no_match_count = 0

    with open(file_path, 'r', encoding='utf-8') as fg:
        for idx, line in enumerate(fg):
            js = json.loads(line.strip())
            if "idx" in js and "gold" in js:
                goldnames.append({
                    "idx": js["idx"],
                    "name": tokenizer_name.tokenize(js["gold"].lower())
                })
            else:
                no_match_count += 1
                print(f"[Warning] Missing keys in line {idx}")

    print(f"[Info] Skipped lines (missing keys): {no_match_count}")
    return goldnames

def evaluate_predictions(original_names, predictions):

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gold_item, pred_item in zip(original_names, predictions):
        original_name = gold_item["name"]
        prediction = pred_item["name"]

        if not isinstance(original_name, list):
            original_name = list(original_name)
        if not isinstance(prediction, list):
            prediction = list(prediction)

        if ''.join(original_name) == ''.join(prediction):
            true_positive += len(original_name)
            continue

        for token in prediction:
            if token in original_name:
                true_positive += 1
            else:
                false_positive += 1

        for token in original_name:
            if token not in prediction:
                false_negative += 1

    return true_positive, false_positive, false_negative

def output(true_positive, false_positive, false_negative):
    if true_positive + false_positive == 0 or true_positive + false_negative == 0:
        return 0, 0, 0
    P = true_positive / (true_positive + false_positive)
    R = true_positive / (true_positive + false_negative)
    if P + R == 0:
        return P, R, 0
    F1 = (2 * P * R) / (P + R)
    return P, R, F1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", default="/XXX/XXX/XXX.jsonl", type=str)
    parser.add_argument("--gold_name_file", default="/XXX/XXX/XXX.jsonl", type=str)
    parser.add_argument("--language", default="XXX", type=str)
    parser.add_argument("--model", default="XXX", type=str)

    args = parser.parse_args()


    dir = '/XXX/XXX/{}/{}/'.format(args.language, args.model)
    if os.path.exists(dir) == False:
        os.makedirs(dir)



    output_path = os.path.join(dir, "XXX.jsonl")
    method_name = extract_names(args.pred_file, output_path)
    # method_name = extract_names_fewshot(args.pred_file, output_path)#starcaht,
    # method_name = extract_names_fewshot_starcoder(args.pred_file, output_path)

    gold_name  = gold_names(args.gold_name_file)

    tp, fp, fn = evaluate_predictions(gold_name, method_name)
    P, R, F1 = output(tp, fp, fn)

    print(f"Precision: {P:.4f}, Recall: {R:.4f}, F1 Score: {F1:.4f}")



if __name__ == '__main__':
    main()

