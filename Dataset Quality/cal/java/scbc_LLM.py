import os
import json
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer



tokenizer_name = RobertaTokenizer.from_pretrained("/XXX/XXX/XXX/", local_files_only=True)

def extract_names(file_path):
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
    return names

def extract_names_fewshot(file_path):
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
    return names


def extract_names_fewshot_starcoder(file_path):
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


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
           torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)


def compute_cosine_similarity_raw_string(gold_list, pred_list, tokenizer, model, batch_size=32, device='cpu'):
    similarities = []
    model.eval()
    model.to(device)

    gold_texts = [' '.join(item["name"]) for item in gold_list]
    pred_texts = [' '.join(item["name"]) for item in pred_list]
    idx_list = [item["idx"] for item in gold_list]

    for i in tqdm(range(0, len(gold_texts), batch_size), desc="Computing cosine similarity"):
        gold_batch = gold_texts[i:i+batch_size]
        pred_batch = pred_texts[i:i+batch_size]
        idx_batch = idx_list[i:i+batch_size]

        inputs = gold_batch + pred_batch
        encoded = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model(**encoded)
            embeddings = mean_pooling(output, encoded['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        N = len(gold_batch)
        gold_embeds = embeddings[:N]
        pred_embeds = embeddings[N:]

        sims = F.cosine_similarity(gold_embeds, pred_embeds, dim=1)

        for idx, sim, g, p in zip(idx_batch, sims, gold_batch, pred_batch):
            print(f"[Idx {idx}] CosSim = {sim:.4f} | GOLD: {g} | PRED: {p}")
            similarities.append(sim.item())

    return sum(similarities) / len(similarities) if similarities else 0.0


def main():
    gold_file = "/XXX/XXX/XXX.jsonl"
    pred_file = "/XXX/XXX/XXX.jsonl"
    model_dir = "/XXX/XXX/XXX"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)

    gold_list = gold_names(gold_file)
    # pred_list = extract_names(pred_file)
    # pred_list = extract_names_fewshot(pred_file)
    pred_list = extract_names_fewshot_starcoder(pred_file)

    gold_list = sorted(gold_list, key=lambda x: x['idx'])
    pred_list = sorted(pred_list, key=lambda x: x['idx'])
    assert [g['idx'] for g in gold_list] == [p['idx'] for p in pred_list], "索引不一致"

    avg_sim = compute_cosine_similarity_raw_string(
        gold_list, pred_list, tokenizer, model, batch_size=64, device=device
    )

    print(f"\n[Result] Average Sample Cosine Similarity: {avg_sim:.4f}")


if __name__ == "__main__":
    main()
