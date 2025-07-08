import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
tokenizer_name = RobertaTokenizer.from_pretrained("/XXX/XXX/XXX/", local_files_only=True )

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
           torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def load_name_list_with_tokenization(file_path):
    name_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                idx_str, name_str = line.strip().split('\t', 1)
                idx = int(idx_str)
                combined = name_str.replace(' ', '').lower()
                tokens = tokenizer_name.tokenize(combined)
                tokenized_str = ' '.join(tokens)
                name_list.append({"idx": idx, "name": tokenized_str})
            except ValueError:
                print(f"[Warning] Skipped line due to parsing error: {line.strip()}")
                continue

    return name_list

def load_and_concat_tokens_codeT5_no_idx(file_path):
    name_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            token_str = line.lower()
            concat = token_str.replace(' ', '')
            concat_token = tokenizer_name.tokenize(concat)
            joined_tokens = ' '.join(concat_token)
            name_list.append({"idx": idx, "name": joined_tokens})
    return name_list

def compute_cosine_similarity_raw_string(gold_list, pred_list, tokenizer, model, batch_size=32, device='cpu'):
    similarities = []
    model.eval()
    model.to(device)

    gold_texts = [item["name"] for item in gold_list]
    pred_texts = [item["name"] for item in pred_list]
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

    gold_file = "XXX/XXX/XXX.gold"
    pred_file = "/XXX/XXX/XXX.output"
    model_dir = "/XXX/XXX/sbcs"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")


    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)


    gold_list = load_and_concat_tokens_codeT5_no_idx(gold_file)
    pred_list = load_and_concat_tokens_codeT5_no_idx(pred_file)
    # gold_list = load_name_list_with_tokenization(gold_file)
    # pred_list = load_name_list_with_tokenization(pred_file)

    assert len(gold_list) == len(pred_list), "预测值与参考值数量不一致"


    avg_sim = compute_cosine_similarity_raw_string(
        gold_list, pred_list, tokenizer, model, batch_size=64, device=device
    )

    print(f"\n[Result] Average Sample Cosine Similarity: {avg_sim:.4f}")

if __name__ == "__main__":
    main()
