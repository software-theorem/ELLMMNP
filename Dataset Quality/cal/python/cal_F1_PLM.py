import re
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)


MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
tokenizer = RobertaTokenizer.from_pretrained("/XXX/XXX/XXX/")

original = []
predict = []


with open('/XXX/XXX/XXX.gold', 'r') as fp:
    for idx, name in enumerate(fp.readlines()):

        if not name:
            original.append([])
            continue

        if '\t' in name:
            name = name.split('\t', 1)[1]
        name = name.lower()
        cleaned = ''.join(c for c in name if c.isalpha() or c == '_')

        parts = cleaned.split('_')
        tokens = [p for p in parts if p.isalpha()]
        if not tokens:
            print(f"[Warning] No valid method name tokens at line {idx}")
        original.append(tokens)


with open('/XXX/XXX/XXX.output', 'r') as f:
    for idx, name in enumerate(f.readlines()):

        if not name:
            original.append([])
            continue

        if '\t' in name:
            name = name.split('\t', 1)[1]
        name = name.lower()
        cleaned = ''.join(c for c in name if c.isalpha() or c == '_')

        parts = cleaned.split('_')
        tokens = [p for p in parts if p.isalpha()]
        if not tokens:
            print(f"[Warning] No valid method name tokens at line {idx}")
        predict.append(tokens)


def evaluate(predictions, original_names):
    true_positive = 0
    false_positive = 0
    false_negative = 0


    for original_name, prediction in zip(original_names, predictions):
        if ''.join(original_name) == ''.join(prediction):
            true_positive += len(original_name)
            continue

        for name in prediction:
            if name in original_name:
                true_positive += 1
            else:
                false_positive += 1
        for name in original_name:
            if name not in prediction:
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



def evaluate_in_batches(original, predict, batch_size=10):
    num_batches = (len(original) + batch_size - 1) // batch_size
    all_f1_scores = []
    total_true_positive = total_false_positive = total_false_negative = 0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(original))
        batch_original = original[start_idx:end_idx]
        batch_predict = predict[start_idx:end_idx]

        true_positive, false_positive, false_negative = evaluate(batch_predict, batch_original)
        total_true_positive += true_positive
        total_false_positive += false_positive
        total_false_negative += false_negative

        _, _, batch_f1 = output(true_positive, false_positive, false_negative)

        print(f"Batch {i + 1}/{num_batches} F1 Score: {batch_f1:.4f}. \n Original: {batch_original}, \nPredict: {batch_predict}")
        all_f1_scores.append(batch_f1)


    overall_precision, overall_recall, overall_f1 = output(total_true_positive, total_false_positive,
                                                           total_false_negative)
    return all_f1_scores, overall_precision, overall_recall, overall_f1



batch_f1_scores, overall_precision, overall_recall, overall_f1 = evaluate_in_batches(original, predict, batch_size=1)


output_str = f"Overall Precision: {overall_precision:.4f}, Overall Recall: {overall_recall:.4f}, Overall F1 Score: {overall_f1:.4f}"
print(output_str)

