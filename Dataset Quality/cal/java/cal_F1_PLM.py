import re
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)


MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}


config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
tokenizer = RobertaTokenizer.from_pretrained("/XXX/XXX/XXX/")


original = []
predict = []


with open('/XXX/XXX/XXX.gold', 'r') as fp:
    for name in fp.readlines():

        res = re.findall(r"[A-Za-z]+", name)
        if not res:
            res = []
        else:
            res = ''.join(res).lower()
        original.append(tokenizer.tokenize(res))

with open('/XXX/XXX/XXX.output', 'r') as fp:
    for name in fp.readlines():

        res = re.findall(r"[A-Za-z]+", name)
        if not res:
            res = []
        else:
            res = ''.join(res).lower()
        predict.append(tokenizer.tokenize(res))


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


true_positive, false_positive, false_negative = evaluate(predict, original)
precision, recall, f1_score = output(true_positive, false_positive, false_negative)


output_str = f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}"


print(output_str)

