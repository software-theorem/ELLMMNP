import json
import os
import logging
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
tokenizer = RobertaTokenizer.from_pretrained("/home/david/LW/Comprehensive_Evaluation_LLM/model/codeT5/")
from util.remove_comments import remove_comments_and_docstrings


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def add_lang_by_task(target_str, task, sub_task):
    if task == 'mnp':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

        # Log the first 5 examples during training for debugging

    if example_index < 5 and stage == 'train':
        logger.info("*** Example {} ***".format(example_index))
        logger.info("source_str: {}".format(source_str))
        logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
        logger.info("target_str: {}".format(target_str))
        logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples

def replace_first_func_name(raw_code, func_name):
    idx = raw_code.find(func_name)
    if idx == -1:
        return raw_code  # 函数名未找到，不替换
    return raw_code[:idx] + " " + raw_code[idx + len(func_name):]
def read_mnp_examples(filename, args):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # if idx>50:
            #     break
            line = line.strip()
            if not line:  # 跳过空行
                continue
            js = json.loads(line)

            func_name = js['func_name'].replace('\n', ' ').split('.')[-1].strip()

            raw_code = js["code"]
            code_without_name = replace_first_func_name(raw_code, func_name)
            try:
                clean_code = remove_comments_and_docstrings(source=code_without_name, lang="python")
            except Exception as e:
                print(f"[Warning] Failed to remove comments for idx {idx}: {e}")
                clean_code = code_without_name  # Fall back to original code if comment removal fails

            source = clean_code
            source = ' '.join(source.strip().split())

            func_name = func_name.lower().rstrip('0123456789')
            func_token = tokenizer.tokenize(func_name)

            nl = ' '.join(func_token).replace('\n', '')
            nl = ' '.join(nl.strip().split())

            examples.append(
                Example(
                    idx=idx,
                    source=source,
                    target=nl,
                )
            )
    return examples
# def read_mnp_examples(filename, data_num):
#     """Read examples from filename."""
#     examples = []
#     with open(filename, encoding="utf-8") as f:
#         for idx, line in enumerate(f):
#             line = line.strip()
#             js = json.loads(line)
#             if 'idx' not in js:
#                 js['idx'] = idx
#             code = ' '.join(js['code_tokens']).replace('\n', ' ')
#             code = ' '.join(code.strip().split())
#             nl = ' '.join(js['docstring_tokens']).replace('\n', '')
#             nl = ' '.join(nl.strip().split())
#             examples.append(
#                 Example(
#                     idx=idx,
#                     source=code,
#                     target=nl,
#                 )
#             )
#             if idx + 1 == data_num:
#                 break
#     return examples
# def read_summarize_examples(filename, data_num):
#     """Read examples from filename and handle any lines that fail to be read."""
#     examples = []
#     skipped_lines = 0  # Counter for skipped lines
#     file_path = os.path.abspath(filename)  # Get the absolute file path
#     # logger.info(f"Reading file: {file_path}")  # Log the file path
#
#     with open(filename, encoding="utf-8") as f:
#         for idx, line in enumerate(f):
#             line = line.strip()
#
#             try:
#                 js = json.loads(line)
#             except json.decoder.JSONDecodeError as e:
#                 # Log the error with file path and line number
#                 # logger.error(f"Error parsing line {idx} in file {file_path}")
#                 skipped_lines += 1  # Increment the skipped lines counter
#                 continue  # Skip this line and proceed with the next one
#
#             func_name = js['func_name'].replace('\n', ' ')
#             if '.' in func_name:
#                 num = func_name.find('.')
#                 func_name = func_name[num + 1:]
#             source = ' '.join(js['code_tokens']).replace('\n', ' ').replace(func_name, ' ')
#             source = ' '.join(source.strip().split())
#             func_name = func_name.lower().rstrip('0123456789')
#             func_token = tokenizer.tokenize(func_name)
#             nl = ' '.join(func_token).replace('\n', '')
#             nl = ' '.join(nl.strip().split())
#
#             examples.append(
#                 Example(
#                     idx=idx,
#                     source=source,
#                     target=nl,
#                 )
#             )
#
#     # Log how many lines were skipped
#     # logger.info(f"Total skipped lines: {skipped_lines}")
#
#     return examples


def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data
