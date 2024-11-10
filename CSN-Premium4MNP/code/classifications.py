
import json
import re
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize   #导入 分句、分词模块
from nltk import pos_tag
from collections import Counter, defaultdict
# nltk.download('punkt')
# nltk.download('wordnet')

def read_sample(filename):
    samples = []
    with open(filename, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
                # if idx == 100:
                #     break
                line = line.strip()
                line = json.loads(line)

                name = line["func_name"].replace('\n', ' ')
                if "." in name:
                    name = name.split(".")[-1]  # 获取 '.' 后的内容
                samples.append({"idx": idx, "func_name": name})
    return samples

def camel_case_split(identifier):
    # 分割 CamelCase 风格的标识符
    matches = re.finditer(r'.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def snake_case_split(identifier):
    # 分割 snake_case 风格的标识符
    # return identifier.split('_')
    return [part for part in identifier.split('_') if part]

def split_identifier(identifier):
    # 根据标识符的格式选择合适的分割方式
    if identifier.strip() == "":
        print("字符串为空串")
        return []
    return snake_case_split(identifier) if '_' in identifier else camel_case_split(identifier)

def load_abbreviation_dict_from_json(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        abbreviation_dict = json.load(file)
        # 确保所有全称和缩写都是小写，以便匹配不区分大小写
        return {full_form.lower(): [abbr.lower() for abbr in abbreviations]
                for full_form, abbreviations in abbreviation_dict.items()}

# 加载词库
if __name__ == "__main__":
    # filename = "python_test.jsonl"python_test test_csn-java.jsonl
    # filename = "/home/david/LW/data/java/clean_test.jsonl"
    filename = "F:/fuwuqi/A_Comprehensive_Evaluation_LLM_MNP/chuli/oridata/python/train.jsonl"
    out_path = "F:/fuwuqi/A_Comprehensive_Evaluation_LLM_MNP/chuli/extradata/python/train/train"
    # filename = "clean_test.jsonl"
    # out_path = "./123/test"
    samples = read_sample(filename)
    result = []
    alltoken = []
    pos_dict = defaultdict(list)

    for sample in samples:
        sub_token = split_identifier(sample['func_name'])
        alltoken = alltoken + sub_token
        alltoken = list(set(filter(None, alltoken)))
        result.append({
            "idx": sample['idx'],
            "func_name": sample['func_name'],
            "SubToken": sub_token
        })

    Single = []
    num_Single = 0
    # Single noun or preposition and Broad mcaning

    Broad_mcaning = []
    num_Broad_mcaning = 0
    Obscure = []
    num_Obscure = 0
    # 2)Obscure abbreviation
    Digits = []
    num_Digits = 0
    # 4)Digits in word
    Incon = []

    # 5)Inconsistent naming
    tokens = []

    for item in result:

        # 1) Single noun or preposition Examples: text.from.ok
        # 3)Broad mcaning Examples: parse, delete, copy
        # 筛选条件:寻找只有一个单词的SubToken，SubToken": ["create"]}
        if len(item['SubToken']) == 1:
            token = item['SubToken'][0]
            word_tokens = word_tokenize(token)
            tags = pos_tag(word_tokens)
            for word, pos in tags:
                # 1) Single noun or preposition Examples: text.from.ok
                if re.match(r'^(NN|CC|IN)', pos):
                    Single.append({
                        "idx": item['idx'],
                        "func_name": item['func_name'],
                        "SubToken": item['SubToken']
                    })
                    num_Single = num_Single + 1
                # 3)Broad mcaning Examples: parse, delete, copy
                if re.match(r'^(VB)', pos):
                    Broad_mcaning.append({
                        "idx": item['idx'],
                        "func_name": item['func_name'],
                        "SubToken": item['SubToken']
                        })
                    num_Broad_mcaning = num_Broad_mcaning + 1
            # continue
        # 4)Digits in word Examples: frame3
        # 筛选条件:寻找存在数字的SubToken，SubToken": ["put11"
        for token in item['SubToken']:
            if re.search(r'\d', token):
                Digits.append({
                    "idx": item['idx'],
                    "func_name": item['func_name'],
                    "SubToken": item['SubToken']
                    })
                num_Digits = num_Digits + 1

        # 2)Obscure abbreviation  Examples:dPrint,ficldInsn, setG ,isAquery
        # 筛选条件:寻找存在单个字母的SubToken，"SubToken": ["scalar", "X", "Map"]}
        for token in item['SubToken']:
            if re.fullmatch(r"[a-zA-Z]", token):
                Obscure.append({
                    "idx": item['idx'],
                    "func_name": item['func_name'],
                     "SubToken": item['SubToken']
                     })
                num_Obscure = num_Obscure + 1
        #
        # 5)Inconsistent naming Examples: attr, attribute
        # 筛选条件:创建一个SubToken的词汇表（无重复），然后匹配寻找存在互相子词的SubToken对，根据SubToken对中的词来匹配数据集。
    abbreviation_dict = load_abbreviation_dict_from_json('abbreviations.json')
        # for token in item['SubToken']:
    alltoken_lower = [token.lower() for token in alltoken]
    alltoken_lower_str = ' '.join(alltoken_lower)
    # print(alltoken_lower)
    Incon_tuple = []
        # 遍历 abbreviation_dict
    for full_form, abbreviations in abbreviation_dict.items():
             # 检查 test_text 中是否包含全称和任意一个缩写
        # if full_form.lower() in alltoken_lower and any(abbr.lower() in alltoken_lower for abbr in abbreviations):
        if re.search(r'\b' + re.escape(full_form.lower()) + r'\b', alltoken_lower_str) and \
                     any(re.search(r'\b' + re.escape(abbr.lower()) + r'\b', alltoken_lower_str) for abbr in abbreviations):
                # 如果同时存在，将全称和缩写都加入输出
            Incon.append(full_form)
            Incon.extend(abbr for abbr in abbreviations if abbr in alltoken_lower)


            for abbr in abbreviations:
                if abbr.lower() in alltoken_lower:
                    Incon_tuple.append((full_form, abbr))  # 将成对的全称和缩写保存为元组



    #读取错误数据集Inconsistent naming的信息
    all_false=[]
    Incon_idx = []
    for entry in result:
        subtoken_lower = [token.lower() for token in entry.get("SubToken", [])]
        # if any(keyword in token for token in subtoken_lower for keyword in Incon):
        for token in subtoken_lower:
            for keyword in Incon:
                if keyword == token:
                    Incon_idx.append(entry)
            break  # 一旦找到匹配项就跳出内层循环


    all_false = Single + Broad_mcaning+ Obscure + Digits + Incon_idx




    all_false = sorted(all_false, key=lambda x: x["idx"])
    deduplicated_data = []
    data_same = []
    seen_indices = set()
    for item in all_false:
        if item["idx"] not in seen_indices:
            deduplicated_data.append(item)
            seen_indices.add(item["idx"])
        else:
            data_same.append(item)

    output_dir = os.path.dirname(out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):  # 如果路径有目录部分且不存在
        os.makedirs(out_dir)


    file_data_mapping = {
        "Single_noun_or_preposition.jsonl": Single,
        "Broad_meaning.jsonl": Broad_mcaning,
        "Obscure_abbreviation.jsonl": Obscure,
        "Digits_in_word.jsonl": Digits,
        "Incon_idx.jsonl": Incon_idx,
        "all_false.jsonl": all_false,
        "deduplicated_data.jsonl":deduplicated_data,
        "data_same.jsonl": data_same,
        "result.jsonl":result
    }
    for file_name, result_data in file_data_mapping.items():
        with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as f:
            for entry in result_data:
                json.dump(entry, f)
                f.write('\n')

    with open(os.path.join(output_dir, "Inconsistent_naming.jsonl"), 'w', encoding='utf-8') as f:
        for full_form, abbr in Incon_tuple:
            # 将每个 (full_form, abbr) 元组转换为字典格式
            entry = {"full_form": full_form, "abbreviation": abbr}
            # 将字典写入文件，每行一个 JSON 对象
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "num_all_false.jsonl"), 'w', encoding='utf-8') as f:
        # 格式化内容，包含每个类型的数量
        content = (
            f"result: {len(result)}\n"
            f"num_Single noun or preposition: {num_Single}\n"
            f"num_Broad meaning: {num_Broad_mcaning}\n"
            f"num_Obscure abbreviation: {num_Obscure}\n"
            f"num_Digits in word: {num_Digits}\n"
            f"num_Inconsistent naming: {len(Incon_idx)}\n"
            f"all_false: {len(all_false)}\n"
            f"deduplicated data: {len(deduplicated_data)}\n"
            f"same data: {len(data_same)}\n"
        )
        f.write(content)

