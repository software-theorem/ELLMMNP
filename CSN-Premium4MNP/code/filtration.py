#
# import json
# import os
#
# oridata = "F:/fuwuqi/A_Comprehensive_Evaluation_LLM_MNP/chuli/oridata/java/test_csn.jsonl"
# deduplicated_data = "F:/fuwuqi/A_Comprehensive_Evaluation_LLM_MNP/chuli/extradata/java/test/deduplicated_data.jsonl"
# out_path = "F:/fuwuqi/A_Comprehensive_Evaluation_LLM_MNP/chuli/filtered/java/test_filtered.jsonl"
#
# # Initialize an empty list to store data from A.jsonl
# a_data = []
# b_data = []
#
# # Load data from A.jsonl
# with open(oridata, 'r') as file_a:
#     for idx, line in enumerate(file_a):
#         # if idx == 100:
#         #     break
#         line = line.strip()
#         line = json.loads(line)
#         a_data.append({"idx": idx, "line": line})
#
# with open(deduplicated_data, 'r') as file_b:
#     for line in file_b:
#         line = line.strip()
#         line_data = json.loads(line)
#         b_data.append(line_data)
#
#
# # Extract idx values from B.jonsl
# b_idx_values = {entry["idx"] for entry in b_data}
# a_idx_values = {entry["idx"] for entry in a_data}
#
# print("a_idx_values",a_idx_values)
# print("b_idx_values",b_idx_values)
#
# # Filter A.jonsl data to exclude entries with idx values present in B.jonsl
# filtered_a_data = [entry for entry in a_data if entry["idx"] not in b_idx_values]
# # Write the filtered data back to A_filtered.jonsl
# with open(out_path, 'w') as file_filtered_a:
#     for entry in filtered_a_data:
#         file_filtered_a.write(json.dumps(entry["line"]) + '\n')
#
#     with open(os.path.join(out_path, "num_idx.jsonl"), 'w', encoding='utf-8') as f:
#         # 格式化内容，包含每个类型的数量
#         content = (
#             f"idx in oridata: {a_idx_values}\n"
#             f"idx in deduplicated_data: {a_idx_values}\n"
#         )
#         f.write(content)
import json
import os

oridata = "F:/fuwuqi/A_Comprehensive_Evaluation_LLM_MNP/chuli/oridata/python/train.jsonl"
deduplicated_data = "F:/fuwuqi/A_Comprehensive_Evaluation_LLM_MNP/chuli/extradata/python/train/deduplicated_data.jsonl"
out_path = "F:/fuwuqi/A_Comprehensive_Evaluation_LLM_MNP/chuli/filtered/python/train/"
filtered_file = os.path.join(out_path, "train_filtered.jsonl")
num_idx_file = os.path.join(out_path, "num_idx.jsonl")

# 初始化空列表来存储 A.jsonl 和 B.jsonl 数据
a_data = []
b_data = []

# 读取 A.jsonl 文件数据并统计行数
with open(oridata, 'r',encoding='utf-8') as file_a:
    oridata_line_count = 0
    for idx, line in enumerate(file_a):
        oridata_line_count += 1
        line = line.strip()
        line = json.loads(line)
        a_data.append({"idx": idx, "line": line})

# 读取 B.jsonl 文件数据并统计行数
with open(deduplicated_data, 'r',encoding='utf-8') as file_b:
    deduplicated_data_line_count = 0
    for line in file_b:
        deduplicated_data_line_count += 1
        line = line.strip()
        line_data = json.loads(line)
        b_data.append(line_data)

# 提取 B.jsonl 文件中的 idx 值
b_idx_values = {entry["idx"] for entry in b_data}
a_idx_values = {entry["idx"] for entry in a_data}

print("a_idx_values", a_idx_values)
print("b_idx_values", b_idx_values)

# 过滤 A.jsonl 数据，排除在 B.jsonl 中出现的 idx 值
filtered_a_data = [entry for entry in a_data if entry["idx"] not in b_idx_values]

# 将过滤后的数据写入 test_filtered.jsonl 文件并统计行数
filtered_file_line_count = 0
with open(filtered_file, 'w',encoding='utf-8') as file_filtered_a:
    for entry in filtered_a_data:
        filtered_file_line_count += 1
        file_filtered_a.write(json.dumps(entry["line"]) + '\n')

# 将 idx 信息和行数统计结果写入 num_idx.jsonl 文件
with open(num_idx_file, 'w', encoding='utf-8') as f:
    content = (
        f"idx in oridata: {a_idx_values}\n"
        f"idx in deduplicated_data: {b_idx_values}\n"
        f"Total lines in oridata: {oridata_line_count}\n"
        f"Total lines in deduplicated_data: {deduplicated_data_line_count}\n"
        f"Total lines in filtered_file: {filtered_file_line_count}\n"
    )
    f.write(content)
