import re
import json



with open(r'XXX\XXX\XXX\XXX\Java_examples.txt', 'r',
          encoding='utf-8') as file:
    content = file.read()



sample_pattern = r'(N0\.\d+=+)(.*?)((?=N0\.\d+=+)|$)'
samples = re.findall(sample_pattern, content, re.DOTALL)

data=[]
for idx, sample in enumerate(samples):
    content = sample[1]


    code_split = re.split(r'猜测的函数名1:', content, maxsplit=1)
    code = code_split[0].strip() if code_split else ""


    def extract_prediction(label, text):
        match = re.search(fr'{label}:\s*(\w+)', text)
        return match.group(1) if match else ""

    groundTruth = extract_prediction('猜测的函数名1', content)
    CodeBERT = extract_prediction('猜测的函数名2', content)
    CodeT5 = extract_prediction('猜测的函数名3', content)
    ChatGPT = extract_prediction('猜测的函数名4', content)


    data.append({
        "idx": idx,
        "code": code,
        "Function Name 1": groundTruth,
        "Function Name 2": CodeBERT,
        "Function Name 3": CodeT5,
        "Function Name 4": ChatGPT
    })

with open('XXX\XXX\XXX\XXX.jsonl', 'w', encoding='utf-8') as file:
     for item in data:
         json_line= json.dumps(item, ensure_ascii=False)
         file.write(json_line + '\n')


