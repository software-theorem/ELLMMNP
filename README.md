This repository provides an evaluation pipeline for method name prediction using large language models (LLMs), such as GPT-3.5, CodeLlama, and StarChat. Results are benchmarked against human evaluations. Supported programming languages include Java and Python.

**requirements.txt**

```
torch==2.1.0
transformers==4.46.3
sentence-transformers==3.2.1
tqdm==4.67.1
huggingface-hub==0.33.0
openai==1.88.0
torchaudio==2.1.0
torchvision==0.16.0
```

```
**LLM_eval_code/**
├── data/                       # Original datasets (Java and Python)
│   ├── *.txt                   # Plain text format
│   └── *.jsonl                 # JSONL format
├── result/                     # Model prediction results
│   ├── java/                   # Evaluation results for Java
│   └── python/                 # Evaluation results for Python
├── java.xlsx                   # Extracted model scores (Excel)
├── cal_tau_rho.py             # Script for computing Kendall’s τ and Spearman’s ρ
├── llm-eval.py                # Main script for evaluating method names using LLMs
├── txt_to_jsonl.py            # Converter: .txt  to .jsonl format
```

`{model}.jsonl`: Contains raw model predictions for automated processing

`{model}-read.txt`: Human-readable formatted version of model predictions

```
├── CSN-Premium4MNP/
│   ├── code/                  # Scripts for dataset extraction and preprocessing
│   └── readme.md              # Dataset statistics and description
**Dataset Quality/**
├── cal/                       # Evaluation metric implementations
│   ├── java               # F1 score and SBCS computation for java
│   └── python                   # F1 score and SBCS computation for python
├── code/
│   ├── LLM/              # StarCoder、StarChat-β、CodeLlama-I、GPT-3.5
│   └── PLM/                   # CodeBERT、CodeT5
```

SentenceBERT + Cosine Similarity (SBCS) uses the following model: [**all-mpnet-base-v2**](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

Files with `PLM` in the name are for pre-trained models such as CodeBERT and CodeT5

Files with `LLM` in the name are for large language models such as StarCoder, StarChat-β, CodeLlama-I, and GPT-3.5

Reference implementation for CodeT5: [CodeT5](https://github.com/salesforce/CodeT5/tree/main/CodeT5)

Reference implementation for CodeBERT: [CodeBERT](https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/code2nl)

```
Usage Example
# Run LLM evaluation for method name prediction
python llm-eval.py --data_file XXX/XXX/test.jsonl --model gpt-3.5
```

**Results & Data**

The results of the **Dataset Quality** evaluation and  **CSN-Premium4MN dataset** have been uploaded to:

[📂 CSN-Premium4MNP](https://drive.google.com/drive/folders/1HEt58MW8tvJrwLvgDQ4Cx6BW9kQuHWj2)
[📂 Dataset Quality evaluation](https://drive.google.com/drive/folders/12uZTmvSVSQ19dRysaANY32bgTJ3yt6OA)

