**abbreviations.json**
This file is a dictionary mapping words to their corresponding abbreviations, used to check for consistency in method naming.

**classifications.py**
This file identifies five types of low-quality method names in the target file, including:

- Single noun or preposition
- Obscure abbreviation
- Broad meaning
- Digits in word
- Inconsistent naming

**filtration.py**
This file filters out low-quality method names from the original data, ensuring that the output contains only high-quality naming entries.



We release the dataset in this [website](https://drive.google.com/drive/folders/1HEt58MW8tvJrwLvgDQ4Cx6BW9kQuHWj2?usp=drive_link).



The following is a table of statistics on the treatment of low-quality method names：

| Java                               | Train     | Valid     | Test     | Sum     |
| ---------------------------------- | --------- | --------- | -------- | ------- |
| total samples                      | 164923    | 5183      | 10955    | 181061  |
| num of Single  noun or preposition | 25234     | 865       | 1937     | 28036   |
| num of Broad  meaning              | 5553      | 115       | 259      | 5927    |
| num of Obscure  abbreviation       | 1412      | 52        | 84       | 1548    |
| num of Digits in  word             | 1985      | 65        | 203      | 2253    |
| num of Inconsistent  naming        | 35729     | 449       | 1341     | 37519   |
| all error sample                   | 69913     | 1546      | 3824     | 75283   |
| deduplicated  error samples        | 58042     | 1326      | 3284     | 62652   |
| overlapping error samples          | 11871     | 220       | 540      | 12631   |
| appropriate  method names          | 106881    | 3857      | 7671     | 118409  |
| **Python**                         | **Train** | **Valid** | **Test** | **Sum** |
| total samples                      | 251820    | 13914     | 14918    | 280652  |
| num of Single  noun or preposition | 58531     | 3744      | 3414     | 65689   |
| num of Broad  meaning              | 7127      | 378       | 381      | 7886    |
| num of Obscure  abbreviation       | 1731      | 167       | 190      | 2088    |
| num of Digits in  word             | 4954      | 312       | 205      | 5471    |
| num of Inconsistent  naming        | 61051     | 2049      | 2408     | 65508   |
| all error sample                   | 133394    | 6650      | 6598     | 146642  |
| deduplicated  error samples        | 112602    | 5723      | 5696     | 124021  |
| overlapping error samples          | 20792     | 927       | 902      | 22621   |
| appropriate  method names          | 139218    | 8191      | 9222     | 156631  |

**Total Samples**: The total number of samples.

**num of Single  noun or preposition**: Number of method names that are a single noun or preposition.

**num of Broad  meaning**: Number of method names with broad or overly general meanings.

**num of Obscure  abbreviation**: Number of method names that use unclear or obscure abbreviations.

**num of Digits in  word**: Number of method names containing digits within words.

**num of Inconsistent  naming**: Number of method names with inconsistent naming conventions.

**all error sample**: Total number of samples across the five types of naming errors.

**deduplicated  error samples**: Total number of unique samples after removing duplicate error samples.

**overlapping error samples**: Number of samples that contain overlapping naming errors from multiple categories.

**appropriate  method names**: Total number of high-quality method names remaining after filtering out all five types of naming errors from the original data.



