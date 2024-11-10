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

| Java                               |       | Test     | Train     | Valid     | Sum     |
| ---------------------------------- | ----- | -------- | --------- | --------- | ------- |
| total samples                      | 10955 | 164923   | 5183      | 181061    | 181061  |
| num of Single  noun or preposition | 1937  | 25234    | 865       | 28036     | 28036   |
| num of Broad  meaning              | 259   | 5553     | 115       | 5927      | 5927    |
| num of Obscure  abbreviation       | 84    | 1412     | 52        | 1548      | 1548    |
| num of Digits in  word             | 203   | 1985     | 65        | 2253      | 2253    |
| num of Inconsistent  naming        | 1341  | 35729    | 449       | 37519     | 37519   |
| all error sample                   | 3824  | 69913    | 1546      | 75283     | 75283   |
| deduplicated  error samples        | 3284  | 58042    | 1326      | 62652     | 62652   |
| overlapping error samples          | 540   | 11871    | 220       | 12631     | 12631   |
| appropriate  method names          | 7671  | 106881   | 3857      | 118409    | 118409  |
| **Python**                         | ****  | **Test** | **Train** | **Valid** | **Sum** |
| total samples                      | 14918 | 251820   | 13914     | 280652    | 280652  |
| num of Single  noun or preposition | 3414  | 58531    | 3744      | 65689     | 65689   |
| num of Broad  meaning              | 381   | 7127     | 378       | 7886      | 7886    |
| num of Obscure  abbreviation       | 190   | 1731     | 167       | 2088      | 2088    |
| num of Digits in  word             | 205   | 4954     | 312       | 5471      | 5471    |
| num of Inconsistent  naming        | 2408  | 61051    | 2049      | 65508     | 65508   |
| all error sample                   | 6598  | 133394   | 6650      | 146642    | 146642  |
| deduplicated  error samples        | 5696  | 112602   | 5723      | 124021    | 124021  |
| overlapping error samples          | 902   | 20792    | 927       | 22621     | 22621   |
| appropriate  method names          | 9222  | 139218   | 8191      | 156631    | 156631  |

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



