# **Text Parsing Methods using NLP**

## **About The Project**

The main goal of this project is the development of a Deep Learning model for Named Entity Recognition (NER) in Slovak. The [**Gerulata/SlovakBERT**](https://huggingface.co/gerulata/slovakbert) based model is fine-tuned on webscraped Slovak news articles. The finished model supports the following IOB tagged entity categories: *Person*, *Organisation*, *Location*, *Date*, *Time*, *Money* and *Percentage*. 


### **Related Work**
[![Thesis][Thesis]][Thesis-url]

[![Publication][Publication]][Publication-url]

[![HuggingFaceModel][HuggingFaceModel]][HuggingFaceModel-url]


### **Built With**
[![Python 3.10][Python]][Python-url]
[![NumPy][Numpy]][Numpy-url]
[![Pandas][Pandas]][Pandas-url]
[![Seaborn][Seaborn]][Seaborn-url]
[![Plotly][Plotly]][Plotly-url]
[![Datasets][Datasets]][Datasets-url]
[![Transformers][Transformers]][Transformers-url]
[![Ray][Ray]][Ray-url]
[![WandB][WandB]][WandB-url]
[![Scikit][Scikit]][Scikit-url]
[![PyTorch][Pytorch]][Pytorch-url]


### **Best Model Training Parameters**

|        **Parameter**        | **Value** |
|:---------------------------:|:---------:|
| per_device_train_batch_size |     4     |
|  per_device_eval_batch_size |     4     |
|        learning_rate        |   5e-05   |
|          adam_beta1         |    0.9    |
|          adam_beta1         |   0.999   |
|         adam_epsilon        |   1e-08   |
|       num_train_epochs      |     15    |
|      lr_scheduler_type      |   linear  |
|             seed            |     42    |

### **Best Model Training History**
Best model results are reached in the 8th training epoch.

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| 0.6721        | 1.0   | 70   | 0.2214          | 0.6972    | 0.7308 | 0.7136 | 0.9324   |
| 0.1849        | 2.0   | 140  | 0.1697          | 0.8056    | 0.8365 | 0.8208 | 0.952    |
| 0.0968        | 3.0   | 210  | 0.1213          | 0.882     | 0.8622 | 0.872  | 0.9728   |
| 0.0468        | 4.0   | 280  | 0.1107          | 0.8372    | 0.907  | 0.8708 | 0.9684   |
| 0.0415        | 5.0   | 350  | 0.1644          | 0.8059    | 0.8782 | 0.8405 | 0.9615   |
| 0.0233        | 6.0   | 420  | 0.1255          | 0.8576    | 0.8878 | 0.8724 | 0.9716   |
| 0.0198        | 7.0   | 490  | 0.1383          | 0.8545    | 0.8846 | 0.8693 | 0.9703   |
| 0.0133        | 8.0   | 560  | 0.1241          | 0.884     | 0.9038 | 0.8938 | 0.9735   |

### **Best Model Results**

Dataset distribution for final evaluation:
|    **NER Tag**    | **Number of Tokens** |
|:-----------------:|:--------------------:|
|       **0**       |         6568         |
|    **B-Person**   |          96          |
|    **I-Person**   |          83          |
| **B-Organizaton** |          583         |
| **I-Organizaton** |          585         |
|   **B-Location**  |          59          |
|   **I-Location**  |          15          |
|     **B-Date**    |          113         |
|     **I-Date**    |          87          |
|      **Time**     |           5          |
|    **B-Money**    |          44          |
|    **I-Money**    |          74          |
|  **B-Percentage** |          57          |
|  **I-Percentage** |          54          |

Confusion Matrix of the final evaluation:
![image](https://github.com/Raychani1/Text_Parsing_Methods_Using_NLP/assets/45550552/e6d1a1c6-e02f-4de9-9684-5882a405d31f)

Evaluation metrics of the final evaluation:
| **Precision** | **Macro-Precision** | **Recall** | **Macro-Recall** | **F1** | **Macro-F1** | **Accuracy** |
|:-------------:|:-------------------:|:----------:|:----------------:|:------:|:------------:|:------------:|
|     0.9897    |        0.9715       |   0.9897   |      0.9433      | 0.9895 |    0.9547    |    0.9897    |

### **Model Prediction Output Example**

![prediction_output](https://github.com/Raychani1/Text_Parsing_Methods_Using_NLP/assets/45550552/723ab7f1-4efb-4d03-87d6-b9ac1e40990f)


## **Getting Started**
To get a local copy up and running follow these simple steps.

### **Prerequisites**
* **Python 3.10.x** - It is either installed on your Linux distribution or on other Operating Systems you can get it from the [Official Website](https://www.python.org/downloads/release/python-3100/), [Microsoft Store](https://apps.microsoft.com/store/detail/python-310/9PJPW5LDXLZ5?hl=en-us&gl=US) or through `Windows Subsystem for Linux (WSL)` using this [article](https://medium.com/@rhdzmota/python-development-on-the-windows-subsystem-for-linux-wsl-17a0fa1839d).

## **Setup and Usage**

1. Clone the repo and navigate to the Project folder
   ```sh
   git clone https://github.com/Raychani1/Text_Parsing_Methods_Using_NLP
   ```

2. Create a new Python Virtual Environment
   ```sh
   python -m venv venv
   ```

3. Activate the Virtual Environment

    On Linux:
    ```sh
    source ./venv/bin/activate
    ```

    On Windows:
    ```sh
    .\venv\Scripts\Activate.ps1
    ```

4. Install Project dependencies

    ```sh
    pip install -r requirements.txt
    ```

5. Update *Weights & Biases* configuration (Optional)
    ```python
    WAND_ENV_VARIABLES = {
        'WANDB_API_KEY': 'YOUR-WANDB-API-KEY',
        'WANDB_PROJECT': 'YOUR-WANDB-PROJECT',
        'WANDB_LOG_MODEL': 'true',
        'WANDB_WATCH': 'false'
    }
    ```

6. Run main script (with prepared use-cases)
    ```sh
    python main.py
    ```

## **License**

Distributed under the **MIT License**. See [LICENSE](https://github.com/Raychani1/Text_Parsing_Methods_Using_NLP/blob/main/LICENSE) for more information.

## **Acknowledgments**
[Gerulata / SlovakBERT (Hugging Face Model)](https://huggingface.co/gerulata/slovakbert)

[Crabz / SlovakBERT-NER (Hugging Face Model)](https://huggingface.co/crabz/slovakbert-ner)

[Rohan Paul / YT_Fine_tuning_BERT_NER_v1 (Tutorial)](https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/blob/master/NLP/YT_Fine_tuning_BERT_NER_v1.ipynb)





<!-- Variables -->

[Thesis]: https://img.shields.io/badge/%F0%9F%93%9C-Masters%20Thesis-blue?style=for-the-badge
[Thesis-url]: #

[Publication]: https://img.shields.io/badge/%F0%9F%93%84-Publication-green?style=for-the-badge
[Publication-url]: #

[HuggingFaceModel]: https://custom-icon-badges.demolab.com/badge/Hugging%20Face%20Model-orange.svg?logo=huggingface&style=for-the-badge&labelColor=555
[HuggingFaceModel-url]: #

[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/

[Numpy]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://numpy.org/

[Pandas]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/

[Seaborn]: https://custom-icon-badges.demolab.com/badge/Seaborn-darkblue.svg?logo=seaborn&style=for-the-badge
[Seaborn-url]: https://seaborn.pydata.org/

[Plotly]: https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white
[Plotly-url]: https://plotly.com/

[Datasets]: https://custom-icon-badges.demolab.com/badge/Datasets-orange.svg?logo=huggingface&style=for-the-badge
[Datasets-url]: https://huggingface.co/datasets


[Transformers]: https://custom-icon-badges.demolab.com/badge/Transformers-orange.svg?logo=huggingface&style=for-the-badge
[Transformers-url]: https://huggingface.co/docs/transformers/index


[Ray]: https://img.shields.io/badge/Ray-028CF0?style=for-the-badge&logo=ray&logoColor=white
[Ray-url]: https://www.ray.io/


[WandB]: https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white
[WandB-url]: https://wandb.ai/site

<!-- [Scikit]: https://custom-icon-badges.demolab.com/badge/Scikit%20learn-blue.svg?logo=scikit-learn&style=for-the-badge -->

[Scikit]: https://img.shields.io/badge/scikit%20learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[Scikit-url]: https://scikit-learn.org/stable/index.html


[Pytorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white
[Pytorch-url]: https://pytorch.org/