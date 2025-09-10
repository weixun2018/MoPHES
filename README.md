# MPChat
Mobile psychological chat application based on on-device language model.
## Data Collection
The data for this study comes from two public psychological counseling resources: Chinese-Psychological-QA-Dataset and EmoLLM.  
- **Chinese-Psychological-QA-Dataset**: After preprocessing, 81,219 valid single-turn Q&A pairs were obtained, with an additional 21,626 discarded due to lack of responses.  
- **EmoLLM**: Provided 32,333 Q&A pairs.  

After merging and unifying into JSON format, a total of 113,552 samples were obtained. Each data entry includes a user question, corresponding answer, and a psychological disorder category label.  
It should be noted that all data is in Chinese, and there is an imbalance in the distribution among different mental health categories. The data is for research purposes only.

## Method
To ensure quality and consistency, several preprocessing steps were applied:  
- **Length Filtering**: Samples were removed if the user query had fewer than 50 characters or the assistant reply had fewer than 100 characters.  
- **AI-based Filtering**: An automated filter was used to detect and remove low-quality or irrelevant responses, eliminating approximately 41,083 samples.  
- **Duplicate Removal**: MinHash with Locality-Sensitive Hashing (LSH) was applied at a 70% similarity threshold, removing around 34,827 near-duplicate pairs.  

After these steps, the dataset became cleaner and more reliable, providing a solid foundation for subsequent analysis and multi-turn dialogue generation.  

### Base Model
The base model used in this study is **MiniCPM4-0.5B**. 

### Prompt
- **ai_rating_prompt.txt**: Used for AI to rate the quality of model outputs.  
- **classify_prompt.txt**: Used for classification tasks, categorizing input content into corresponding categories.  
- **filter_prompt.txt**: Used to filter out low-quality or irrelevant content.  
- **single_to_multi_prompt.txt**: Expands single-turn dialogue data into multi-turn dialogue data.  
- **user_state_prompt_cn.txt**: Dialogue prompts carrying user state in a Chinese environment.  
- **user_state_prompt.txt**: Dialogue prompts carrying user state in an English environment.

## Train
### SFT
We applied **Supervised Fine-Tuning (SFT)** on the base model **MiniCPM4-0.5B** using task-specific psychological counseling data.  
SFT aligns the model with desired behaviors by training it on high-quality input–output pairs, where the input is a user query and the output is an expert-annotated or reliable assistant response.  

In our setup:  
- **Training data** and **evaluation data** were drawn from preprocessed counseling QA pairs.  
- **LoRA (Low-Rank Adaptation)** was used to improve training efficiency while reducing memory and computation costs.  
- Key hyperparameters included a **learning rate of 1e-4**, **max sequence length of 1024**, and **1000 training steps** with periodic evaluation and checkpoint saving.  

This fine-tuning stage provides the foundation for adapting the base model to domain-specific dialogue generation tasks.  


### RL

## Evaluation
### Chat Model Evaluation
The detailed BLEU and ROUGE results are summarized in the table below. They show that commercial models like GPT-4.1 and Gemini generally achieve stronger scores, while domain-specific models such as EmoLLM, MeChat, and PsyCha


| Model             | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | Score |
|-------------------|--------|--------|--------|--------|---------|---------|---------|-------|
| **MoPHES**        |        |        |        |        |         |         |         |       |
| ├─ Label          | 38.89  | 23.89  | 16.39  | 11.99  | 41.32   | 17.45   | 35.18   | 27.09 |
| └─ Output         | 35.61  | 20.03  | 13.00  | 9.27   | 38.19   | 14.01   | 31.62   | 23.74 |
| **MiniCPM4-0.5B** |        |        |        |        |         |         |         |       |
| ├─ Label          | 18.69  | 7.72   | 3.67   | 1.96   | 24.36   | 4.92    | 18.15   | 11.92 |
| └─ Output         | 8.83   | 3.37   | 1.51   | 0.83   | 18.10   | 2.74    | 10.87   | 7.11  |
| **gpt-4.1**       |        |        |        |        |         |         |         |       |
| ├─ Label          | 42.07  | 22.32  | 11.22  | 5.67   | 40.04   | 12.61   | 28.78   | 23.76 |
| └─ Output         | 40.23  | 21.20  | 10.85  | 5.68   | 39.75   | 12.20   | 28.22   | 23.13 |
| **gemini-2.0-flash** |      |        |        |        |         |         |         |       |
| ├─ Label          | 35.99  | 16.63  | 7.91   | 4.06   | 36.36   | 9.20    | 25.31   | 19.91 |
| └─ Output         | 31.74  | 14.59  | 6.83   | 3.38   | 35.48   | 8.69    | 24.49   | 18.53 |
| **DeepSeek-7B**   |        |        |        |        |         |         |         |       |
| ├─ Label          | 8.17   | 2.68   | 1.08   | 0.55   | 16.88   | 1.87    | 9.86    | 6.34  |
| └─ Output         | 8.23   | 3.02   | 1.31   | 0.66   | 17.33   | 2.28    | 10.03   | 6.60  |
| **Qwen2-7B**      |        |        |        |        |         |         |         |       |
| ├─ Label          | 15.59  | 5.71   | 2.73   | 1.60   | 21.41   | 3.45    | 15.16   | 9.89  |
| └─ Output         | 9.86   | 3.34   | 1.47   | 0.83   | 16.74   | 2.19    | 10.60   | 6.87  |
| **ChatGLM4**      |        |        |        |        |         |         |         |       |
| ├─ Label          | 13.56  | 5.68   | 2.83   | 1.63   | 21.60   | 4.17    | 14.66   | 9.71  |
| └─ Output         | 8.29   | 2.96   | 1.31   | 0.70   | 16.80   | 2.26    | 9.83    | 6.48  |
| **EmoLLM**        |        |        |        |        |         |         |         |       |
| ├─ Label          | 32.79  | 16.58  | 9.03   | 5.58   | 32.65   | 9.97    | 25.89   | 19.44 |
| └─ Output         | 32.20  | 16.33  | 9.00   | 5.56   | 32.01   | 10.06   | 25.74   | 19.21 |
| **MeChat**        |        |        |        |        |         |         |         |       |
| ├─ Label          | 35.00  | 17.79  | 9.24   | 5.30   | 35.58   | 10.63   | 25.52   | 20.39 |
| └─ Output         | 30.25  | 14.65  | 7.25   | 3.98   | 32.99   | 8.92    | 23.54   | 17.95 |
| **PsyChat**       |        |        |        |        |         |         |         |       |
| ├─ Label          | 31.82  | 15.41  | 7.37   | 3.75   | 35.66   | 9.07    | 23.12   | 18.63 |
| └─ Output         | 27.66  | 12.55  | 5.50   | 2.66   | 33.10   | 7.78    | 21.46   | 16.46 |
| **SoulChat**      |        |        |        |        |         |         |         |       |
| ├─ Label          | 32.86  | 16.65  | 8.67   | 4.99   | 33.08   | 9.66    | 24.10   | 19.06 |
| └─ Output         | 28.42  | 13.77  | 6.89   | 3.88   | 29.69   | 8.07    | 22.73   | 16.72 |


As shown in the results, OpenAI and Gemini achieve the highest scores, while our fine-tuned MiniCPM4-0.5B-35k consistently surpasses its base model across most dimensions.

| Model                | Und. | Emp. | Prof. | Reg. | Saf. | Total |
|-----------------------|------|------|-------|------|------|-------|
| **DeepSeek-R1-7B**    |      |      |       |      |      |       |
| ├─ Label             | 1.061| 0.624| 0.376 | 0.659| 1.998| 4.718 |
| └─ Output            | 0.950| 0.575| 0.342 | 0.649| 1.992| 4.508 |
| **glm-4-9b-chat**     |      |      |       |      |      |       |
| ├─ Label             | 1.356| 1.040| 0.694 | 1.044| 1.998| 6.133 |
| └─ Output            | 1.262| 0.940| 0.495 | 0.948| 1.998| 5.643 |
| **Qwen2.5-7B**        |      |      |       |      |      |       |
| ├─ Label             | 1.239| 1.016| 0.681 | 0.961| 2.000| 5.897 |
| └─ Output            | 1.184| 0.907| 0.486 | 0.888| 2.000| 5.465 |
| **EmoLLM**            |      |      |       |      |      |       |
| ├─ Label             | 1.219| 1.002| 0.917 | 0.994| 1.975| 6.107 |
| └─ Output            | 1.185| 0.978| 0.874 | 0.953| 1.969| 5.959 |
| **MeChat**            |      |      |       |      |      |       |
| ├─ Label             | 1.255| 1.031| 0.813 | 0.994| 2.000| 6.093 |
| └─ Output            | 1.185| 1.002| 0.669 | 0.969| 2.000| 5.825 |
| **PsyChat**           |      |      |       |      |      |       |
| ├─ Label             | 1.512| 1.286| 1.396 | 1.231| 2.000| 7.425 |
| └─ Output            | 1.517| 1.278| 1.313 | 1.213| 2.000| 7.321 |
| **SoulChat**          |      |      |       |      |      |       |
| ├─ Label             | 1.166| 0.997| 0.670 | 0.986| 1.998| 5.817 |
| └─ Output            | 1.084| 0.946| 0.523 | 0.943| 2.000| 5.496 |
| **OpenAI**            |      |      |       |      |      |       |
| ├─ Label             | 1.856| 1.636| 1.670 | 1.523| 2.000| 8.685 |
| └─ Output            | 1.907| 1.731| 1.757 | 1.585| 2.000| 8.980 |
| **Gemini**            |      |      |       |      |      |       |
| ├─ Label             | 1.787| 1.519| 1.488 | 1.372| 2.000| 8.166 |
| └─ Output            | 1.784| 1.474| 1.303 | 1.286| 2.000| 7.847 |
| **MiniCPM4-0.5B**     |      |      |       |      |      |       |
| ├─ Label             | 1.385| 1.150| 1.107 | 0.969| 2.000| 6.611 |
| └─ Output            | 1.219| 0.985| 0.854 | 0.829| 2.000| 5.887 |
| **MiniCPM4-0.5B-35k** |      |      |       |      |      |       |
| ├─ Label             | 1.462| 1.210| 1.461 | 1.072| 2.000| 7.204 |
| └─ Output            | 1.449| 1.201| 1.433 | 1.069| 2.000| 7.152 |


### Judgment Model Evaluation
The results show that our fine-tuned MiniCPM models outperform the baseline on depression and anxiety detection. The 5k with-system model performs best with a normalized score of 0.898, suggesting that system prompts enhance stability and reliability in mental health classification.

| Model                       | Acc-D | Acc-A | Prec-D | Prec-A | Prec-All | Rec-D | Rec-A | Rec-All | F1-D | F1-A | F1-All | Norm-D | Norm-A | Norm-Total |
|-----------------------------|-------|-------|--------|--------|----------|-------|-------|---------|------|------|--------|--------|--------|------------|
| MiniCPM4-0.5B               | 0.055 | 0.050 | 0.091  | 0.835  | 0.016    | 0.062 | 0.057 | 0.034   | 0.025| 0.045| 0.010  | 0.380  | 0.310  | 0.345      |
| MiniCPM4-0.5B-5k            | 0.630 | 0.805 | 0.658  | 0.776  | 0.513    | 0.630 | 0.805 | 0.510   | 0.630| 0.781| 0.494  | 0.870  | 0.927  | 0.898      |
| DeepSeek-R1-Distill-Qwen-7B | 0.515 | 0.590 | 0.624  | 0.694  | 0.420    | 0.515 | 0.590 | 0.325   | 0.438| 0.610| 0.284  | 0.825  | 0.853  | 0.839      |
| glm-4-9b-chat               | 0.760 | 0.745 | 0.782  | 0.764  | 0.612    | 0.760 | 0.745 | 0.545   | 0.758| 0.750| 0.503  | 0.918  | 0.913  | 0.916      |
| Qwen2.5-7B                  | 0.515 | 0.330 | 0.598  | 0.617  | 0.292    | 0.515 | 0.330 | 0.200   | 0.518| 0.393| 0.200  | 0.802  | 0.712  | 0.757      |

- **ACC-D / ACC-A**: Classification accuracy (Accuracy), corresponding to **Diagnosis tasks** and **Assessment/Advice tasks** respectively.
- **Prec-D / Prec-A / Prec-All**: Precision, measuring how many of the predictions made by the model for a certain class are actually correct. D and A correspond to different task dimensions, and All is the overall precision.
- **Rec-D / Rec-A / Rec-All**: Recall, measuring how many of the samples that truly belong to a certain class are successfully identified by the model.
- **F1-D / F1-A / F1-All**: F1 score, the harmonic mean of Precision and Recall, comprehensively reflecting the model's performance across different dimensions.
- **Norm-D / Norm-A / Norm-Total**: Normalized Score, mapping the metrics of different dimensions to a unified range for easier comparison across dimensions and models.

## Reference
- [PsyQA](https://github.com/thu-coai/PsyQA)
- [SmileChat](https://github.com/qiuhuachuan/smile)
- [SoulChat](https://github.com/scutcyr/SoulChat)
- [EmoLLM](https://github.com/SmartFlowAI/EmoLLM)
- [PsycoLLM](https://github.com/MACLAB-HFUT/PsycoLLM)
## Citation