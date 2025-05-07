

# Model Description


 Difficult problems, which often result in long reasoning traces, are widely recognized as key factors for enhancing the performance of reasoning models. However, such high-challenge problems are scarce, limiting the size of available datasets. In this paper, we propose a simple method to decouple the reliance on problem difficulty. First, we empirically demonstrate that reasoning length, rather than problem difficulty, primarily influences the performance of trained models. Second, we identify a scaling law on reasoning length, showing that model performance increases in a log-linear fashion as the reasoning data length grows. Finally, we introduce a straightforward technique to generate reasoning data of arbitrary length, and show that synthesized data is effective for training reasoning models. After fine-tuning the Qwen2.5-32B-Instruct language model on our Long1 dataset, we present our model, Long1-32B, which achieves remarkable performance with only 1,000 training samples, achieving 95.6% accuracy on MATH, and 71.1% on GPQA outperforming DeepSeek-R1-Distill-Qwen-32B.
1. Challenging a common assumption: We question the prevalent belief that problem difficulty is the most critical factor. Instead, our experiments suggest that reasoning length is key to training high-performance reasoning models. This insight allows us to build large-scale, long-reasoning datasets without being constrained by the rarity of extremely difficult problems.
2. Identifying a scaling law on reasoning length: We observe that model performance improves nearly linearly as the length of training data increases exponentially. This phenomenon highlights the efficiency gains achievable by focusing on the length of reasoning sequences.
3. Proposing a simple synthesis method: We introduce a technique to generate arbitrarily long reasoning data. Using this method, we release the Long1K dataset, upon which our Long1K-32B model is fine-tuned. This model surpasses existing baselines on benchmarks such as MATH500 and GPQA Diamond, demonstrating that extended reasoning sequences can significantly enhance model performance.



# Detail

  Among the work of the thesis, we firstly did two sets of experiments, namely, conceptual synthetic long problems with conceptual synthetic difficult problems, and synthetic long problems with original difficult problems. The related results are shown in the following figure. It turns out that the models perform similarly in mathematical reasoning when the training token lengths are similar. We make a conclusion that the key factor affecting the model's reasoning effectiveness is not the difficulty.

![img_3.jpg](img_2.jpg)


  Therefore, we shifted our focus from the difficulty of mathematical problems to the length of mathematical problems. We made the assumption that length is the key factor in constructing inference models. To this end, we explored the effect of different tokens lengths on the reasoning ability of the model at the same difficulty level. Firstly, we classify the token length into 4 levels, whose lengths are 1.5k,3k,6k,12k. Then, we set the number of questions to 500, and conduct experimental validation on Qwen2.5-32B model. The results are shown below. The data show that on the math500 dataset, the performance is close to linearly increasing as the length increases.

![img_3.jpg](img_1.jpg)

  In addition, we compared the reasoning processes of two models trained with reasoning lengths of 1.5k and 12k, respectively, on the MATH500 test set, including both successful and failed reasoning attempts. Our analysis included statistical comparisons of the average reasoning token length and the top 10 most frequently used words during reasoning. The goal was to understand why the model trained with a reasoning length of 12k achieved an accuracy improvement of over 5%.

![img_33.jpg](img_33.jpg)



# Training Data
  We conducted relevant experiments using our own synthesized [LONG1k](https://huggingface.co/datasets/ZTss/LONG1k) dataset. LONG1k is a composite data generated for model training from two datasets, Openthouhts114k and s1.1. Specifically, on one hand, we randomly select two mathematical problems from Openthouhts114k. The problems, reasoning processes, and results of these two mathematical problems are concatenated together using different linking words to increase the length of the prompts. On the other hand, in order to avoid overfitting of the model to two mathematical problems and improve its robustness, we also extracted a certain number of mathematical problems that meet the length requirements from the s1.1 dataset and fused them into LONG1k. Ultimately, the synthetic data LONG1k used for model training will consist of these two parts of data. Of course, in different experiments, the ratio of the length of the two parts of the problem and the number of markers will be dynamically adjusted according to the experimental requirements.


# Evaluation

![img_3.jpg](img_3.jpg)

Performance comparison of different models across multiple reasoning benchmarks (pass@1). The best results for each benchmark are highlighted in bold, with the second-best underlined. The data for s1 does not use budget forcing, and the data for s1.1 that does not use budget forcing comes from Open Thoughts.


# Uses
The LONG1-32B model file has been uploaded here. If you want to use it, please download it [here]. We have uploaded our reasoning and evaluation scripts. If you are interested in using it, please follow the steps below.

  ## Requirement
```
  pip install -r requirements.txt
```
  ## Reasoning
  After downloading the model, please use the following code to perform result inference.
```
  bash predict.sh
```


  ## Evaluation
  Use the following code to calculate indicators.
```
  python calc_metric_lc.py
```

