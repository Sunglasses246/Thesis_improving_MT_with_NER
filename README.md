# Thesis improving MT with NER
This is the github page of the thesis written by Simon Bosch providing the files used for the experiment: improving machine translation with named-entity awareness.

- LLM_Code contains all the code used for creating the outputs for each model.
- BLEU_CH_JP_KR contains all the .ja, .ko and .zh files used for calculating the BLEU scores of the English-Chinese, English-Korean and Engish-Japanese language pairs.
- Baseline, COT, Few-Shot and Zero-Shot contain all the m-ETA prediction files used to calculate the score.
- Baseline_translations contains the translated outputs for each model using the baseline prompt.
- Euro_translations, Gemma_translations, Tower_translations, Qwen_translations and Llama_translations contain all the prompt engineered outputs of the models.
- Experiment_test.ipynb is the jupyter notebook in which all the metric values were calculated and tables and figures made.
- src contains the code used for m-ETA calculation.
- data/xct/references/all contains all the source texts which the models had to translate.
