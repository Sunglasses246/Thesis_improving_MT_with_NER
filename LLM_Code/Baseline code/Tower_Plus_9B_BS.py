from vllm import LLM, SamplingParams
import pandas as pd
import json
import os
from tqdm import tqdm

# Making the dataframe useable/modifyable
XCT_DE = pd.read_json("/home/sbosch/Experiment/ml-kg-mt/data/xct/references/all/de_DE.jsonl", lines = True)
# Copying so the original dataframe won't be changed
XCT_DE = XCT_DE.copy()
# Creating a dataset which only contains the sentences to be translated
XCT_IN = XCT_DE['source']

llm = LLM(model="Unbabel/Tower-Plus-9B", tensor_parallel_size=1)
sampling_params = SamplingParams(
    best_of=1,
    temperature=0,
    max_tokens=400,
  )

batch_size = 512


outputs_df_list = []
for i in tqdm(range(0, len(XCT_IN), batch_size)):
    batch = XCT_IN[i:i+batch_size]

    messages = [
    [{
        "role":"system",
        "content":"You are a translation engine. You output the translation from the source text to the target language only. No explanation. No rewriting. No correction. Literal translation strictly. No empty lines. You don't output any other language than the target language."
    },
    {
        "role":"user",
        "content":f"Translate the English source text to the target language Chinese (China):\nEnglish: {x}\nChinese (China):"
    }]
    for x in batch
    ]


    results = llm.chat(messages, sampling_params)

    for out in results:
        outputs_df_list.append(out.outputs[0].text.strip())

df_translation = pd.DataFrame(data = outputs_df_list)
df_translation.to_csv("translation_BS_Tower_DE.csv", index=False)
print(df_translation)