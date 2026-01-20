# pip install vllm
# Gemma by default only uses 4k context. You need to set the following variables:
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
from vllm import LLM, SamplingParams
import pandas as pd
import json
import os
from tqdm import tqdm

# pip install vllm
# Gemma by default only uses 4k context. You need to set the following variables:
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# Iterate over files in directory
# Making the dataframe useable/modifyable
XCT_DE = pd.read_json("/home/sbosch/Experiment/ml-kg-mt/data/xct/references/all/zh_TW.jsonl", lines = True)
# Copying so the original dataframe won't be changed
XCT_DE = XCT_DE.copy()
# Creating a dataset which only contains the sentences to be translated
XCT_IN = XCT_DE['source']
# Creating a dataset which only containes the 'correctly' translated sentences
# XCT_TARGET = XCT_DE['targets']
# Mofidying the target dataset, because it is still written as a list instead of a dictionary which can be used for the dataframe
# XCT_TARGET = XCT_TARGET.apply(lambda x: pd.Series(x[0]))
# from vllm import LLM, SamplingParams
# import pandas as pd
# import json
# import os
outputs_df_list = []

llm = LLM(model="utter-project/EuroLLM-9B-Instruct", tensor_parallel_size=1)
sampling_params = SamplingParams(
    best_of=1,
    temperature=0,
    max_tokens=400,
  )

batch_size = 512


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
# for x in XCT_IN:
#   messages = [{"role": "user", "content": }]
#   outputs = llm.chat(messages, sampling_params)
#   outputs_df_list.append(pd.Series(outputs[0].outputs[0].text))
# Make sure your prompt_token_ids look like this
# print(outputs[0].outputs[0].text)
# > Olá, mundo!

df_translation = pd.DataFrame(data = outputs_df_list)
df_translation.to_csv("translation_PE_ZS_EURO_CH.csv", index=False)
print(df_translation)