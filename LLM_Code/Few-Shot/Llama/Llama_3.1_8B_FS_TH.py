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
XCT_DE = pd.read_json("/home/sbosch/Experiment/ml-kg-mt/data/xct/references/all/th_TH.jsonl", lines = True)
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

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=1)
sampling_params = SamplingParams(
    best_of=1,
    temperature=0,
    max_tokens=30,
  )

batch_size = 512

outputs_df_list = []
for i in tqdm(range(0, len(XCT_IN), batch_size)):
    batch = XCT_IN[i:i+batch_size]

    messages = [
    [{
  "role": "system",
  "content": f"""You are a translation engine.
Translate from English to Thai (Thailand).
Output only the translation.
Preserve named entities accurately in the translation.
Do not add, omit, or explain anything.
Give the output in one sentence.
"""
}
,
    {
        "role":"user",
        "content": f"""English: Maria was talking about The Guy.
Thai (Thailand): มาเรียกำลังพูดถึง “The Guy”

English: When did Galileo cross the Alps?
Thai (Thailand): กาลิเลโอข้ามเทือกเขาแอลป์เมื่อใด

English: How did the Rebels from Star Wars destroy the Death Star?
Thai (Thailand): กบฏจากสตาร์ วอร์ส ทำลายเดธสตาร์ได้อย่างไร

English: {x}
Thai (Thailand):"""

    }]
    for x in batch
    ]


    results = llm.chat(messages, sampling_params)

    for out in results:
        outputs_df_list.append(out.outputs[0].text.strip())

# Make sure your prompt_token_ids look like this
# print(outputs[0].outputs[0].text)
# > Olá, mundo!

df_translation = pd.DataFrame(data = outputs_df_list)
df_translation.to_csv("translation_FS_LLAMA_TH.csv", index=False)
print(df_translation)