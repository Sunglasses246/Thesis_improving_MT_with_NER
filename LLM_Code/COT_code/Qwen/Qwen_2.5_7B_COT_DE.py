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
XCT_DE = pd.read_json("/home/sbosch/Experiment/ml-kg-mt/data/xct/references/all/de_DE.jsonl", lines = True)
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

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=1)
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
  "role": "system",
  "content": f"""You are a translation engine.
Translate from English to German (Germany).
Output only the translation.
Preserve named entities accurately.
Do not add, omit, or explain anything.
"""
}
,
    {
        "role":"user",
        "content": f"""Analysis of the translation sentence:
### Step 1: Tokenizing the sentence
Split each sentence into each individual words.

### Step 2: Identify which tokens are potential named entities.

### Step 3: Look at the context the potential named entity is used in.

### Step 4: Translate the potential named entity according to the context.
        
### Example 1:
English: Maria was talking about The Guy.

Thoughts: 
- Step 1: [Maria, was, talking, about, the, Guy]
- Step 2: [Maria, Guy] are potential named entities.
- Step 3: Looking at the context Maria refers to a person and the Guy refers to a title.
- Step 4: Maria will be translated as Maria and the Guy as the Guy.
German (Germany): Maria hat über „The Guy“ gesprochen.

### Example 2:
English: When did Galileo cross the Alps?

Thoughts: 
- Step 1: [When, did, Galileo, cross, the, Alps, ?]
- Step 2: [Galileo, Alps] are possible named entities.
- Step 3: Looking at the context Galileo refers to a person and Alps refers to a nature reservate.
- Step 4: Galileo will be Galileo and Alps will be Alpen.
German (Germany): Wann überquerte Galileo die Alpen?

### Example 3: 
English: How did the Rebels from Star Wars destroy the Death Star?

Thoughts:
- Step 1: [How, did, the, Rebels, from, Star, Wars, destroy, the, Death, Star, ?]
- Step 2: [Rebels, Star, Wars, Death, Star] are potential named entities.
- Step 3: Looking at the context Rebels refers to the Rebel Alliance in Star Wars, 
Star Wars is probably together as it is a franchise name and Death Star is probalby together as well, 
because it is a connected name in the Star Wars franchise. 
- Step 4: Rebels will be translated as Rebellen, Star Wars as Star Wars and Death Star as Todesstern.
German (Germany): Wie haben die Rebellen aus Star Wars den Todesstern zerstört?

### Your translation
English: {x}"""

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
df_translation.to_csv("translation_COT_Qwen_DE.csv", index=False)
print(df_translation)