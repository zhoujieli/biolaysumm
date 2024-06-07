import transformers
import torch
import pandas as pd
import json
from tqdm import tqdm
few_shot=True
# Initialize the pipeline configuration
model_path = "/data/dachengma/dachengma/biolaysumm2024_data/LLaMA-Factory/llama3_lora_merged_eLife_train"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Function to generate lay summary from DataFrame
def get_laysumm(df, i,mode):
    
    
    if mode=="val":
        # title = df.iloc[i]['title']
        abstract = df.iloc[i]['abstract']
        topshot = df.iloc[i]['topshot_abstract']
        topshot_summ = df.iloc[i]["topshot_laysumm"]
        if few_shot:
            messages = [
            {"role": "system", "content": f"Write lay summary for the given input (a summary that is suitable for non-experts). PLEASE BE SHORT AND EASY TO UNDERSTAND(NO LONGER THAN 200 WORDS, AROUND 10 SENTENCES).  Here is the example input abstract: {topshot}, example output lay summary:{topshot_summ}. Here is the given title and abstract,"},
            {"role": "user", "content": f"{abstract}"}
            ]
        else:
            messages = [
            {"role": "system", "content": f"Write lay summary for the given input (a summary that is suitable for non-experts). PLEASE BE SHORT AND EASY TO UNDERSTAND(NO LONGER THAN 200 WORDS, AROUND 10 SENTENCES).  Here is the given title and abstract,"},
            {"role": "user", "content": f"{abstract}"}
            ]
    elif mode=="test":
        title = df.iloc[i]['title']
        abstract = df.iloc[i]['abstract']
        topshot = df.iloc[i]['topshot']
        topshot_summ = df.iloc[i]["topshot_laysumm"]
        messages = [
            {"role": "system", "content": f"Write lay summary for the given input (a summary that is suitable for non-experts). PLEASE BE SHORT AND EASY TO UNDERSTAND(NO LONGER THAN 200 WORDS, AROUND 10 SENTENCES).  Here is the example input abstract: {topshot}, example output lay summary:{topshot_summ}. Here is the given title and abstract,"},
            {"role": "user", "content": f"{title}\n\n {abstract}"}
        ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,#2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.99,#0.0001,
        top_p=0.9,
        #repetition_penalty=1,  
    )
    response = outputs[0]["generated_text"][len(prompt):]
    return response

# Mode variable to switch between "test" and "val"
mode = "val"  # Set to either "test" or "val"

# Load data based on mode
if mode == "test":
    data_file = "/data/dachengma/dachengma/l3ft/elife_test_with_top_shot.csv"
    data_df = pd.read_csv(data_file)
else:
    journal = "eLife"  # or "PLOS"
    data_file = "/data/dachengma/dachengma/l3ft/elife_val_with_top_shot.csv"
    #data_file = f"/data/dachengma/dachengma/biolaysumm2024_data/{journal}_val_abstracts_and_summaries.json"
    '''
    with open(data_file, 'r') as f:
        data = json.load(f)
    data_df = pd.DataFrame(data)'''
    data_df = pd.read_csv(data_file)

# Process data and generate lay summaries
responses = []
for i in tqdm(range(len(data_df))):
    response = get_laysumm(data_df, i,mode)
    response = response.replace('\n', ' ').strip()
    responses.append(response)

# Write responses to a file
if few_shot:
    output_file = f'{mode}_{journal}_llama3_infer_5-19_temp099_maxtoken256_topshot.txt'
else:
    output_file = f'{mode}_{journal}_llama3_infer_5-19_temp099_maxtoken256.txt'
with open(output_file, 'w') as f:
    for res in responses:
        f.write(res + '\n')
