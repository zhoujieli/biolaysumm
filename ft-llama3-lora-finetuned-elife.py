import transformers
import torch

# model_id = "aaditya/OpenBioLLM-Llama3-8B"
#model_path = "/home/dachengma/Llama3-OpenBioLLM-8B/"
# model_path = "/home/dachengma/models/LLM-Research/Meta-Llama-3-8B-Instruct"
model_path="/data/dachengma/dachengma/biolaysumm2024_data/LLaMA-Factory/llama3_lora_merged_eLife_train"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


def get_laysumm(df, i):
    title = df.iloc[i]['title']
    abstract = df.iloc[i]['abstract']
    topshot = df.iloc[i]['topshot']
    topshot_summ = df.iloc[i]["topshot_laysumm"]
    '''messages = [
        {"role": "system", "content": f"You are an expert in writing lay summaries for biomedical literature (a summary that is suitable for non-experts), \
         for example, here is the most similar example of abstract and paired lay summary that you should learn to follow: sample abstract, \
         sample lay summary , you should learn as much as can from this example, and then write lay summary for the abstract that is given to you. \
         write in readable way but maximize relevance and factuality.  Here is the example input abstract: {topshot}, example output lay summary:{topshot_summ}. PLEASE BE SHORT AND EASY TO UNDERSTAND(NO LONGER THAN 200 WORDS, AROUND 10 SENTENCES.) only respond the actual lay summary, here is the given title and abstract,"},
        {"role": "user", "content": f"{title}\n\n {abstract}"}
    ]'''
    
    messages = [
        {"role": "system", "content": f"Write lay summary for the given input (a summary that is suitable for non-experts). PLEASE BE SHORT AND EASY TO UNDERSTAND(NO LONGER THAN 200 WORDS, AROUND 10 SENTENCES).  Here is the example input abstract: {topshot}, example output lay summary:{topshot_summ}. Here is the given title and abstract,"},#Here is the example input abstract: {topshot}, example output lay summary:{topshot_summ}. PLEASE BE SHORT AND EASY TO UNDERSTAND(NO LONGER THAN 200 WORDS, AROUND 10 SENTENCES). 
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


from tqdm import tqdm
import pandas as pd
elife_test=pd.read_csv("/data/dachengma/dachengma/l3ft/elife_test_with_top_shot.csv")
# !export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elife_responses = []
'''
for i in tqdm(range(len(elife_test))):
    response = get_laysumm(elife_test, i)
    elife_responses.append(response)
    
    
# Open the file in text write mode ('w') instead of binary write mode ('wb')
with open('elife_llama3_lora_finetuned_top1_responses_5-5_try2.txt', 'w') as f:
    for res in elife_responses:
        # Remove all newline characters from the string
        res = res.replace('\n', '')
        # Write the modified string to the file, adding a newline character after each string
        f.write(res + '\n')
'''        
        

# Process each row and write the response to the file
for i in tqdm(range(len(elife_test))):
    response = get_laysumm(elife_test, i)
    # Normalize the response to remove unexpected newlines and excess spaces
    response = response.replace('\n', ' ').strip()
    # Open the file in append mode and write the response
    with open('elife_llama3_lora_finetuned_top1_responses_5-14_maxtoken256_topshot_temp099.txt', 'a') as f:
        f.write(response + '\n')
