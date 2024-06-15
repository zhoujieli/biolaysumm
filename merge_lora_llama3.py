import json

args = dict(
  model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct", # use official non-quantized Llama-3-8B-Instruct model
  adapter_name_or_path="llama3_lora_eLife_train",            # load the saved LoRA adapters
  template="llama3",                     # same to the one in training
  finetuning_type="lora",                  # same to the one in training
  export_dir="llama3_lora_merged_eLife",              # the path to save the merged model
  export_size=2,                       # the file shard size (in GB) of the merged model
  export_device="cuda",                    # the device used in export, can be chosen from `cpu` and `cuda`
  #export_hub_model_id="your_id/your_model",         # the Hugging Face hub ID to upload model
)

json.dump(args, open("merge_llama3.json", "w", encoding="utf-8"), indent=2)

%cd /content/LLaMA-Factory/

!llamafactory-cli export merge_llama3.json
