import json
from datasets import load_dataset

"""
data = [
{
    "messages" : []
}
]

"""

def convert_json_to_huggingface(data_path, save_data_path):
    dataset = load_dataset("json", data_files=data_path)
    dataset.save_to_disk(save_data_path)
    print(dataset)
    print(dataset["train"][0])


if __name__ == "__main__":
    convert_json_to_huggingface("data/prm800k/phase2_train_final_critique_formatted.json", 
    "data/prm800k/processed_sft_data/phase2_train_final_critique")
    