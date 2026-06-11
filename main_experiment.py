import argparse
from utils.load_huggingface import load_hf_model
from utils.inference import process_one_word, process_standard_dataset, process_standard_dataset_with_replacement
import json
import os
import pickle

parser = argparse.ArgumentParser(description="Token Contextualization Main Experiment")

parser.add_argument("--model_name", type=str, required=True, help="Path to the HuggingFace model.")
parser.add_argument("--huggingface_token", type=str, default=None, help="Huggingface token for model access. Empty to use os.environ['HF_TOKEN'] or no token.")

parser.add_argument("--replaced_source_key_word", type=str, default=None, help="The key word in the dataset to be replaced. If not provided, no replacement will be done.")
parser.add_argument("--replaced_container_key_word", type=str, default=None, help="The key word in the dataset to replace with. If not provided, no replacement will be done.")

parser.add_argument("--dataset_path", type=str, default="datasets/dataset.json", help="Path to the dataset.")
parser.add_argument("--output_path", type=str, default="res/results.pkl", help="Path to save the output pickle file.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on.")

args = parser.parse_args()

model, tokenizer = load_hf_model(args.model_name, huggingface_token=args.huggingface_token, device=args.device)
with open(args.dataset_path, "r") as f:
    dataset = json.load(f)

if args.replaced_source_key_word is not None and args.replaced_container_key_word is not None:
    res = process_standard_dataset_with_replacement(
        model,
        tokenizer,
        dataset,
        args.replaced_source_key_word,
        args.replaced_container_key_word
    )
else:
    res = process_standard_dataset(model, tokenizer, dataset)

path = args.output_path
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(args.output_path, "wb") as f:
    pickle.dump(res, f)
with open(args.output_path + ".json", "w") as f:
    serializable_res = {}
    for word_type, word_dict in res.items():
        serializable_res[word_type] = {}
        for word, word_res in word_dict.items():
            serializable_res[word_type][word] = {
                "similarities": str(word_res["similarities"]),
            }
    json.dump(serializable_res, f, indent=4)