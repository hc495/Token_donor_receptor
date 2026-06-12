import argparse
from utils.load_huggingface import load_hf_model
from Token_donor_receptor.utils.exp2_kit import process_standard_dataset
import json
import os
import pickle

parser = argparse.ArgumentParser(description="Token Contextualization Main Experiment")

parser.add_argument("--model_name", type=str, required=True, help="Path to the HuggingFace model.")
parser.add_argument("--huggingface_token", type=str, default=None, help="Huggingface token for model access. Empty to use os.environ['HF_TOKEN'] or no token.")

parser.add_argument("--step_1", action="store_true", help="Whether to run step 1 of the experiment (inner semantics amount).")
parser.add_argument("--step_2", action="store_true", help="Whether to run step 2 of the experiment (semantics receptor and donator).")

parser.add_argument("--replaced_source_key_word", type=str, default=None, help="The key word in the dataset to be replaced. If not provided, no replacement will be done.")
parser.add_argument("--replaced_container_key_word", type=str, default=None, help="The key word in the dataset to replace with. If not provided, no replacement will be done.")
parser.add_argument("--experiment_type", type=str, default="receptor", help="Type of experiment to run. Options: 'receptor', 'donator', 'both'.")
parser.add_argument("--aggression", type=str, default="max", help="Aggression method for similarity calculation. Only used in donator experiments. Options: 'mean', 'max'.")
parser.add_argument("--save_hidden_states", action="store_true", help="Whether to save hidden states in the output. This may significantly increase the output file size.")

parser.add_argument("--dataset_path", type=str, default="datasets/dataset.json", help="Path to the dataset.")
parser.add_argument("--output_path", type=str, default="res/results.pkl", help="Path to save the output pickle file.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on.")

args = parser.parse_args()

model, tokenizer = load_hf_model(args.model_name, huggingface_token=args.huggingface_token, device=args.device)
with open(args.dataset_path, "r") as f:
    dataset = json.load(f)


## Step 2
if args.step_2:
    if args.experiment_type != "both":
        res = process_standard_dataset(
            model,
            tokenizer,
            dataset,
            args.replaced_source_key_word,
            args.replaced_container_key_word,
            experiment_type=args.experiment_type,
            donator_aggression=args.aggression,
            save_hidden_states=args.save_hidden_states, 
        )
        path = args.output_path + f"_{args.experiment_type}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(res, f)
        with open(path + ".json", "w") as f:
            serializable_res = {}
            for word_type, word_dict in res.items():
                serializable_res[word_type] = {}
                for word, word_res in word_dict.items():
                    serializable_res[word_type][word] = {
                        "similarities": str(word_res["similarities"]),
                    }
            json.dump(serializable_res, f, indent=4)
    else:
        res_receptor = process_standard_dataset(
            model,
            tokenizer,
            dataset,
            replaced_source_key_word=args.replaced_source_key_word,
            replaced_container_key_word=args.replaced_container_key_word,
            experiment_type="receptor",
            donator_aggression=args.aggression,
            save_hidden_states=args.save_hidden_states, 
        )
        path = args.output_path + "_receptor"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(res_receptor, f)
        with open(path + ".json", "w") as f:
            serializable_res = {}
            for word_type, word_dict in res_receptor.items():
                serializable_res[word_type] = {}
                for word, word_res in word_dict.items():
                    serializable_res[word_type][word] = {
                        "similarities": str(word_res["similarities"]),
                    }
            json.dump(serializable_res, f, indent=4)
        res_donator = process_standard_dataset(
            model,
            tokenizer,
            dataset,
            replaced_source_key_word=args.replaced_source_key_word,
            replaced_container_key_word=args.replaced_container_key_word,
            experiment_type="donator",
            donator_aggression=args.aggression,
            save_hidden_states=args.save_hidden_states,
        )
        path = args.output_path + "_donator"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(res_donator, f)
        with open(path + ".json", "w") as f:
            serializable_res = {}
            for word_type, word_dict in res_donator.items():
                serializable_res[word_type] = {}
                for word, word_res in word_dict.items():
                    serializable_res[word_type][word] = {
                        "similarities": str(word_res["similarities"]),
                    }
            json.dump(serializable_res, f, indent=4)