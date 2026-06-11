import numpy as np
from tqdm import tqdm as tqdm
import torch

def test_tokenizer_prefix_length(
    tokenizer
):
    test_case = "A"
    tokenized = tokenizer(test_case)
    return len(tokenized.input_ids) - 1


def find_token_location_in_sentence(
    tokenizer,
    sentence,
    word,
):
    prefix_length = test_tokenizer_prefix_length(tokenizer)
    word_tokenized = tokenizer(word)
    word_tokenized_ids = word_tokenized.input_ids[prefix_length:]
    word_tokenized_with_space = tokenizer(" " + word)
    word_tokenized_with_space_ids = word_tokenized_with_space.input_ids[prefix_length:]
    caps_word_tokenized = tokenizer(word.capitalize())    
    caps_word_tokenized_ids = caps_word_tokenized.input_ids[prefix_length:]
    sentence_tokenized = tokenizer(sentence)
    sentence_tokenized_ids = sentence_tokenized.input_ids
    word_begin = None
    word_end = None
    for i in range(len(sentence_tokenized_ids)):
        if sentence_tokenized_ids[i:i+len(word_tokenized_ids)] == word_tokenized_ids:
            word_begin = i
            word_end = i + len(word_tokenized_ids)
            break 
        elif sentence_tokenized_ids[i:i+len(word_tokenized_with_space_ids)] == word_tokenized_with_space_ids:
            word_begin = i
            word_end = i + len(word_tokenized_with_space_ids)
            break
        elif sentence_tokenized_ids[i:i+len(caps_word_tokenized_ids)] == caps_word_tokenized_ids:
            word_begin = i
            word_end = i + len(caps_word_tokenized_ids)
            break
    return word_begin, word_end


def replace_token_in_sentence(
    tokenized_sentence,
    word_begin,
    word_end,
    new_token_ids
):
    new_tokenized_sentence = tokenized_sentence.copy()
    new_tokenized_sentence.input_ids = torch.cat([
        tokenized_sentence.input_ids[:, :word_begin],
        new_token_ids.to(tokenized_sentence.input_ids.device),
        tokenized_sentence.input_ids[:, word_end:]
    ], dim=1)
    new_tokenized_sentence.attention_mask = torch.cat([
        tokenized_sentence.attention_mask[:, :word_begin],
        torch.ones_like(new_token_ids).to(tokenized_sentence.attention_mask.device),
        tokenized_sentence.attention_mask[:, word_end:]
    ], dim=1)
    
    new_word_begin = word_begin
    new_word_end = word_begin + new_token_ids.shape[1]
    return new_tokenized_sentence, new_word_begin, new_word_end


def process_one_word(
    model,
    tokenizer,
    word,
    sentences,
    replaced_word = None
):
    res = []
    for sentence in sentences:
        word_begin, word_end = find_token_location_in_sentence(tokenizer, sentence, word)
        if word_begin is None or word_end is None:
            print(f"Warning: The word '{word}' is not found in the sentence '{sentence}'. Skipping this sentence.")
            continue
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        if replaced_word is not None:
            # print(f"Replacing '{word}' with '{replaced_word}' in the sentence '{sentence}'.")
            # print(f"Original token ids: {inputs.input_ids}")
            if word_begin == test_tokenizer_prefix_length(tokenizer):
                replaced_word_real = replaced_word
            elif tokenizer.decode(inputs.input_ids[0, word_begin - 1])[-1] == ' ':
                replaced_word_real = replaced_word + ' '
            else:
                replaced_word_real = ' ' + replaced_word
            replaced_word_tokenized = tokenizer(replaced_word_real, return_tensors="pt")
            replaced_word_tokenized_ids = replaced_word_tokenized.input_ids[:, test_tokenizer_prefix_length(tokenizer):]
            inputs, word_begin, word_end = replace_token_in_sentence(inputs, word_begin, word_end, replaced_word_tokenized_ids)
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        hs_in_layer = []
        for layer in range(len(hidden_states)):
            hs_in_layer.append(hidden_states[layer][0, word_begin:word_end, :].detach().to(torch.float32).cpu().numpy())
        res.append(hs_in_layer)

    if not res:
        return {
            "word": word,
            "similarities": [],
            "hidden_states": [],
        }

    pre_processed_res = [] # Token Averaged
    for i in range(len(res)):
        pre_processed_res.append([])
        for layer in range(len(res[i])):
            pre_processed_res[i].append(np.mean(res[i][layer], axis=0, keepdims=False))

    sim_res = []
    for layer in range(len(pre_processed_res[0])):
        averaged_feature = sum([pre_processed_res[i][layer] for i in range(len(pre_processed_res))]) / len(pre_processed_res)
        temp_sim_res_layer = []
        for i in range(len(pre_processed_res)):
            sim = np.dot(pre_processed_res[i][layer], averaged_feature) / (np.linalg.norm(pre_processed_res[i][layer]) * np.linalg.norm(averaged_feature))
            temp_sim_res_layer.append(sim)
        sim_res.append(np.mean(temp_sim_res_layer).item())

    return {
        "word": word,
        "similarities": sim_res,
        "hidden_states": res,
    }


def process_standard_dataset(
    model,
    tokenizer,
    dataset: dict[str, dict[str, list]]
):
    res = {}
    for word_type, word_sentences_dict in tqdm(dataset.items(), desc="Processing dataset"):
        res[word_type] = {}
        for word, sentences in tqdm(word_sentences_dict.items(), desc=f"Processing words in {word_type}", leave=False):
            res[word_type][word] = process_one_word(model, tokenizer, word, sentences)
    return res


def process_standard_dataset_with_replacement(
    model,
    tokenizer,
    dataset: dict[str, dict[str, list]],
    replaced_source_key_word: str,
    replaced_container_key_word: str,
):
    res = {}
    replaced_source_list = []
    for word, sentences in dataset[replaced_source_key_word].items():
        replaced_source_list.append(word)
    res[replaced_source_key_word + " replaced"] = {}
    for source_word in tqdm(replaced_source_list):
        for replaced_container_key, sentences in tqdm(dataset[replaced_container_key_word].items(), desc=f"Processing words in {replaced_source_key_word} with replacement from {replaced_container_key_word}", leave=False):
            res[replaced_source_key_word + " replaced"][replaced_container_key + " -> " + source_word] = process_one_word(
                model, 
                tokenizer, 
                word = replaced_container_key, 
                sentences = dataset[replaced_container_key_word][replaced_container_key], 
                    replaced_word = source_word
                )
    return res