import numpy as np
from tqdm import tqdm as tqdm

def test_tokenizer_prefix_length(
    tokenizer
):
    test_case = "A"
    tokenized = tokenizer(test_case, add_special_tokens=False)
    return len(tokenized.input_ids) - 1


def find_token_location_in_sentence(
    tokenizer,
    sentence,
    word,
):
    prefix_length = test_tokenizer_prefix_length(tokenizer)
    word_tokenized = tokenizer(word, add_special_tokens=False)
    word_tokenized_ids = word_tokenized.input_ids[prefix_length:]
    word_tokenized_with_space = tokenizer(" " + word, add_special_tokens=False)
    word_tokenized_with_space_ids = word_tokenized_with_space.input_ids[prefix_length:]
    caps_word_tokenized = tokenizer(word.capitalize(), add_special_tokens=False)    
    caps_word_tokenized_ids = caps_word_tokenized.input_ids[prefix_length:]
    sentence_tokenized = tokenizer(sentence, add_special_tokens=False)
    sentence_tokenized_ids = sentence_tokenized.input_ids
    word_begin = None
    word_end = None
    for i in range(len(sentence_tokenized_ids)):
        if sentence_tokenized_ids[i:i+len(word_tokenized_ids)] == word_tokenized_ids or sentence_tokenized_ids[i:i+len(word_tokenized_with_space_ids)] == word_tokenized_with_space_ids or sentence_tokenized_ids[i:i+len(caps_word_tokenized_ids)] == caps_word_tokenized_ids:
            word_begin = i
            word_end = i + len(word_tokenized_ids)
            break

    return word_begin, word_end


def process_one_word(
    model,
    tokenizer,
    word,
    sentences
):
    # prefix_length = test_tokenizer_prefix_length(tokenizer)
    # word_tokenized = tokenizer(word, add_special_tokens=False)
    # word_tokenized_ids = word_tokenized.input_ids[prefix_length:]
    # word_tokenized_with_space = tokenizer(" " + word, add_special_tokens=False)
    # word_tokenized_with_space_ids = word_tokenized_with_space.input_ids[prefix_length:]
    # caps_word_tokenized = tokenizer(word.capitalize(), add_special_tokens=False)    
    # caps_word_tokenized_ids = caps_word_tokenized.input_ids[prefix_length:]
    res = []
    for sentence in sentences:
        # sentence_tokenized = tokenizer(sentence, add_special_tokens=False)
        # sentence_tokenized_ids = sentence_tokenized.input_ids
        # ## Check if the word is in the sentence
        # word_begin = None
        # word_end = None
        # for i in range(len(sentence_tokenized_ids)):
        #     if sentence_tokenized_ids[i:i+len(word_tokenized_ids)] == word_tokenized_ids or sentence_tokenized_ids[i:i+len(word_tokenized_with_space_ids)] == word_tokenized_with_space_ids or sentence_tokenized_ids[i:i+len(caps_word_tokenized_ids)] == caps_word_tokenized_ids:
        #         # Found the word in the sentence
        #         word_begin = i
        #         word_end = i + len(word_tokenized_ids)
        #         break
        word_begin, word_end = find_token_location_in_sentence(tokenizer, sentence, word)
        if word_begin is None or word_end is None:
            print(f"Warning: The word '{word}' is not found in the sentence '{sentence}'. Skipping this sentence.")
            # print(f"Tokenized sentence ids: {sentence_tokenized_ids}, word tokenized ids: {word_tokenized_ids}, word tokenized with space ids: {word_tokenized_with_space_ids}, caps word tokenized ids: {caps_word_tokenized_ids}")
            continue
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        hs_in_layer = []
        for layer in range(len(hidden_states)):
            hs_in_layer.append(hidden_states[layer][0, word_begin:word_end, :].detach().cpu().numpy())
        res.append(hs_in_layer)

    if not res:
        return {
            "word": word,
            "similarities": [],
            "hidden_states": [],
        }

    pre_processed_res = []
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