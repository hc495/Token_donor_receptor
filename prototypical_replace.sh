python main_experiment.py \
    --model_name meta-llama/Llama-3.2-1B --dataset_path datasets/dataset.json --output_path res/output.pkl \
    --device cuda:2 \
    --replaced_source_key_word "famous people name" --replaced_container_key_word "normal people name"