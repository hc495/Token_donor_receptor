python main_experiment.py \
    --model_name meta-llama/Llama-3.2-1B --dataset_path datasets/dataset.json --output_path res/output_replace.pkl \
    --device cuda:5 \
    --experiment_type donator \
    --aggression max \
    --replaced_source_key_word "famous people name" --replaced_container_key_word "normal people name"