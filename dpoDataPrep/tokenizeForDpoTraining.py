import json
from transformers import AutoTokenizer, AutoModel
import os

def tokenize_and_format_dpo(data_file, tokenizer_dir, output_file, max_length=1024):
    """
    Tokenizes data from a JSONL file, formats it for DPO training, and saves it to a new JSONL file.

    Args:
        data_file (str): Path to the input JSONL file containing "prompt", "chosen", and "rejected" keys.
        tokenizer_dir (str): Path to the directory containing the tokenizer files (tokenizer.json, etc.).
        output_file (str): Path to the output JSONL file to save the tokenized data.
        max_length (int): Maximum sequence length for tokenization.
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)  # Load from the directory
    tokenizer.pad_token = tokenizer.eos_token # crucial if tokenizer doesn't have a pad token

    with open(data_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            try:
                example = json.loads(line.strip())

                # Check for required keys
                if not all(key in example for key in ["prompt", "chosen", "rejected"]):
                    print(f"Skipping example due to missing keys: {example.keys()}")
                    continue

                prompt = example["prompt"]
                chosen = example["chosen"]
                rejected = example["rejected"]

                # Tokenize the data
                prompt_tokenized = tokenizer(prompt, truncation=True, max_length=max_length, padding="longest", return_tensors="pt") # Use padding="longest" for consistent shape
                chosen_tokenized = tokenizer(chosen, truncation=True, max_length=max_length, padding="longest", return_tensors="pt")
                rejected_tokenized = tokenizer(rejected, truncation=True, max_length=max_length, padding="longest", return_tensors="pt")

                # Create the DPO format
                dpo_example = {
                    "prompt_ids": prompt_tokenized.input_ids[0].tolist(),
                    "chosen_ids": chosen_tokenized.input_ids[0].tolist(),
                    "rejected_ids": rejected_tokenized.input_ids[0].tolist()
                }

                # Write to the output file
                outfile.write(json.dumps(dpo_example) + '\n')

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
            except Exception as e:
                print(f"Skipping example due to error: {e}")

if __name__ == "__main__":
    # Example Usage (replace with your actual paths and tokenizer)
    data_file = "model/dpo_training_data_shree.jsonl"  # Path to your input JSONL file
    tokenizer_dir = "./output/llama2-13b-sft"  # Path to your SFT model output directory
    output_file = "dpo_data/dpo_training_data_tokenized.jsonl"  # Path to the output JSONL file

    tokenize_and_format_dpo(data_file, tokenizer_dir, output_file)

    print(f"Tokenized data saved to {output_file}")

