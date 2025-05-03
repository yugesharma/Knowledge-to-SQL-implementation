import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from tqdm import tqdm
import os

# Load the base model and tokenizer
base_model_path = "meta-llama/Llama-2-13b-chat-hf"
adapter_path = "./output/llama2-13b-sft"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

print("Loading base model...")
# Create an offload directory
os.makedirs("offload", exist_ok=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    offload_folder="offload"
)
model.eval()

# Load your processed dataset
print("Loading dataset...")
with open("model/processed_train.json", "r") as f:
    dataset = json.load(f)
    # Function to generate knowledge with chunking for long schemas
def generate_knowledge(question, schema, max_new_tokens=512):
    # Truncate schema if it's too long
    if len(schema) > 4000:  # Arbitrary threshold
        schema = schema[:4000] + "..."

    prompt = f"""Given the following database schema and question, generate the knowledge needed to write a SQL query:

Schema: {schema}
Question: {question}

Knowledge:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )

    knowledge = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the knowledge part (after "Knowledge:")
    knowledge = knowledge.split("Knowledge:")[-1].strip()
    return knowledge

# Create output directory if it doesn't exist
os.makedirs("dpo_data", exist_ok=True)

# Process each example and generate knowledge
results = []
print("Generating knowledge...")
for example in tqdm(dataset):
    try:
         # Parse schema and question from the input field
        input_text = example["input"]

        # Extract schema and question from the input field
        schema_part = input_text.split("--question:")[0].strip()
        if schema_part.startswith("--schema:"):
            schema = schema_part[len("--schema:"):].strip()
        else:
            schema = schema_part

        question_part = input_text.split("--question:")[1].strip() if "--question:" in input_text else ""

        # Generate knowledge using the extracted schema and question
        generated_knowledge = generate_knowledge(question_part, schema)

        # Get the gold knowledge from the output field
        gold_knowledge = example.get("output", "")

        results.append({
            "question": question_part,
            "schema": schema,
            "generated_knowledge": generated_knowledge,
            "gold_knowledge": gold_knowledge
        })

        # Save intermediate results every 100 examples
        if len(results) % 100 == 0:
            print(f"Processed {len(results)} examples. Saving intermediate results...")
            with open(f"dpo_data/generated_expert_knowledge_partial_{len(results)}.json", "w") as f:
                json.dump(results, f, indent=2)

    except Exception as e:
        print(f"Error processing example: {e}")
        continue

# Save the final results
print("Saving results...")
with open("dpo_data/generated_expert_knowledge.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Generated expert knowledge for {len(results)} examples")
