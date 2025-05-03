import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # import tqdm for progress bar

# Model and tokenizer loading
model_name = "support-pvelocity/Code-Llama-2-7B-instruct-text2sql"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)

def build_prompt(schema, question, knowledge):
    prompt = (
        "You are a helpful SQL assistant.\n"
        "Given the following database schema:\n"
        f"{schema.strip()}\n"
        "And the following relevant knowledge:\n"
        f"{knowledge.strip()}\n"
        "Write a syntactically correct and executable SQL query for this question:\n"
        f"{question.strip()}\n"
        "SQL:"
    )
    return prompt

# Load your expert knowledge JSON file
with open("./dpo_data/part2.json", "r") as f:
    examples = json.load(f)

results = []
step = 0

# Wrap the examples with tqdm to show progress bar
for ex in tqdm(examples, desc="Generating SQL queries", unit="example"):
    question = ex["question"]
    schema = ex["schema"]
    gold_knowledge = ex["gold_knowledge"]
    generated_knowledge = ex["generated_knowledge"]

    # Generate Gold SQL
    gold_prompt = build_prompt(schema, question, gold_knowledge)
    gold_inputs = tokenizer(gold_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gold_ids = model.generate(
            **gold_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
        )
    gold_output = tokenizer.decode(gold_ids[0], skip_special_tokens=True)
    gold_sql = gold_output.split("SQL:")[-1].strip()
    gold_sql = gold_sql.split("\n")[0]
    if not gold_sql.endswith(";"):
        gold_sql += ";"

    # Generate SQL using generated knowledge
    gen_prompt = build_prompt(schema, question, generated_knowledge)
    gen_inputs = tokenizer(gen_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(
            **gen_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    generated_sql = gen_output.split("SQL:")[-1].strip()
    generated_sql = generated_sql.split("\n")[0]
    if not generated_sql.endswith(";"):
        generated_sql += ";"

    results.append({
        "question": question,
        "schema": schema,
        "gold_knowledge": gold_knowledge,
        "generated_knowledge": generated_knowledge,
        "gold_sql": gold_sql,
        "generated_sql": generated_sql
    })

    step += 1
    if step % 50 == 0:
        with open(f"dpo_sql_generation_results_{step}.json", "w") as f:
            json.dump(results, f, indent=2)
        tqdm.write(f"Partial results saved after {step} steps.")

# Save final results to file
with open("dpo_sql_generation_results_final.json", "w") as f:
    json.dump(results, f, indent=2)

print("Gold and generated SQL generation complete. Final results saved to dpo_sql_generation_results_final.json")



