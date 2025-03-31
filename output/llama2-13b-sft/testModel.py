import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Step 1: Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", device_map="auto")
model = PeftModel.from_pretrained(base_model, "./output/llama2-13b-sft", device_map="auto")
model.eval()

# Step 2 & 3: Prepare the prompt and generate expert knowledge
def generate_knowledge(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            num_beams=4,
            repetition_penalty=1.15,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("###Assistant:")[-1].strip()

# Example prompt
prompt = """###Human: For all customers who paid more than $50, list their names and order dates.
TABLE orders (order_id, customer_id, amount, order_date)
TABLE customers (customer_id, name)
###Assistant:"""

# Generate expert knowledge
generated_knowledge = generate_knowledge(prompt, model, tokenizer)
print("Generated Knowledge:")
print(generated_knowledge)

