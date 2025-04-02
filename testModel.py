import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Set the offload directory
offload_dir = "./offload"
os.makedirs(offload_dir, exist_ok=True)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    device_map="auto",
    offload_folder=offload_dir,
)
model = PeftModel.from_pretrained(
    base_model,
    "./output/llama2-13b-sft",
    device_map="auto",
    offload_folder=offload_dir,
)
model.eval()

def generate_knowledge(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=300,  # Increased to avoid truncation
            temperature=0.7,
            top_p=0.9,
            num_beams=4,
            repetition_penalty=1.15,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("###Assistant:")[-1].strip()

# Hardcoded test inputs
user_question = "List doctors who have treated more than 10 patients with diabetes."
database_schema = "TABLE patients (patient_id, name, age, gender, diagnosis) TABLE visits (visit_id, patient_id, doctor_id, hospital_id, visit_date) TABLE doctors (doctor_id, name, specialization) TABLE hospitals (hospital_id, name, location)"

# Construct the prompt
prompt = f"""###Human: {user_question} {database_schema}
###Assistant:"""

# Generate and print knowledge
generated_knowledge = generate_knowledge(prompt, model, tokenizer)
print("Generated Knowledge:")
print(generated_knowledge)

