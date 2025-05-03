import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse

# Define the hardcoded schema
SCHEMA = """DROP TABLE IF EXISTS PATIENTS;
CREATE TABLE PATIENTS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL UNIQUE,
    GENDER VARCHAR(5) NOT NULL,
    DOB TIMESTAMP(0) NOT NULL,
    DOD TIMESTAMP(0)
);

DROP TABLE IF EXISTS ADMISSIONS;
CREATE TABLE ADMISSIONS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL UNIQUE,
    ADMITTIME TIMESTAMP(0) NOT NULL,
    DISCHTIME TIMESTAMP(0),
    ADMISSION_TYPE VARCHAR(50) NOT NULL,
    ADMISSION_LOCATION VARCHAR(50) NOT NULL,
    DISCHARGE_LOCATION VARCHAR(50),
    INSURANCE VARCHAR(255) NOT NULL,
    LANGUAGE VARCHAR(10),
    MARITAL_STATUS VARCHAR(50),
    ETHNICITY VARCHAR(200) NOT NULL,
    AGE INT NOT NULL,
    FOREIGN KEY(SUBJECT_ID) REFERENCES PATIENTS(SUBJECT_ID)
);

DROP TABLE IF EXISTS D_ICD_DIAGNOSES;
CREATE TABLE D_ICD_DIAGNOSES
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    ICD9_CODE VARCHAR(10) NOT NULL UNIQUE,
    SHORT_TITLE VARCHAR(50) NOT NULL,
    LONG_TITLE VARCHAR(255) NOT NULL
);

DROP TABLE IF EXISTS D_ICD_PROCEDURES;
CREATE TABLE D_ICD_PROCEDURES
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    ICD9_CODE VARCHAR(10) NOT NULL UNIQUE,
    SHORT_TITLE VARCHAR(50) NOT NULL,
    LONG_TITLE VARCHAR(255) NOT NULL
);

DROP TABLE IF EXISTS D_LABITEMS;
CREATE TABLE D_LABITEMS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    ITEMID INT NOT NULL UNIQUE,
    LABEL VARCHAR(200) NOT NULL
);

DROP TABLE IF EXISTS D_ITEMS;
CREATE TABLE D_ITEMS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    ITEMID INT NOT NULL UNIQUE,
    LABEL VARCHAR(200) NOT NULL,
    LINKSTO VARCHAR(50) NOT NULL
);

DROP TABLE IF EXISTS DIAGNOSES_ICD;
CREATE TABLE DIAGNOSES_ICD
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICD9_CODE VARCHAR(10) NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICD9_CODE) REFERENCES D_ICD_DIAGNOSES(ICD9_CODE)
);

DROP TABLE IF EXISTS PROCEDURES_ICD;
CREATE TABLE PROCEDURES_ICD
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICD9_CODE VARCHAR(10) NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICD9_CODE) REFERENCES D_ICD_PROCEDURES(ICD9_CODE)
);

DROP TABLE IF EXISTS LABEVENTS;
CREATE TABLE LABEVENTS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ITEMID INT NOT NULL,
    CHARTTIME TIMESTAMP(0),
    VALUENUM DOUBLE PRECISION,
    VALUEUOM VARCHAR(20),
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ITEMID) REFERENCES D_LABITEMS(ITEMID)
);

DROP TABLE IF EXISTS PRESCRIPTIONS;
CREATE TABLE PRESCRIPTIONS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    STARTDATE TIMESTAMP(0) NOT NULL,
    ENDDATE TIMESTAMP(0),
    DRUG VARCHAR(100) NOT NULL,
    DOSE_VAL_RX VARCHAR(120) NOT NULL,
    DOSE_UNIT_RX VARCHAR(120) NOT NULL,
    ROUTE VARCHAR(120) NOT NULL,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);

DROP TABLE IF EXISTS COST;
CREATE TABLE COST
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    EVENT_TYPE VARCHAR(20) NOT NULL,
    EVENT_ID INT NOT NULL,
    CHARGETIME TIMESTAMP(0) NOT NULL,
    COST DOUBLE PRECISION NOT NULL,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(EVENT_ID) REFERENCES DIAGNOSES_ICD(ROW_ID),
    FOREIGN KEY(EVENT_ID) REFERENCES PROCEDURES_ICD(ROW_ID),
    FOREIGN KEY(EVENT_ID) REFERENCES LABEVENTS(ROW_ID),
    FOREIGN KEY(EVENT_ID) REFERENCES PRESCRIPTIONS(ROW_ID)
);

DROP TABLE IF EXISTS CHARTEVENTS;
CREATE TABLE CHARTEVENTS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT NOT NULL,
    ITEMID INT NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    VALUENUM DOUBLE PRECISION,
    VALUEUOM VARCHAR(50),
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICUSTAY_ID) REFERENCES ICUSTAYS(ICUSTAY_ID),
    FOREIGN KEY(ITEMID) REFERENCES D_ITEMS(ITEMID)
);

DROP TABLE IF EXISTS INPUTEVENTS_CV;
CREATE TABLE INPUTEVENTS_CV
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    ITEMID INT NOT NULL,
    AMOUNT DOUBLE PRECISION,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICUSTAY_ID) REFERENCES ICUSTAYS(ICUSTAY_ID),
    FOREIGN KEY(ITEMID) REFERENCES D_ITEMS(ITEMID)
);

DROP TABLE IF EXISTS OUTPUTEVENTS;
CREATE TABLE OUTPUTEVENTS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    ITEMID INT NOT NULL,
    VALUE DOUBLE PRECISION,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICUSTAY_ID) REFERENCES ICUSTAYS(ICUSTAY_ID),
    FOREIGN KEY(ITEMID) REFERENCES D_ITEMS(ITEMID)
);

DROP TABLE IF EXISTS MICROBIOLOGYEVENTS;
CREATE TABLE MICROBIOLOGYEVENTS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    SPEC_TYPE_DESC VARCHAR(100),
    ORG_NAME VARCHAR(100),
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);

DROP TABLE IF EXISTS ICUSTAYS;
CREATE TABLE ICUSTAYS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT NOT NULL,
    FIRST_CAREUNIT VARCHAR(20) NOT NULL,
    LAST_CAREUNIT VARCHAR(20) NOT NULL,
    FIRST_WARDID SMALLINT NOT NULL,
    LAST_WARDID SMALLINT NOT NULL,
    INTIME TIMESTAMP(0) NOT NULL,
    OUTTIME TIMESTAMP(0),
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);

DROP TABLE IF EXISTS TRANSFERS;
CREATE TABLE TRANSFERS
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT,
    EVENTTYPE VARCHAR(20) NOT NULL,
    CAREUNIT VARCHAR(20),
    WARDID SMALLINT,
    INTIME TIMESTAMP(0) NOT NULL,
    OUTTIME TIMESTAMP(0),
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);"""

def parse_args():
    parser = argparse.ArgumentParser(description="Generate expert knowledge for a natural language question")
    parser.add_argument("--model_path", type=str, default="./output/llama2-13b-sft", 
                        help="Path to the SFT trained model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-13b-chat-hf", 
                        help="Base model path")
    parser.add_argument("--temperature", type=float, default=0.2, 
                        help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=512, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--output_file", type=str, default="expert_knowledge_output.txt", 
                        help="File to save the generated knowledge")
    return parser.parse_args()

def load_model(args):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("Loading base model...")
    # Create an offload directory
    os.makedirs("offload", exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload"
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        args.model_path,
        offload_folder="offload"
    )
    model.eval()
    
    return model, tokenizer

def generate_knowledge(question, schema, model, tokenizer, max_new_tokens=512, temperature=0.7):
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
            temperature=temperature,
            do_sample=True
        )

    knowledge = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the knowledge part (after "Knowledge:")
    knowledge = knowledge.split("Knowledge:")[-1].strip()
    return knowledge

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model(args)
    
    # Get user input
    print("\n=== Expert Knowledge Generator ===")
    print("Enter your natural language question (or 'quit' to exit):")
    
    while True:
        question = input("\nQuestion: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Exiting program.")
            break
            
        if not question.strip():
            print("Please enter a valid question.")
            continue
            
        print("\nGenerating expert knowledge...")
        try:
            # Generate knowledge using the hardcoded schema
            knowledge = generate_knowledge(
                question, 
                SCHEMA, 
                model, 
                tokenizer, 
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Display the generated knowledge
            print("\n=== Generated Expert Knowledge ===")
            print(knowledge)
            
            # Save to file
            with open(args.output_file, "a") as f:
                f.write(f"Question: {question}\n\n")
                f.write(f"Expert Knowledge:\n{knowledge}\n\n")
                f.write("-" * 80 + "\n\n")
                
            print(f"\nKnowledge saved to {args.output_file}")
            
        except Exception as e:
            print(f"Error generating knowledge: {e}")

if __name__ == "__main__":
    main()

