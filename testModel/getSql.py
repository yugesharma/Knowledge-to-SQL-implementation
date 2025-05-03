import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

# Hardcoded schema - this will be used for all queries
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
    parser = argparse.ArgumentParser(description="Generate SQL query from natural language question and expert knowledge")
    parser.add_argument("--model", type=str, default="support-pvelocity/Code-Llama-2-7B-instruct-text2sql",
                        help="Model for SQL generation")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for SQL generation")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate for SQL")
    parser.add_argument("--output_file", type=str, default="generated_sql_queries.txt",
                        help="File to save the generated SQL queries")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization to reduce memory usage")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization to reduce memory usage")
    return parser.parse_args()

def load_model(args):
    print(f"Loading SQL generation model: {args.model}")

    # Empty CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load model with quantization if specified
    if args.use_8bit:
        print("Using 8-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            load_in_8bit=True,
        )
    elif args.use_4bit:
        print("Using 4-bit quantization")
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map="auto",
                quantization_config=quantization_config,
            )
        except ImportError:
            print("BitsAndBytes not available, falling back to 16-bit precision")
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map="auto",
                torch_dtype=torch.float16,
            )
    else:
        # Default to FP16
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    return model, tokenizer

def generate_sql(schema, question, knowledge, model, tokenizer, args):
    prompt = (
        "You are a helpful SQL assistant.\n"
        "Given the following database schema:\n"
        f"{schema.strip()}\n\n"
        "And the following relevant knowledge:\n"
        f"{knowledge.strip()}\n\n"
        "Write a syntactically correct and executable SQL query for this question:\n"
        f"{question.strip()}\n\n"
        "SQL:"
    )

    # Print memory usage before generation
    if torch.cuda.is_available():
        print(f"GPU memory allocated before generation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Use a context manager to ensure CUDA memory is properly managed
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory error. Trying with CPU fallback...")
                # Move inputs to CPU
                inputs = {k: v.to('cpu') for k, v in inputs.items()}
                # Move model to CPU
                model.to('cpu')
                # Generate on CPU
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                raise e

    # Print memory usage after generation
    if torch.cuda.is_available():
        print(f"GPU memory allocated after generation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    sql_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql_query = sql_output.split("SQL:")[-1].strip()

    # Clean up the generated SQL
    sql_lines = sql_query.split("\n")
    clean_sql_lines = []

    for line in sql_lines:
        # Skip empty lines and lines that start with comments
        if line.strip() and not line.strip().startswith("--"):
            clean_sql_lines.append(line)
            # If we encounter a semicolon, stop processing further lines
            if ";" in line:
                break

    sql_query = " ".join(clean_sql_lines)

    # Add semicolon if missing
    if not sql_query.endswith(";"):
        sql_query += ";"

    return sql_query

def main():
    args = parse_args()

    # Load SQL generation model
    try:
        model, tokenizer = load_model(args)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get user input
    print("\n=== Natural Language to SQL Converter ===")
    print("Enter your natural language question and expert knowledge.")
    print("Type 'quit' to exit at any prompt.")

    while True:
        # Get natural language question
        question = input("\nQuestion: ")
        if question.lower() in ['quit', 'exit', 'q']:
            print("Exiting program.")
            break

        if not question.strip():
            print("Please enter a valid question.")
            continue

        # Get expert knowledge manually
        print("\nEnter expert knowledge (type 'done' on a new line when finished):")
        knowledge_lines = []
        while True:
            line = input()
            if line.lower() == 'done':
                break
            if line.lower() in ['quit', 'exit', 'q']:
                print("Exiting program.")
                return
            knowledge_lines.append(line)

        knowledge = "\n".join(knowledge_lines)

        if not knowledge.strip():
            print("Warning: No expert knowledge provided. This may affect SQL generation quality.")
            continue_anyway = input("Continue anyway? (y/n): ")
            if continue_anyway.lower() != 'y':
                continue

        # Generate SQL query
        try:
            print("\nGenerating SQL query...")

            # Clear CUDA cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            sql_query = generate_sql(
                SCHEMA,
                question,
                knowledge,
                model,
                tokenizer,
                args
            )

            # Display the generated SQL
            print("\n=== Generated SQL Query ===")
            print(sql_query)

            # Save to file
            with open(args.output_file, "a") as f:
                f.write(f"Question: {question}\n\n")
                f.write(f"Expert Knowledge:\n{knowledge}\n\n")
                f.write(f"SQL Query:\n{sql_query}\n\n")
                f.write("-" * 80 + "\n\n")

            print(f"\nQuery saved to {args.output_file}")

        except Exception as e:
            print(f"Error generating SQL: {e}")

            # Provide a helpful message for CUDA OOM errors
            if "CUDA out of memory" in str(e):
                print("\nMemory optimization tips:")
                print("1. Try running with --use_8bit or --use_4bit flags to reduce memory usage")
                print("2. Restart your Python environment to free up memory")
                print("3. Try a smaller model if available")

if __name__ == "__main__":
    main()
