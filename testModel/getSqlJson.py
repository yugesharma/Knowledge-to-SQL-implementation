import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import os

# Your full schema string here
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
);
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SQL queries from JSON input")
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file")
    parser.add_argument("--model", type=str, default="support-pvelocity/Code-Llama-2-7B-instruct-text2sql")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    return parser.parse_args()

def load_model(args):
    print(f"Loading SQL generation model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.use_8bit:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", load_in_8bit=True)
    elif args.use_4bit:
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
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
                inputs = {k: v.to('cpu') for k, v in inputs.items()}
                model.to('cpu')
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                raise e
    sql_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql_query = sql_output.split("SQL:")[-1].strip()
    # Clean up the generated SQL
    sql_lines = sql_query.split("\n")
    clean_sql_lines = []
    for line in sql_lines:
        if line.strip() and not line.strip().startswith("--"):
            clean_sql_lines.append(line)
            if ";" in line:
                break
    sql_query = " ".join(clean_sql_lines)
    if not sql_query.endswith(";"):
        sql_query += ";"
    return sql_query

def main():
    args = parse_args()
    model, tokenizer = load_model(args)
    # Load input JSON
    with open(args.input_json, "r") as f:
        data = json.load(f)
    output = []
    for entry in data:
        question = entry.get("question", "")
        knowledge = entry.get("expert_knowledge", "")
        relevant_schema = entry.get("relevant_schema", "")
        confidence = float(entry.get("confidence", 0.0))
        # Use relevant schema if present and confidence is high enough
        if relevant_schema and confidence >= args.confidence_threshold:
            schema_to_use = relevant_schema
        else:
            schema_to_use = SCHEMA
        print(f"Generating SQL for: {question} (confidence: {confidence:.2f})")
        sql_query = generate_sql(schema_to_use, question, knowledge, model, tokenizer, args)
        entry["generated_sql"] = sql_query
        output.append(entry)
    # Write output JSON
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {args.output_json}")

if __name__ == "__main__":
    main()
