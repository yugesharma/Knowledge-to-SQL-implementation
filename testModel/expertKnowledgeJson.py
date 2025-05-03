import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rapidfuzz import fuzz, process

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON file with questions and SQL")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file")
    parser.add_argument("--model_path", type=str, default="../output/llama2-13b-dpo")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=512)
    return parser.parse_args()

def load_model(args):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Loading base model...")
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

def extract_tables_and_columns(schema):
    tables = {}
    table_regex = re.compile(r'CREATE TABLE (\w+)\s*\((.*?)\);', re.DOTALL)
    for match in table_regex.finditer(schema):
        table_name = match.group(1)
        columns_block = match.group(2)
        column_lines = [line.strip() for line in columns_block.split(',') if line.strip() and 'FOREIGN KEY' not in line and 'PRIMARY KEY' not in line]
        columns = []
        for col_line in column_lines:
            col_match = re.match(r'(\w+)\s', col_line)
            if col_match:
                columns.append(col_match.group(1))
        tables[table_name] = columns
    return tables

SYNONYM_MAP = {
    # General Entities
    "patient": "subject_id",
    "patients": "subject_id",
    "patient id": "subject_id",
    "patient identifier": "subject_id",
    "person": "subject_id",
    "people": "subject_id",

    # Admissions
    "admission": "admissions",
    "admissions": "admissions",
    "admission id": "hadm_id",
    "admit id": "hadm_id",
    "admit time": "admittime",
    "admission time": "admittime",
    "admission date": "admittime",
    "discharge time": "dischtime",
    "discharge date": "dischtime",
    "admission type": "admission_type",
    "admission location": "admission_location",
    "discharge location": "discharge_location",

    # Demographics
    "gender": "gender",
    "sex": "gender",
    "female": "gender",
    "male": "gender",
    "woman": "gender",
    "man": "gender",
    "birth date": "dob",
    "date of birth": "dob",
    "dob": "dob",
    "born": "dob",
    "age": "age",
    "date of death": "dod",
    "dod": "dod",
    "died": "dod",
    "died in hospital": "dod",

    # Insurance & Status
    "insurance": "insurance",
    "medicare": "insurance",
    "marital status": "marital_status",
    "ethnicity": "ethnicity",
    "language": "language",

    # Diagnoses
    "diagnosis": "diagnoses_icd",
    "diagnoses": "diagnoses_icd",
    "diagnosis code": "icd9_code",
    "icd code": "icd9_code",
    "icd9": "icd9_code",
    "diagnosis name": "long_title",
    "diagnosis title": "short_title",
    "diagnosis date": "charttime",
    "diagnosis time": "charttime",
    "diagnosis id": "row_id",

    # Procedures
    "procedure": "procedures_icd",
    "procedures": "procedures_icd",
    "procedure code": "icd9_code",
    "procedure name": "long_title",
    "procedure title": "short_title",
    "procedure date": "charttime",
    "procedure time": "charttime",
    "appendectomy": "short_title",

    # Labs
    "lab": "labevents",
    "labs": "labevents",
    "lab event": "labevents",
    "lab result": "labevents",
    "lab value": "valuenum",
    "lab test": "label",
    "test": "label",
    "test type": "label",
    "glucose": "label",
    "test result": "valuenum",
    "result": "valuenum",
    "unit": "valueuom",

    # Prescriptions/Medications
    "prescription": "prescriptions",
    "prescriptions": "prescriptions",
    "medication": "prescriptions",
    "medications": "prescriptions",
    "drug": "drug",
    "dose": "dose_val_rx",
    "dose value": "dose_val_rx",
    "dose unit": "dose_unit_rx",
    "route": "route",
    "start date": "startdate",
    "end date": "enddate",

    # Costs
    "cost": "cost",
    "total cost": "cost",
    "medication event": "event_type",
    "event type": "event_type",
    "event id": "event_id",
    "charge time": "chargetime",

    # ICU/Stay/Transfers
    "icu": "icustays",
    "icu stay": "icustays",
    "length of stay": "outtime",
    "stay": "icustays",
    "intensive care": "icustays",
    "icu admission": "intime",
    "icu discharge": "outtime",
    "first care unit": "first_careunit",
    "last care unit": "last_careunit",
    "first ward": "first_wardid",
    "last ward": "last_wardid",
    "intime": "intime",
    "outtime": "outtime",

    # Chart/Events
    "chart event": "chartevents",
    "chart time": "charttime",
    "chart value": "valuenum",

    # Input/Output Events
    "input event": "inputevents_cv",
    "output event": "outputevents",
    "amount": "amount",
    "value": "value",

    # Microbiology
    "microbiology": "microbiologyevents",
    "specimen": "spec_type_desc",
    "organism": "org_name",

    # Transfers
    "transfer": "transfers",
    "event type": "eventtype",
    "care unit": "careunit",
    "ward": "wardid",

    # Misc
    "row id": "row_id",
    "id": "row_id"
}

def expand_question_with_synonyms(question):
    # Replace synonyms/phrases in the question with schema terms
    q = question.lower()
    for k, v in SYNONYM_MAP.items():
        q = q.replace(k, v)
    return q

def select_relevant_tables_and_columns(question, tables, top_k=5):
    question_expanded = expand_question_with_synonyms(question)
    question_tokens = question_expanded.lower().split()
    results = []
    # 1. Fuzzy match columns
    for table, columns in tables.items():
        for column in columns:
            # Fuzzy match each question token to column name
            best_score = max(fuzz.partial_ratio(column.lower(), qt) for qt in question_tokens)
            results.append((table, column, best_score))
    # 2. Keep top_k matches above threshold
    results.sort(key=lambda x: x[2], reverse=True)
    selected = [r for r in results if r[2] > 60][:top_k]
    relevant = {}
    confidences = []
    for table, column, score in selected:
        relevant.setdefault(table, []).append(column)
        confidences.append(score)
    # 3. Fallback: if nothing matched, try table-level match
    if not relevant:
        for table in tables:
            if any(fuzz.partial_ratio(table.lower(), qt) > 80 for qt in question_tokens):
                relevant[table] = tables[table]
                confidences.append(70)
                break
    confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
    return relevant, confidence

def build_reduced_schema(schema, relevant):
    tables_in_schema = extract_tables_and_columns(schema)
    reduced_schema = []
    for table, columns in relevant.items():
        if table in tables_in_schema:
            table_regex = re.compile(r'(CREATE TABLE {}[^\(]*\((.*?)\);)'.format(table), re.DOTALL)
            match = table_regex.search(schema)
            if match:
                all_columns_block = match.group(2)
                lines = all_columns_block.split(',')
                kept_lines = [line for line in lines if any(col in line for col in columns)]
                reduced_table = f"CREATE TABLE {table} (\n" + ",\n".join(kept_lines) + "\n);"
                reduced_schema.append(reduced_table)
    return "\n\n".join(reduced_schema)

def generate_knowledge(question, schema, model, tokenizer, max_new_tokens=512, temperature=0.7):
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
    knowledge = knowledge.split("Knowledge:")[-1].strip()
    return knowledge

def main():
    args = parse_args()
    # Load model and tokenizer
    model, tokenizer = load_model(args)
    # Load input JSON
    with open(args.input_json, "r") as f:
        data = json.load(f)
    tables = extract_tables_and_columns(SCHEMA)
    output = []
    for entry in data:
        question = entry["question"]
        # 1. Extract relevant tables/columns and confidence
        relevant, confidence = select_relevant_tables_and_columns(question, tables, top_k=5)
        # 2. Build reduced schema
        reduced_schema = build_reduced_schema(SCHEMA, relevant)
        # 3. Generate expert knowledge
        expert_knowledge = generate_knowledge(
            question,
            reduced_schema,
            model,
            tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        output.append({
            "category": entry["category"],
            "question": entry["question"],
            "sql": entry["sql"],
            "relevant_schema": reduced_schema,
            "expert_knowledge": expert_knowledge,
            "confidence": confidence
        })
        print(f"Processed: {question} | Confidence: {confidence:.2f}")
    # Write output JSON
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {args.output_json}")

if __name__ == "__main__":
    main()
