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
CREATE TABLE patients
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL UNIQUE,
    gender VARCHAR(5) NOT NULL,
    dob TIMESTAMP(0) NOT NULL,
    dod TIMESTAMP(0)
);

DROP TABLE IF EXISTS admissions;
CREATE TABLE admissions
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL UNIQUE,
    admittime TIMESTAMP(0) NOT NULL,
    dischtime TIMESTAMP(0),
    admission_type VARCHAR(50) NOT NULL,
    admission_location VARCHAR(50) NOT NULL,
    discharge_location VARCHAR(50),
    insurance VARCHAR(255) NOT NULL,
    language VARCHAR(10),
    marital_status VARCHAR(50),
    age INT NOT NULL,
    FOREIGN KEY(subject_id) REFERENCES patients(subject_id)
);

DROP TABLE IF EXISTS d_icd_diagnoses;
CREATE TABLE d_icd_diagnoses
(
    row_id INT NOT NULL PRIMARY KEY,
    icd_code VARCHAR(10) NOT NULL UNIQUE,
    long_title VARCHAR(255) NOT NULL
);

DROP TABLE IF EXISTS d_icd_procedures;
CREATE TABLE d_icd_procedures 
(
    row_id INT NOT NULL PRIMARY KEY,
    icd_code VARCHAR(10) NOT NULL UNIQUE,
    long_title VARCHAR(255) NOT NULL
);

DROP TABLE IF EXISTS d_labitems;
CREATE TABLE d_labitems 
(
    row_id INT NOT NULL PRIMARY KEY,
    itemid INT NOT NULL UNIQUE,
    label VARCHAR(200)
);

DROP TABLE IF EXISTS d_items;
CREATE TABLE d_items 
(
    row_id INT NOT NULL PRIMARY KEY,
    itemid INT NOT NULL UNIQUE,
    label VARCHAR(200) NOT NULL,
    abbreviation VARCHAR(200) NOT NULL,
    linksto VARCHAR(50) NOT NULL
);

DROP TABLE IF EXISTS diagnoses_icd;
CREATE TABLE diagnoses_icd
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    icd_code VARCHAR(10) NOT NULL,
    charttime TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(icd_code) REFERENCES d_icd_diagnoses(icd_code)
);

DROP TABLE IF EXISTS procedures_icd;
CREATE TABLE procedures_icd
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    icd_code VARCHAR(10) NOT NULL,
    charttime TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(icd_code) REFERENCES d_icd_procedures(icd_code)
);

DROP TABLE IF EXISTS labevents;
CREATE TABLE labevents
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    itemid INT NOT NULL,
    charttime TIMESTAMP(0),
    valuenum DOUBLE PRECISION,
    valueuom VARCHAR(20),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(itemid) REFERENCES d_labitems(itemid)
);

DROP TABLE IF EXISTS prescriptions;
CREATE TABLE prescriptions
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    starttime TIMESTAMP(0) NOT NULL,
    stoptime TIMESTAMP(0),
    drug VARCHAR(255) NOT NULL,
    dose_val_rx VARCHAR(100) NOT NULL,
    dose_unit_rx VARCHAR(50) NOT NULL,
    route VARCHAR(50) NOT NULL,
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id)
);

DROP TABLE IF EXISTS cost;
CREATE TABLE cost
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    event_type VARCHAR(20) NOT NULL,
    event_id INT NOT NULL,
    chargetime TIMESTAMP(0) NOT NULL,
    cost DOUBLE PRECISION NOT NULL,
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(event_id) REFERENCES diagnoses_icd(row_id),
    FOREIGN KEY(event_id) REFERENCES procedures_icd(row_id),
    FOREIGN KEY(event_id) REFERENCES labevents(row_id),
    FOREIGN KEY(event_id) REFERENCES prescriptions(row_id)  
);

DROP TABLE IF EXISTS chartevents;
CREATE TABLE chartevents
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    stay_id INT NOT NULL,
    itemid INT NOT NULL,
    charttime TIMESTAMP(0) NOT NULL,
    valuenum DOUBLE PRECISION,
    valueuom VARCHAR(50),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(stay_id) REFERENCES icustays(stay_id),
    FOREIGN KEY(itemid) REFERENCES d_items(itemid)
);

DROP TABLE IF EXISTS inputevents;
CREATE TABLE inputevents
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    stay_id INT NOT NULL,
    starttime TIMESTAMP(0) NOT NULL,
    itemid INT NOT NULL,
    totalamount DOUBLE PRECISION,
    totalamountuom VARCHAR(50),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(stay_id) REFERENCES icustays(stay_id),
    FOREIGN KEY(itemid) REFERENCES d_items(itemid)
);

DROP TABLE IF EXISTS outputevents;
CREATE TABLE outputevents
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    stay_id INT NOT NULL,
    charttime TIMESTAMP(0) NOT NULL,
    itemid INT NOT NULL,
    value DOUBLE PRECISION,
    valueuom VARCHAR(50),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(stay_id) REFERENCES icustays(stay_id),
    FOREIGN KEY(itemid) REFERENCES d_items(itemid)
);

DROP TABLE IF EXISTS microbiologyevents;
CREATE TABLE microbiologyevents
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    charttime TIMESTAMP(0) NOT NULL,
    spec_type_desc VARCHAR(100),
    test_name VARCHAR(100),
    org_name VARCHAR(100),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id)
);

DROP TABLE IF EXISTS icustays;
CREATE TABLE icustays
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    stay_id INT NOT NULL UNIQUE,
    first_careunit VARCHAR(20) NOT NULL,
    last_careunit VARCHAR(20) NOT NULL,
    intime TIMESTAMP(0) NOT NULL,
    outtime TIMESTAMP(0),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id)
);

DROP TABLE IF EXISTS transfers;
CREATE TABLE transfers
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    transfer_id INT NOT NULL,
    eventtype VARCHAR(20) NOT NULL,
    careunit VARCHAR(20),
    intime TIMESTAMP(0) NOT NULL,
    outtime TIMESTAMP(0),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id)
);"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON file with questions and SQL")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file")
    parser.add_argument("--model_path", type=str, default="../output/llama2-13b-dpo")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-13b-hf")
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
    # ===================== PATIENTS =====================
    "patient": "subject_id",
    "patients": "subject_id",
    "patient id": "subject_id",
    "patient identifier": "subject_id",
    "person": "subject_id",
    "people": "subject_id",
    "subject id": "subject_id",
    "subject": "subject_id",
    "row id": "row_id",
    "row": "row_id",
    "gender": "gender",
    "sex": "gender",
    "male": "gender",
    "female": "gender",
    "man": "gender",
    "woman": "gender",
    "date of birth": "dob",
    "dob": "dob",
    "birthdate": "dob",
    "birth date": "dob",
    "born": "dob",
    "birthday": "dob",
    "age": "age",
    "date of death": "dod",
    "dod": "dod",
    "death date": "dod",
    "deceased": "dod",
    "died": "dod",
    "dead": "dod",
    "passed away": "dod",
    "expired": "dod",

    # ===================== ADMISSIONS =====================
    "admission": "hadm_id",
    "admissions": "hadm_id",
    "admission id": "hadm_id",
    "hadm id": "hadm_id",
    "hospital admission id": "hadm_id",
    "admit id": "hadm_id",
    "admit time": "admittime",
    "admission time": "admittime",
    "admission date": "admittime",
    "admission datetime": "admittime",
    "admittime": "admittime",
    "discharge time": "dischtime",
    "discharge date": "dischtime",
    "discharge datetime": "dischtime",
    "dischtime": "dischtime",
    "admission type": "admission_type",
    "type of admission": "admission_type",
    "admission category": "admission_type",
    "admission location": "admission_location",
    "admitted from": "admission_location",
    "admission source": "admission_location",
    "discharge location": "discharge_location",
    "discharged to": "discharge_location",
    "insurance": "insurance",
    "payer": "insurance",
    "coverage": "insurance",
    "language": "language",
    "spoken language": "language",
    "marital status": "marital_status",
    "marital": "marital_status",
    "married": "marital_status",
    "single": "marital_status",
    "widowed": "marital_status",
    "divorced": "marital_status",

    # ===================== ICD DIAGNOSES =====================
    "diagnosis": "icd_code",
    "diagnoses": "icd_code",
    "diagnosis code": "icd_code",
    "diagnoses code": "icd_code",
    "icd code": "icd_code",
    "icd10": "icd_code",
    "icd9": "icd_code",
    "diagnosis id": "row_id",
    "diagnosis name": "long_title",
    "diagnosis title": "long_title",
    "diagnosis description": "long_title",
    "diagnosis label": "long_title",
    "diagnosis long title": "long_title",
    "diagnosis date": "charttime",
    "diagnosis time": "charttime",
    "diagnosed at": "charttime",
    "diagnosed on": "charttime",

    # ===================== ICD PROCEDURES =====================
    "procedure": "icd_code",
    "procedures": "procedures_icd",
    "procedure code": "icd_code",
    "procedure id": "row_id",
    "procedure name": "long_title",
    "procedure title": "long_title",
    "procedure description": "long_title",
    "procedure label": "long_title",
    "procedure long title": "long_title",
    "procedure date": "charttime",
    "procedure time": "charttime",
    "performed at": "charttime",
    "performed on": "charttime",

    # ===================== LAB ITEMS / LABEVENTS =====================
    "lab": "itemid",
    "labs": "labevent",
    "lab event": "labevents",
    "lab events": "labevents",
    "lab result": "valuenum",
    "lab results": "valuenum",
    "lab value": "valuenum",
    "lab test": "label",
    "lab test name": "label",
    "test": "label",
    "test name": "label",
    "test type": "label",
    "test result": "valuenum",
    "result": "valuenum",
    "value": "valuenum",
    "measurement": "valuenum",
    "unit": "valueuom",
    "units": "valueuom",
    "result unit": "valueuom",
    "reference range": "valueuom",
    "lab id": "itemid",
    "item id": "itemid",
    "itemid": "itemid",
    "lab time": "charttime",
    "lab date": "charttime",
    "lab datetime": "charttime",
    "lab timestamp": "charttime",

    # ===================== PRESCRIPTIONS =====================
    "prescription": "drug",
    "prescriptions": "drug",
    "medication": "drug",
    "medications": "drug",
    "med": "drug",
    "drug": "drug",
    "medicine": "drug",
    "med name": "drug",
    "medication name": "drug",
    "dose": "dose_val_rx",
    "dosage": "dose_val_rx",
    "dose value": "dose_val_rx",
    "dose amount": "dose_val_rx",
    "dose unit": "dose_unit_rx",
    "dosage unit": "dose_unit_rx",
    "route": "route",
    "administration route": "route",
    "start date": "starttime",
    "start time": "starttime",
    "start datetime": "starttime",
    "stop date": "stoptime",
    "stop time": "stoptime",
    "end date": "stoptime",
    "end time": "stoptime",

    # ===================== COST =====================
    "cost": "cost",
    "total cost": "cost",
    "charge": "cost",
    "charges": "cost",
    "event type": "event_type",
    "event": "event_type",
    "event id": "event_id",
    "charge time": "chargetime",
    "charge date": "chargetime",
    "charge datetime": "chargetime",

    # ===================== ICU STAYS =====================
    "icu": "icustays",
    "icu stay": "icustays",
    "icu stays": "icustays",
    "intensive care": "icustays",
    "intensive care unit": "icustays",
    "icu admission": "intime",
    "icu admit": "intime",
    "icu entry": "intime",
    "icu discharge": "outtime",
    "icu exit": "outtime",
    "stay id": "stay_id",
    "first careunit": "first_careunit",
    "last careunit": "last_careunit",
    "first unit": "first_careunit",
    "last unit": "last_careunit",
    "in time": "intime",
    "out time": "outtime",
    "length of stay": "outtime",
    "los": "outtime",

    # ===================== CHART EVENTS =====================
    "chart event": "chartevents",
    "chart events": "chartevents",
    "chart time": "charttime",
    "chart date": "charttime",
    "chart datetime": "charttime",
    "chart value": "valuenum",
    "chart measurement": "valuenum",
    "chart item": "itemid",

    # ===================== INPUT/OUTPUT EVENTS =====================
    "input event": "inputevents",
    "input events": "inputevents",
    "output event": "outputevents",
    "output events": "outputevents",
    "input": "inputevents",
    "output": "outputevents",
    "total amount": "totalamount",
    "amount": "totalamount",
    "amount unit": "totalamountuom",
    "input amount": "totalamount",
    "output value": "value",
    "output amount": "value",
    "output unit": "valueuom",
    "input unit": "totalamountuom",

    # ===================== MICROBIOLOGY EVENTS =====================
    "microbiology": "microbiologyevents",
    "microbiology event": "microbiologyevents",
    "microbiology events": "microbiologyevents",
    "microbio": "microbiologyevents",
    "specimen": "spec_type_desc",
    "specimen type": "spec_type_desc",
    "sample type": "spec_type_desc",
    "test name": "test_name",
    "organism": "org_name",
    "organism name": "org_name",

    # ===================== TRANSFERS =====================
    "transfer": "transfers",
    "transfers": "transfers",
    "transfer id": "transfer_id",
    "event type": "eventtype",
    "transfer type": "eventtype",
    "care unit": "careunit",
    "unit": "careunit",
    "ward": "careunit",
    "ward id": "careunit",
    "transfer intime": "intime",
    "transfer outtime": "outtime",

    # ===================== GENERAL TIME PHRASES =====================
    "date": "charttime",
    "time": "charttime",
    "datetime": "charttime",
    "timestamp": "charttime",
    "when": "charttime",
    "at": "charttime",
    "on": "charttime",
    "during": "charttime",
    "after": "charttime",
    "before": "charttime",
    "since": "charttime",
    "until": "charttime",

    # ===================== MISC =====================
    "id": "row_id",
    "identifier": "row_id",
    "record": "row_id",
    "record id": "row_id"
}


def expand_question_with_synonyms(question):
    # Replace synonyms/phrases in the question with schema terms
    q = question.lower()
    for k, v in SYNONYM_MAP.items():
        q = q.replace(k, v)
        print(q)
    return q

def select_relevant_tables_and_columns(question, tables, top_k=7):
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
    fallback_used=False
    if not relevant:
        for table in tables:
            if any(fuzz.partial_ratio(table.lower(), qt) > 80 for qt in question_tokens):
                relevant[table] = tables[table]
                fallback_used=True
                break
    if fallback_used:
        confidence = 0.0
    elif confidences:
        confidence = float(sum(confidences) / len(confidences))
    else:
        confidence = 0.0

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
    prompt = f"""Given the following database schema and question, generate the knowledge needed to write an accurate executable SQL query:

Schema: {schema}
Question: {question}

Consider ICD code mappings, Lab test identifiers, time calculations, medication routes
Use tables and columns only available in the schema provided
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
#        relevant, confidence = select_relevant_tables_and_columns(question, tables, top_k=5)
        # 2. Build reduced schema
 #       print(relevant)
  #      reduced_schema = build_reduced_schema(SCHEMA, relevant)
        # 3. Generate expert knowledge
        expert_knowledge = generate_knowledge(
            question,
            tables,
            model,
            tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        output.append({
            "category": entry["category"],
            "question": entry["question"],
            "sql_query": entry["sql_query"],
           # "relevant_schema": reduced_schema,
            "expert_knowledge": expert_knowledge,
           # "confidence": confidence
        })
        #print(f"Processed: {question} | Confidence: {confidence:.2f}")
    # Write output JSON
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {args.output_json}")

if __name__ == "__main__":
    main()
