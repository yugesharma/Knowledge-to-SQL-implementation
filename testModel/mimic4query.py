import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import os
import sqlite3
import time
import re

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

def apply_synonym_map(question, synonym_map):
    """Replace synonyms in the question with canonical schema terms."""
    q = question.lower()
    for syn in sorted(synonym_map, key=len, reverse=True):
        pattern = r'\b' + re.escape(syn) + r'\b'
        q = re.sub(pattern, synonym_map[syn], q)
    return q

SCHEMA = """DROP TABLE IF EXISTS patients;
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
    parser = argparse.ArgumentParser(description="Generate SQL queries, execute, and calculate VES")
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file")
    parser.add_argument("--sqlite_db", type=str, required=True, help="SQLite DB file")
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
    promptTemplate = """You are an expert SQL assistant.

You must generate syntactically correct and executable SQLite SQL queries for the given questions, using only the provided database schema.
- Do NOT invent columns or tables; use only what is present in the schema.
- For date and time operations, use SQLite syntax (e.g., strftime('%Y', column) for extracting the year).
- Use only lowercase values for gender ('f', 'm').
- If a question asks for a count, use COUNT(*).
- If a question asks for averages or sums, use AVG() or SUM().
- For JOINs, use the correct foreign key relationships as defined in the schema.
- If a question asks for a filter (e.g., "female patients"), use a WHERE clause.
- If the question asks for patients above/below a certain age, calculate age using the date of birth and CURRENT_DATE.
- If a field is not present in the schema, explain in a comment.
- Always end your SQL with a semicolon.

Database schema:
{schema}
And the following relevant knowledge:
{knowledge}

Given question:
{question}

SQL:
"""
    prompt=promptTemplate.format(schema=schema.strip(), knowledge=knowledge.strip(), question=question.strip())
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
    # Connect to SQLite database
    conn = sqlite3.connect(args.sqlite_db)
    cursor = conn.cursor()
    print(f"Connected to SQLite database: {args.sqlite_db}")
    # Load input JSON
    with open(args.input_json, "r") as f:
        data = json.load(f)
    results_dict = []
    total_ves = 0
    valid_ves_count = 0
    for i, entry in enumerate(data):
        question = entry.get("question", "")
        # Preprocess question with synonym map
        normalized_question = apply_synonym_map(question, SYNONYM_MAP)
        knowledge = entry.get("expert_knowledge", "")
        relevant_schema = entry.get("relevant_schema", "")
        confidence = float(entry.get("confidence", 0.0))
        if relevant_schema and confidence >= args.confidence_threshold:
            schema_to_use = relevant_schema
        else:
            schema_to_use = SCHEMA
        print(f"\nQuery {i+1}/{len(data)}: {question} (confidence: {confidence:.2f})")
        print(f"Normalized: {normalized_question}")
        # Generate SQL
        generated_sql = generate_sql(SCHEMA, question, knowledge, model, tokenizer, args)
        entry["generated_sql"] = generated_sql
        # Prepare for VES calculation
        query_results = {"query_num": i+1, "question": question}
        original_sql = entry.get('sql_query')
        original_execution_time = None
        generated_execution_time = None
        is_valid = 0  # Default to invalid
        # Execute original SQL if present
        if original_sql:
            try:
                print(f"Original SQL: {original_sql}")
                start_time = time.perf_counter()
                cursor.execute(original_sql)
                results = cursor.fetchall()
                end_time = time.perf_counter()
                original_execution_time = end_time - start_time
                print(f"Original SQL execution time: {original_execution_time:.6f} seconds")
                query_results["original_sql"] = {
                    "sql": original_sql,
                    "execution_time": original_execution_time,
                    "results": results[:10]
                }
            except sqlite3.Error as e:
                print(f"Error executing original SQL: {e}")
                query_results["original_sql"] = {
                    "sql": original_sql,
                    "error": str(e)
                }
        # Execute generated SQL
        clean_sql = generated_sql.split('[')[0].split('*/')[0].strip()
        try:
            print(f"Generated SQL: {clean_sql}")
            start_time = time.perf_counter()
            cursor.execute(clean_sql)
            results = cursor.fetchall()
            end_time = time.perf_counter()
            generated_execution_time = end_time - start_time
            is_valid = 1
            print(f"Generated SQL execution time: {generated_execution_time:.6f} seconds")
            query_results["generated_sql"] = {
                "sql": clean_sql,
                "execution_time": generated_execution_time,
               # "results": results[:10],
                "is_valid": is_valid
            }
        except sqlite3.Error as e:
            print(f"Error executing generated SQL: {e}")
            query_results["generated_sql"] = {
                "sql": clean_sql,
                "error": str(e),
                "is_valid": 0
            }
        # Calculate VES if both queries executed
        if original_execution_time is not None and generated_execution_time is not None and is_valid == 1:
            ves = is_valid * max(0,1 - (generated_execution_time / original_execution_time))
            query_results["ves"] = ves
            print(f"VES Score: {ves:.6f}")
            total_ves += ves
            valid_ves_count += 1
        else:
            query_results["ves"] = 0 if is_valid == 0 else None
            print(f"VES Score: {query_results['ves']} (Query invalid or original query failed)")
        results_dict.append(query_results)
    # Summary statistics
    average_ves = total_ves / valid_ves_count if valid_ves_count > 0 else 0
    summary = {
        "total_queries": len(data),
        "valid_ves_count": valid_ves_count,
        "average_ves": average_ves
    }
    print("\n===== SUMMARY =====")
    print(f"Total queries: {summary['total_queries']}")
    print(f"Queries with valid VES: {summary['valid_ves_count']}")
    print(f"Average VES: {summary['average_ves']:.6f}")
    results_dict.append({"summary": summary})
    # Save results to output JSON
    with open(args.output_json, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_json}")
    conn.close()
    print("\nDatabase connection closed")

if __name__ == "__main__":
    main()

