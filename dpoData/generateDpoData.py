import sqlite3
import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Base path to database directory
BASE_DB_PATH = os.path.expanduser("~/Knowledge-to-SQL/model/data/train_databases")

def get_database_path(db_name):
    """Get the path to the specific database file"""
    return os.path.join(BASE_DB_PATH, db_name, f"{db_name}.sqlite")

def execute_sql(database_path, query):
    """Execute SQL query and return results with error handling"""
    try:
        conn = sqlite3.connect(database_path)
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result, None
    except Exception as e:
        return None, str(e)

def compare_results(gold_df, gen_df, gold_error, gen_error):
    """Compare result sets and determine preference"""
    # Handle error cases first
    if gold_error and not gen_error:
        return "generated"
    if gen_error and not gold_error:
        return "gold"
    if gold_error and gen_error:
        # Both have errors, prefer the one with more informative error message
        return "gold" if len(gold_error) <= len(gen_error) else "generated"
    
    # Handle empty result cases
    if gold_df is None or gen_df is None:
        return "gold"
    if gold_df.empty and gen_df.empty:
        return "equal"
        
    # Check schema differences
    if set(gold_df.columns) != set(gen_df.columns):
        return "gold"
    
    # Row count comparison
    if gold_df.shape[0] != gen_df.shape[0]:
        return "gold"
    
    # Content comparison with tolerance for sort differences
    try:
        # Sort both dataframes to ignore order differences
        gold_sorted = gold_df.sort_values(by=gold_df.columns.tolist()).reset_index(drop=True)
        gen_sorted = gen_df.sort_values(by=gen_df.columns.tolist()).reset_index(drop=True)
        
        if gold_sorted.equals(gen_sorted):
            return "equal"
        
        # Check for partial matches
        merged = gold_sorted.merge(gen_sorted, how='outer', indicator=True)
        different_rows = merged[merged['_merge'] != 'both'].shape[0]
        
        # If less than 20% different, consider generated close enough but still prefer gold
        if different_rows / len(merged) < 0.2:
            return "close_match"
            
        return "gold"
    except Exception:
        # If comparison fails, default to gold
        return "gold"

def extract_db_name(example):
    """Extract database name from the example"""
    # Check if db_id is directly provided
    if "db_id" in example:
        return example["db_id"]
    
    # Try to extract from schema
    schema = example.get("schema", "")
    first_line = schema.split('\n')[0].lower() if schema else ""
    
    # Check database names for a match in schema
    for db_name in os.listdir(BASE_DB_PATH):
        if db_name.lower() in first_line:
            return db_name
            
    # Default to first database with a .sqlite file
    for db_name in os.listdir(BASE_DB_PATH):
        if os.path.isdir(os.path.join(BASE_DB_PATH, db_name)):
            if os.path.exists(os.path.join(BASE_DB_PATH, db_name, f"{db_name}.sqlite")):
                return db_name
    
    return None

def process_example(example):
    """Process a single example to create DPO training pair"""
    db_name = extract_db_name(example)
    if not db_name:
        return None, "Could not determine database"
        
    database_path = get_database_path(db_name)
    if not os.path.exists(database_path):
        return None, f"Database file not found: {database_path}"
    
    # Execute both SQL queries
    gold_data, gold_err = execute_sql(database_path, example["gold_sql"])
    gen_data, gen_err = execute_sql(database_path, example["generated_sql"])
    
    # Build prompt
    prompt = (f"Schema: {example['schema']}\n"
             f"Question: {example['question']}\n"
             f"Knowledge: {example.get('generated_knowledge', '')}")
    
    # Determine preference
    result_comparison = compare_results(gold_data, gen_data, gold_err, gen_err)
    
    if result_comparison == "equal":
        return None, "Equal results"
    
    # Create preference pair based on comparison
    if result_comparison in ["gold", "close_match"]:
        return {
            "prompt": prompt,
            "chosen": example["gold_sql"],
            "rejected": example["generated_sql"],
        }, None
    else:  # "generated"
        return {
            "prompt": prompt,
            "chosen": example["generated_sql"],
            "rejected": example["gold_sql"],
        }, None

def create_dpo_dataset(sql_results, output_prefix, partial_save_every=50, max_workers=4):
    """Generate DPO dataset with parallel processing and partial saving"""
    dpo_data = []
    stats = {"processed": 0, "errors": 0, "equal": 0, "gold_preferred": 0, "gen_preferred": 0}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_example, ex): i for i, ex in enumerate(sql_results)}
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Generating DPO pairs")):
            idx = futures[future]
            stats["processed"] += 1
            
            try:
                result, error = future.result()
                if error:
                    if error == "Equal results":
                        stats["equal"] += 1
                    else:
                        stats["errors"] += 1
                        print(f"Error processing example {idx}: {error}")
                        
                if result:
                    dpo_data.append(result)
                    if result["chosen"] == sql_results[idx]["gold_sql"]:
                        stats["gold_preferred"] += 1
                    else:
                        stats["gen_preferred"] += 1
            except Exception as e:
                stats["errors"] += 1
                print(f"Exception on example {idx}: {str(e)}")

            # Save partial results every N processed queries
            if (i + 1) % partial_save_every == 0:
                partial_file = f"{output_prefix}_partial_{i+1}.jsonl"
                with open(partial_file, 'w') as f:
                    for example in dpo_data:
                        f.write(json.dumps(example) + '\n')
                print(f"Partial results saved after {i+1} queries to {partial_file}")
    
    print(f"\nProcessing summary:")
    print(f"Total examples: {len(sql_results)}")
    print(f"Successfully processed: {stats['gold_preferred'] + stats['gen_preferred']}")
    print(f"Gold preferred: {stats['gold_preferred']}")
    print(f"Generated preferred: {stats['gen_preferred']}")
    print(f"Equal results (skipped): {stats['equal']}")
    print(f"Errors: {stats['errors']}")
    
    return dpo_data

def parse_args():
    parser = argparse.ArgumentParser(description="Generate DPO training data from SQL execution results")
    parser.add_argument("--input_path", type=str, required=True, 
                        help="./dpo_sql_generation_results_combined.json")
    parser.add_argument("--output_path", type=str, required=True,
                        help="./dpo_data/")
    parser.add_argument("--partial_save_every", type=int, default=50,
                        help="Save partial results every N examples")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Maximum number of worker threads")
    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    args = parse_args()
    
    # Configuration
    INPUT_JSON = args.input_path
    OUTPUT_FILE = args.output_path
    
    # Load SQL generation results
    print(f"Loading SQL results from {INPUT_JSON}...")
    with open(INPUT_JSON) as f:
        sql_results = json.load(f)
    
    # Generate DPO dataset
    dpo_dataset = create_dpo_dataset(
        sql_results, 
        output_prefix=os.path.splitext(OUTPUT_FILE)[0],
        partial_save_every=args.partial_save_every,
        max_workers=args.max_workers
    )
    
    # Save to JSONL format for DPO training
    with open(OUTPUT_FILE, 'w') as f:
        for example in dpo_dataset:
            f.write(json.dumps(example) + '\n')
    
    print(f"DPO training data saved to {OUTPUT_FILE} with {len(dpo_dataset)} preference pairs")
