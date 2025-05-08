# Knowledge-to-SQL implementation


## Setup

### Environment

```
git clone https://github.com/yugesharma/Knowledge-to-SQL-implementation.git
conda create -n dellm python=3.11.3
conda init
Conda activate dellm
```

### Dataset
We trained our model on BIRD dataset. Download BIRD dataset from [here](https://bird-bench.github.io/). Extract the zip files and save them in the Knowledge-to-SQL-implementation directory. Next we ran dataset/preprocessor.py.

```
python dataset/preprocessor.py --path/to/train.json --db_root_path path/to/train_databases/ --output_path bird/processed_train.json
```

### Other dependencies
We used Llama2 13b model. Log in to HuggingFace and request for access. After obtaining access token, login using the command below and follow instructions:

```
huggingface-cli login
```

Obtain access token for Weight and Balance from [here](https://wandb.ai/settings).

## Training

### SFT
Make required changes to the sbatch file **run_llama2_sft.sh** according to your paths and system. You can also modify the training arguments as per your needs.
Submit the sbatch file for training.

### DPO
The directory _dpoDataPrep_ contain all dependencies for DPO training.
- generateKowledge.py: Generates expert knowledge json file
- generateSql.py: Generates pairs of SQL queries using both expert knowledge and gold knowledge
- generateDpoData.py: For generating preference pairs
- TokenizeForDpoTraining: Final step of data tokenization for DPO training
- dpo_training.sh: Bash file to submit for DPO training

Keep checking the .out file for status and *squeue --me* for status check.

## Output

Once successfully completed, a directory **output** will get generated which will have all the files generated from the model training including the Lora weights for fine tuning.

## Evaluation
The directory _testModel_ contains various versions of files used for testing evaluating the model. The final tests and evaluations are based on natural language questions pertaining to MIMIC IV medical dataset. 
- mimic4dellm.py: Outputs a json file containing expert knowledge for natural language questions
- mimic4query.py: Generates a json file containing the generated SQL queries along with VES scores
