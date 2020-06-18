python -m preprocess.data_process --bert --data_path data/train_original.json --table_path data/tables.json --output data/train_sl_no_rule.json
#python -m preprocess.data_process --bert --data_path data/dev_original.json --table_path data/tables.json --output data/dev_sl_no_rule.json

# add 'rule_label'
python -m preprocess.sql2SemQL --data_path data/train_sl_no_rule.json --table_path data/tables.json --output data/train_sl.json
#python -m preprocess.sql2SemQL --data_path data/dev_sl_no_rule.json --table_path data/tables.json --output data/dev_sl.json
