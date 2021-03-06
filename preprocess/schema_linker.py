"""
This module loads BERT for question and schema entities (column, tables)
and links each question token to an entity.
"""

#import transformers
from transformers import PreTrainedTokenizer, BertTokenizer, \
        PreTrainedModel, BertModel, BertConfig
from nltk.corpus import stopwords
import torch
import json
from torch.nn import CosineSimilarity
from typing import Dict, List


# def cross_encode_all(question: str,
#                      db_wpieces: List[str]
#                      )

def cross_encode_pair(question: str,
                      schema_entity_description: str,
                      tokenizer: PreTrainedTokenizer,
                      model: PreTrainedModel,
                      second_to_last: bool = True) -> (torch.Tensor, torch.Tensor):
    """
    Encode question and schema entity with the same BERT model.
    Input sequence to model is [CLS] question [SEP] schema_entity [SEP],
    where schema_entity is a table or column annotation.

    Returns the BERT hidden output for each question token (question_len x 768),
    as well as the mean-pooled BERT hidden output for the schema entity.
    """
    # prepare input sequence for BERT: [CLS] question [SEP] table [SEP]
    encode_dict = tokenizer.encode_plus(text=question,
                                        text_pair=schema_entity_description, 
                                        add_special_tokens=True)

    # encode input sequence with BERT
    input_seq = torch.LongTensor([encode_dict['input_ids']])
    input_seq_type = torch.LongTensor([encode_dict['token_type_ids']])
    
    with torch.no_grad():
        # 1 x T x 768, (1 x T x 768, ...)
        last_hidden, _, all_hidden = model(input_ids=input_seq,
                                           token_type_ids=input_seq_type)

        # mean-pool hidden states of schema entity tokens
        # 1 x 768
        selected_hidden = all_hidden[-2] if second_to_last else all_hidden[-1]
        entity_output = torch.mean(selected_hidden[:, schema_tok_idx:-1,:],
                                   dim=1)

        # len(question) x 768
        question_output = selected_hidden[0, 1:schema_tok_idx-1,:]                           
        
    return (question_output, entity_output)

def bi_encode_pair(question: str,
                   schema_entity_description: str,
                   tokenizer: PreTrainedTokenizer,
                   model: PreTrainedModel,
                   second_to_last: bool = True) -> (torch.Tensor, torch.Tensor):
    """
    Encode question and schema entity separately with the same BERT model.
    The two input sequences are:
        [CLS] question [SEP] 
        and 
        [CLS] schema_entity [SEP],
    where schema_entity is a table or column annotation.

    Returns the BERT hidden output for each question token (question_len x 768),
    as well as the mean-pooled BERT hidden output for the schema entity.
    """
    question_dict = tokenizer.encode_plus(text=question,
                                          add_special_tokens=True)
    question_input = torch.LongTensor([question_dict['input_ids']])
    question_input_type = torch.LongTensor([question_dict['token_type_ids']])

    entity_dict = tokenizer.encode_plus(text=schema_entity_description, 
                                        add_special_tokens=True)
    entity_input = torch.LongTensor([entity_dict['input_ids']])
    entity_input_type = torch.LongTensor([entity_dict['token_type_ids']])

    # encode the two sequences in parallel
    with torch.no_grad():
        # 1 x T x 768, (1 x T x 768, ...)
        last_hidden, _, all_hidden = model(input_ids=question_input,
                                           token_type_ids=question_input_type)
        
        # trim [CLS] and [SEP]
        # len(question) x 768
        question_output = all_hidden[-2] if second_to_last else all_hidden[-1]
        question_output = question_output[0, 1:question_input.size(1)-1, :]

         # 1 x T x 768, (1 x T x 768, ...)
        last_hidden, _, all_hidden = model(input_ids=entity_input,
                                           token_type_ids=entity_input_type)

        # take [CLS] output
        # 1 x 768
        entity_output = all_hidden[-2] if second_to_last else all_hidden[-1]
        entity_output = entity_output[:, 0, :]
    
    return (question_output, entity_output)

def generate_db_table_annotations(table_dict: Dict) -> List[str]:
    """
    Given a database, generate table annotations with the following structure:
    TABLE_NAME $table_name COLUMN_NAMES $col_1 $type_col_1 $col_2 ...
    PRIMARY_KEY $col $type_col [FOREIGN_KEY $column_name $parent_table_name 
    $primary_key_of_parent_table]+
    Args:  
        table_dict: dictionary from tables.json containing tables info for a 
                    given database
    """
    table_annotations = []
    table_names = table_dict['table_names']
    column_list = table_dict['column_names']
    column_types = table_dict['column_types']
    # each table has one (or more) several columns as primary key:
    # {table_name: [column_1, column2_, ... ]}
    primary_dict = {}
    for primary_key_idx in table_dict['primary_keys']:
        primary_key_table_idx, primary_key_name = column_list[primary_key_idx]
        primary_key_table = table_names[primary_key_table_idx]
        if primary_key_table in primary_dict:
            primary_dict[primary_key_table].append(primary_key_name)
        else:
            primary_dict[primary_key_table] = [primary_key_name]

    # each table has one or more foreign keys
    foreign_dict = {} 
    for foreign_keys in table_dict['foreign_keys']:
        child_idx, parent_idx = foreign_keys[0], foreign_keys[1]
        child_table_idx, child_column = column_list[child_idx]
        parent_table_idx, parent_column = column_list[parent_idx]

        child_table = table_names[child_table_idx]
        parent_table = table_names[parent_table_idx]

        if child_table in foreign_dict:
            foreign_dict[child_table].append((child_column, 
                                              parent_table,
                                              parent_column))
        else:
            foreign_dict[child_table] = [(child_column, 
                                          parent_table,
                                          parent_column)]

    for idx, table_name in enumerate(table_names):
        annot_list = ['TABLE_NAME', table_name, 'COLUMN_NAMES']  
        # add columns and their types
        column_names = [e[1] for e in column_list if e[0] == idx]
        column_typs = [t for t, e in zip(column_types, column_list) if e[0] == idx]
        assert len(column_names) == len(column_typs), "uneven seq lengths"
        for col_name, col_type in zip(column_names, column_typs):
            annot_list.extend([col_name, col_type])

        # add primary key
        primary_keys = primary_dict.get(table_name)
        if primary_keys:
            annot_list.extend(['PRIMARY_KEY'] + primary_keys)
        else:
            annot_list.extend(['PRIMARY_KEY', "none"])

        # add foreign keys if it's the case
        # foreign key (index_of_child_column index_of_parent_column)
        foreign_keys = foreign_dict.get(table_name)
        if foreign_keys:
            for foreign_tuple in foreign_keys:
                annot_list.extend(['FOREIGN_KEY'] + list(foreign_tuple))
        else:
            annot_list.extend(['FOREIGN_KEY none'])

        table_annotations.append(' '.join(annot_list))

    return table_annotations

def generate_db_column_annotations(table_dict: Dict) -> List[str]:
    """
    Given a database, generate column annotations with the following structure:
    COLUMN_NAME $column_name $column_type TABLE_NAME $table_name ...
    PRIMARY_KEY/FOREIGN_KEY/OTHER
    Args:  
        table_dict: dictionary from tables.json containing tables info for a 
                    given database
    """
    primary_keys_idx = table_dict["primary_keys"]
    foreign_keys_idx = [e[0] for e in table_dict["foreign_keys"]]
    table_names = table_dict["table_names"]
    column_annotations = []
    for idx, column_tuple in enumerate(table_dict["column_names"]):
        if idx == 0:
            continue
        idx_in_tables, column_name = column_tuple[0], column_tuple[1]
        table_name = table_names[idx_in_tables]
        annot_list = ['COLUMN_NAME', column_name, 'TABLE_NAME', table_name]
        if idx in primary_keys_idx:
            annot_list.append('PRIMARY_KEY')
        elif idx in foreign_keys_idx:
            annot_list.append('FOREIGN_KEY')
        else:
            annot_list.append('OTHER')

        column_annotations.append(' '.join(annot_list))

    return column_annotations

def tokenize_db(table_dict: Dict,
                tokenizer: PreTrainedTokenizer) -> Dict:
    """
    Given dictionary with tables info, tokenize its following representation:
    "$TABLE_1 $COL_11. $COL_12 [SEP] ... $TABLE_N $COL_N1 $COL_N2 ... [SEP]"
    
    Return a dictionary with the following key/values:
        'entities': a list of database entities
        'wordpieces': a list of database wordpieces (from tokenizing entities)
        'entity_idx': lists of entity indices for each wordpiece of a db entity
        
    Example: 
        {'entities': ['schooling', 'school id', 'location', '[SEP]', ...],
         'wordpieces': ['school', '##ing', 'school', 'id', 'locat', '##ion", ...]
         'entity_idx': [[17, 17], [0, 0], [1, 1], [-1], [18], [2], [3], ...]
        }
    Table indices start from $number_of_columns onwards and correspond to the
    order in 'table_names' list. Column indices are in the same range as in 
    the 'column_names' list.
    """
    table_names = table_dict["table_names"]
    column_list = table_dict["column_names"]
    col_size = len(column_list) 

    entities = []
    wordpieces = []
    entity_idx = [] #index in 'column_names' or 'table_names' list
    for table_idx, table_name in enumerate(table_names):
        entities.append(table_name)
        table_toks = tokenizer.tokenize(table_name)
        wordpieces.extend(table_toks)
        entity_idx.append([col_size + table_idx] * len(table_toks))
        
        for col_idx, (col_table_idx, col_name) in enumerate(column_list):
            if col_table_idx == table_idx:
                entities.append(col_name)
                col_toks = tokenizer.tokenize(col_name)
                wordpieces.extend(col_toks)      
                entity_idx.append([col_idx] * len(col_toks))

        entities.append(tokenizer._sep_token)
        wordpieces.append(tokenizer._sep_token)
        entity_idx.append([-1])

        # "baseball_1" database is too long
        if len(wordpieces) > 430:  
           break
        
    return {"entities": entities,
            "wordpieces": wordpieces,
            "entity_idx": entity_idx}


def get_db_dict_from_id(db_id: str, tables_dict:Dict) -> Dict:
    """
    Returns the dictionary corresponding to a given database identifier.
    Args:
        db_id: string identifier unique to a database
    """
    for table_dict in tables_dict:
        if table_dict['db_id'] == db_id:
            return table_dict

    return None

if __name__ == "__main__":
    # initialize BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained(
        'bert-base-uncased',
        output_hidden_states=True
    )
    #config = model.config
    #config.update({"output_hidden_states": True})
    #model = BertModel(config)
    model.eval()

    # other inits
    cos_sim = CosineSimilarity(dim=1)
    stop_words = set(stopwords.words('english'))
    other_toks = set(['[CLS]', '[SEP]', ',', '.', '?'])
    stop_words.update(other_toks)
     
    # read database from tables.json
    with open('/home/fbrad/bit/IRNet/data/tables.json') as json_file:
        databases_dict = json.load(json_file)
    # db_dict = get_db_dict_from_id("car_1", databases_dict)
    # db_dict = get_db_dict_from_id("cre_Drama_Workshop_Groups", databases_dict)
    # table_annots = generate_db_table_annotations(db_dict)
    # column_annots = generate_db_column_annotations(db_dict)

    for db_dict in databases_dict:
        if db_dict["db_id"] != "car_1":
           continue
        out_dict = tokenize_db(db_dict, tokenizer)

        entities = out_dict["entities"]
        wordpieces = out_dict["wordpieces"]
        entity_idx = out_dict["entity_idx"]
        print("entity_idx = ", entity_idx)
        assert len(entities) == len(entity_idx), "wordpieces != entities len"

        entities_str = " ".join(entities)
        wps = tokenizer.tokenize(entities_str)
        assert wps == wordpieces, "different tokenization output"

    # sample question
    question = ("For the cars with 4 cylinders, which model has the largest "
                "horsepower?")
    
    # self.encode(text) <=> self.convert_tokens_to_ids(self.tokenize(text))
    # [101, ..., 102]
    question_ids = tokenizer.encode(text=question)

    # mask stopwords with 0s
    question_tokens = tokenizer.convert_ids_to_tokens(question_ids)
    stopwords_mask = [1 for tok in question_tokens]
    for idx, tok in enumerate(question_tokens):
        if tok in stop_words:
            if idx == len(question_tokens) - 1:
                stopwords_mask[idx] = 0
            elif not question_tokens[idx+1].startswith("##"):
                stopwords_mask[idx] = 0

    # [CLS] question_tokens [SEP] schema_tok_1 ... schema_tok_i
    question_offset = len(question_ids) # + 2

    # [..., 102]
    db_wordpiece_ids = tokenizer.convert_tokens_to_ids(wordpieces)
    assert len(db_wordpiece_ids) == len(wordpieces), "different wordpiece len"

    #seq_ids = tokenizer.encode(text=question, text_pair=entities_str,
    #                           add_special_tokens=True)
    seq_ids = torch.LongTensor([question_ids + db_wordpiece_ids])
    seq_type_ids = torch.LongTensor([[0] * len(question_ids) + 
                                     [1] * len(db_wordpiece_ids)])

    # feed sequence to BERT
    with torch.no_grad():
        # 1 x T x 768, (1 x T x 768, ...)
        last_hidden, _, all_hidden = model(input_ids=seq_ids,
                                           token_type_ids=seq_type_ids)

        # compute embedding for each entity, by averaging over its span tokens
        entities_output = torch.zeros(len(entities), 768)
        entity_offset = question_offset
        for idx, entity in enumerate(entities):
            if entity == "[SEP]":
                entity_offset += 1
                continue
            entity_len = len(entity_idx[idx])

            entity_output = last_hidden[0, 
                                        entity_offset:entity_offset+entity_len,
                                        :].mean(0)
            entities_output[idx] = entity_output
            entity_offset += entity_len
        
        # compute question token->entity alignments
        #question_output = last_hidden[0, 1:question_offset-1,:]

        # cosine alignment
        #print("Question size: ", question_output[0:1,:].size())
        print("Entities size: ", entities_output.size())
        print("Entities: ", entities)
        for idx, token in enumerate(question_tokens):
            if stopwords_mask[idx] == 0:
                continue
            
            question_output = last_hidden[:, idx, :]
            align_scores = cos_sim(question_output, entities_output)
            max_idx = torch.argmax(align_scores).item()
            print("%s->%s (%d)" % (token, entities[max_idx], max_idx))


    """
    alignments_list = []
    for schema_entity_description in table_annots:
        question_output, entity_output = cross_encode_pair(
            question, 
            schema_entity_description,
            tokenizer, 
            model,
            second_to_last=True
        )        
        # question_output, entity_output = bi_encode_pair(
        #     question, 
        #     schema_entity_description,
        #     tokenizer, 
        #     model,
        #     second_to_last=False
        # )   

        # compute cosine similarity between question tokens and schema entity
        # len(question) x 1 
        alignments = cos_sim(question_output, entity_output).squeeze()
        alignments_list.append(alignments)

    #for idx, table_annot in enumerate(column_annots):
    #    print(table_annot)

    for idx, tok in enumerate(question_tokens):
        if stopwords_mask[idx] == 0:
            continue

        # cosine between current token and schema elements
        # align_scores = ['{:.2f}'.format(x[idx]) for x in alignments_list]
        align_scores = [x[idx] for x in alignments_list]

        max_index = align_scores.index(max(align_scores))
        print("%s -> %s: %f" %(tok, table_annots[max_index][:30], max(align_scores)))

        #print("cos(%10s, entity) = %s" % (tok, ' '.join(align_scores)))
    """
    