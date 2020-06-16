# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/24
# @Author  : Jiaqi&Zecheng
# @File    : data_process.py
# @Software: PyCharm
"""
import json
import argparse
import nltk
import os
import pickle
import torch
from .utils import symbol_filter, re_lemma, fully_part_header, group_header, partial_header, num2year, group_symbol, group_values, group_digital
from .utils import AGG, wordnet_lemmatizer
from .utils import load_dataSets
from transformers import PreTrainedTokenizer, BertTokenizer, \
        PreTrainedModel, BertModel, BertConfig
from nltk.corpus import stopwords
from preprocess.schema_linker import tokenize_db
from torch.nn import CosineSimilarity

def process_datas(datas, args):
    """

    :param datas:
    :param args:
    :return:
    """
    with open(os.path.join(args.conceptNet, 'english_RelatedTo.pkl'), 'rb') as f:
        english_RelatedTo = pickle.load(f)

    with open(os.path.join(args.conceptNet, 'english_IsA.pkl'), 'rb') as f:
        english_IsA = pickle.load(f)

    # copy of the origin question_toks
    for d in datas:
        if 'origin_question_toks' not in d:
            d['origin_question_toks'] = d['question_toks']

    for entry in datas:
        entry['question_toks'] = symbol_filter(entry['question_toks'])
        origin_question_toks = symbol_filter([x for x in entry['origin_question_toks'] if x.lower() != 'the'])
        
        # remove 'the' from question tokens
        question_toks = [wordnet_lemmatizer.lemmatize(x.lower()) for x in entry['question_toks'] if x.lower() != 'the']

        entry['question_toks'] = question_toks

        table_names = []
        table_names_pattern = []

        for y in entry['table_names']:
            # 'singing in concert' => 'sing in concert'
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            table_names.append(" ".join(x))
            # ??
            x = [re_lemma(x.lower()) for x in y.split(' ')]
            table_names_pattern.append(" ".join(x))

        header_toks = [] # list of column names
        header_toks_list = [] # list of lists of column tokens

        header_toks_pattern = []
        header_toks_list_pattern = []

        for y in entry['col_set']:
            # add lemmatized version of each column
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            header_toks.append(" ".join(x))
            header_toks_list.append(x)

            x = [re_lemma(x.lower()) for x in y.split(' ')]
            header_toks_pattern.append(" ".join(x))
            header_toks_list_pattern.append(x)

        num_toks = len(question_toks)
        idx = 0
        tok_concol = []  # question tokens/spans
        type_concol = [] # types (TABLE/COLUMN) of tokens/spans
        nltk_result = nltk.pos_tag(question_toks)

        while idx < num_toks:

            # fully header
            # check if question span (no single tokens) matches a column name
            end_idx, header = fully_part_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for table
            # check if question span/token matches a table name
            end_idx, tname = group_header(question_toks, idx, num_toks, table_names)
            if tname:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["table"])
                idx = end_idx
                continue

            # check if question span/token matches a column name
            end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check if span contains same tokens as a column
            end_idx, tname = partial_header(question_toks, idx, header_toks_list)
            if tname:
                tok_concol.append(tname)
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for aggregation tokens (avg/max/min etc.)
            end_idx, agg = group_header(question_toks, idx, num_toks, AGG)
            if agg:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["agg"])
                idx = end_idx
                continue

            if nltk_result[idx][1] == 'RBR' or nltk_result[idx][1] == 'JJR':
                tok_concol.append([question_toks[idx]])
                type_concol.append(['MORE'])
                idx += 1
                continue

            if nltk_result[idx][1] == 'RBS' or nltk_result[idx][1] == 'JJS':
                tok_concol.append([question_toks[idx]])
                type_concol.append(['MOST'])
                idx += 1
                continue

            # string match for Time Format
            # replace '1999' with 'year' and search again for spans that match
            # some column
            if num2year(question_toks[idx]):
                question_toks[idx] = 'year'
                end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
                if header:
                    tok_concol.append(question_toks[idx: end_idx])
                    type_concol.append(["col"])
                    idx = end_idx
                    continue

            def get_concept_result(toks, graph):
                for begin_id in range(0, len(toks)):
                    for r_ind in reversed(range(1, len(toks) + 1 - begin_id)):
                        tmp_query = "_".join(toks[begin_id:r_ind])
                        if tmp_query in graph:
                            mi = graph[tmp_query]
                            for col in entry['col_set']:
                                if col in mi:
                                    return col

            # search value spans (spans surrounded by quotations '')
            # in ConceptNet;
            end_idx, symbol = group_symbol(question_toks, idx, num_toks)
            if symbol:
                tmp_toks = [x for x in question_toks[idx: end_idx]]
                assert len(tmp_toks) > 0, print(symbol, question_toks)
                # symbol = 'New York'
                # return column that matches any entity in ConceptNet(query),
                # where query is 'New' and 'New York'
                pro_result = get_concept_result(tmp_toks, english_IsA)
                if pro_result is None:
                    pro_result = get_concept_result(tmp_toks, english_RelatedTo)
                if pro_result is None:
                    pro_result = "NONE"
                for tmp in tmp_toks:
                    # add individual tokens of a value span
                    tok_concol.append([tmp]) 
                    # add matched column/NONE 
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            # search for proper noun spans?
            end_idx, values = group_values(origin_question_toks, idx, num_toks)
            if values and (len(values) > 1 or question_toks[idx - 1] not in ['?', '.']):
                tmp_toks = [wordnet_lemmatizer.lemmatize(x) for x in question_toks[idx: end_idx] if x.isalnum() is True]
                assert len(tmp_toks) > 0, print(question_toks[idx: end_idx], values, question_toks, idx, end_idx)
                pro_result = get_concept_result(tmp_toks, english_IsA)
                if pro_result is None:
                    pro_result = get_concept_result(tmp_toks, english_RelatedTo)
                if pro_result is None:
                    pro_result = "NONE"
                for tmp in tmp_toks:
                    tok_concol.append([tmp])
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            # check if number
            result = group_digital(question_toks, idx)
            if result is True:
                tok_concol.append(question_toks[idx: idx + 1])
                type_concol.append(["value"])
                idx += 1
                continue
            if question_toks[idx] == ['ha']:
                question_toks[idx] = ['have']

            tok_concol.append([question_toks[idx]])
            type_concol.append(['NONE'])
            idx += 1
            continue

        entry['question_arg'] = tok_concol
        entry['question_arg_type'] = type_concol
        entry['nltk_pos'] = nltk_result

    return datas

def process_datas_bert_linking(datas, args):
    """
    Similar to process_datas function, but links tokens to schema entities
    using BERT similarity
    :param datas:
    :param args:
    :return:
    """
    tables = args.tables

    with open(os.path.join(args.conceptNet, 'english_RelatedTo.pkl'), 'rb') as f:
        english_RelatedTo = pickle.load(f)

    with open(os.path.join(args.conceptNet, 'english_IsA.pkl'), 'rb') as f:
        english_IsA = pickle.load(f)

    # copy of the origin question_toks
    for d in datas:
        if 'origin_question_toks' not in d:
            d['origin_question_toks'] = d['question_toks']

    count = 0
    for entry in datas:
        #if entry['db_id'] != 'baseball_1':
        #    continue
        count += 1
        print("Count = %d/%d" % (count, len(datas)))
        #if count == 20:
        #    break
        entry['question_toks'] = symbol_filter(entry['question_toks'])
        origin_question_toks = symbol_filter([x for x in entry['origin_question_toks'] if x.lower() != 'the'])
        
        # remove 'the' from question tokens
        question_toks = [wordnet_lemmatizer.lemmatize(x.lower()) for x in entry['question_toks'] if x.lower() != 'the']

        entry['question_toks'] = question_toks
        #print("question_toks = ", question_toks)

        entry_db = tables[entry['db_id']]
        db_entities = entry_db['entities']
        #print("db_entities = ", db_entities)
        db_wordpieces = entry_db['wordpieces']
        db_entity_idx = entry_db['entity_idx']

        # [101, question_toks, 102, db_wordpiece_ids, 102]
        db_wordpiece_ids = args.tokenizer.convert_tokens_to_ids(db_wordpieces)
        q_toks = ['[CLS]'] + question_toks + ['[SEP]']
        q_wordpiece_ids = args.tokenizer.convert_tokens_to_ids(q_toks)
        q_offset = len(q_wordpiece_ids)

        seq_ids = torch.LongTensor([q_wordpiece_ids + db_wordpiece_ids])
        seq_type_ids = torch.LongTensor([[0] * len(q_wordpiece_ids) + 
                                         [1] * len(db_wordpiece_ids)])

        # compute embedding for each db entity, by averaging its span tokens
        with torch.no_grad():
            # 1 x T x 768, (1 x T x 768, ...)
            last_hidden, _, _ = model(input_ids=seq_ids,
                                      token_type_ids=seq_type_ids)

        entities_output = torch.zeros(len(db_entities), 768)
        entity_offset = q_offset
        for idx, entity in enumerate(db_entities):
            if entity == "[SEP]":
                entity_offset += 1
                continue
            entity_len = len(db_entity_idx[idx])

            entity_output = last_hidden[0, 
                                        entity_offset:entity_offset+entity_len,
                                        :].mean(0)
            entities_output[idx] = entity_output
            entity_offset += entity_len

        table_names = []
        table_names_pattern = []

        for y in entry['table_names']:
            # 'singing in concert' => 'sing in concert'
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            table_names.append(" ".join(x))
            # ??
            x = [re_lemma(x.lower()) for x in y.split(' ')]
            table_names_pattern.append(" ".join(x))

        header_toks = [] # list of column names
        header_toks_list = [] # list of lists of column tokens

        header_toks_pattern = []
        header_toks_list_pattern = []

        for y in entry['col_set']:
            # add lemmatized version of each column
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            header_toks.append(" ".join(x))
            header_toks_list.append(x)

            x = [re_lemma(x.lower()) for x in y.split(' ')]
            header_toks_pattern.append(" ".join(x))
            header_toks_list_pattern.append(x)

        num_toks = len(question_toks)
        idx = 0
        tok_concol = []  # question tokens/spans
        type_concol = [] # types (TABLE/COLUMN) of tokens/spans
        nltk_result = nltk.pos_tag(question_toks)

        while idx < num_toks:
            # SKIP THIS
            # check if question span (no single tokens) matches a column name
            # end_idx, header = fully_part_header(question_toks, idx, num_toks, header_toks)
            # if header:
            #     tok_concol.append(question_toks[idx: end_idx])
            #     type_concol.append(["col"])
            #     idx = end_idx
            #     continue

            # SKIP THIS
            # check if question span/token matches a table name
            # end_idx, tname = group_header(question_toks, idx, num_toks, table_names)
            # if tname:
            #     tok_concol.append(question_toks[idx: end_idx])
            #     type_concol.append(["table"])
            #     idx = end_idx
            #     continue

            # SKIP THIS
            # check if question span/token matches a column name
            # end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
            # if header:
            #     tok_concol.append(question_toks[idx: end_idx])
            #     type_concol.append(["col"])
            #     idx = end_idx
            #     continue

            # SKIP THIS
            # check if span contains same tokens as a column
            # end_idx, tname = partial_header(question_toks, idx, header_toks_list)
            # if tname:
            #     tok_concol.append(tname)
            #     type_concol.append(["col"])
            #     idx = end_idx
            #     continue

            # check for aggregation tokens (avg/max/min etc.)
            end_idx, agg = group_header(question_toks, idx, num_toks, AGG)
            if agg:
                #print("     agg span: ", question_toks[idx: end_idx])
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["agg"])
                idx = end_idx
                continue

            if nltk_result[idx][1] == 'RBR' or nltk_result[idx][1] == 'JJR':
                #print("     adverb token: ", question_toks[idx])
                tok_concol.append([question_toks[idx]])
                type_concol.append(['MORE'])
                idx += 1
                continue

            if nltk_result[idx][1] == 'RBS' or nltk_result[idx][1] == 'JJS':
                #print("     adverb token: ", question_toks[idx])
                tok_concol.append([question_toks[idx]])
                type_concol.append(['MOST'])
                idx += 1
                continue

            # SKIP THIS
            # string match for Time Format: replace '1999' with 'year' and 
            # search again for spans that match Some column
            # if num2year(question_toks[idx]):
            #     question_toks[idx] = 'year'
            #     end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
            #     if header:
            #         tok_concol.append(question_toks[idx: end_idx])
            #         type_concol.append(["col"])
            #         idx = end_idx
            #         continue

            def get_concept_result(toks, graph):
                for begin_id in range(0, len(toks)):
                    for r_ind in reversed(range(1, len(toks) + 1 - begin_id)):
                        tmp_query = "_".join(toks[begin_id:r_ind])
                        if tmp_query in graph:
                            mi = graph[tmp_query]
                            for col in entry['col_set']:
                                if col in mi:
                                    return col

            # search value spans (spans surrounded by quotations '')
            # in ConceptNet;
            end_idx, symbol = group_symbol(question_toks, idx, num_toks)
            if symbol:
                tmp_toks = [x for x in question_toks[idx: end_idx]]
                assert len(tmp_toks) > 0, print(symbol, question_toks)
                # symbol = 'New York'
                # return column that matches any entity in ConceptNet(query),
                # where query is 'New' and 'New York'
                pro_result = get_concept_result(tmp_toks, english_IsA)
                if pro_result is None:
                    pro_result = get_concept_result(tmp_toks, english_RelatedTo)
                if pro_result is None:
                    pro_result = "NONE"
                #print("     value span %s -> %s: ", (tmp_toks, pro_result))
                for tmp in tmp_toks:
                    # add individual tokens of a value span
                    tok_concol.append([tmp]) 
                    # add matched column/NONE 
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            # search for proper noun spans?
            end_idx, values = group_values(origin_question_toks, idx, num_toks)
            if values and (len(values) > 1 or question_toks[idx - 1] not in ['?', '.']):
                tmp_toks = [wordnet_lemmatizer.lemmatize(x) for x in question_toks[idx: end_idx] if x.isalnum() is True]
                assert len(tmp_toks) > 0, print(question_toks[idx: end_idx], values, question_toks, idx, end_idx)
                pro_result = get_concept_result(tmp_toks, english_IsA)
                if pro_result is None:
                    pro_result = get_concept_result(tmp_toks, english_RelatedTo)
                if pro_result is None:
                    pro_result = "NONE"
                #print("     proper noun span %s -> %s: ", (tmp_toks, pro_result))
                for tmp in tmp_toks:
                    tok_concol.append([tmp])
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            # check if number
            result = group_digital(question_toks, idx)
            if result is True:
                #print("     number token %s", question_toks[idx])
                tok_concol.append(question_toks[idx: idx + 1])
                type_concol.append(["value"])
                idx += 1
                continue
            if question_toks[idx] == ['ha']:
                question_toks[idx] = ['have']

            if question_toks[idx] in args.stop_words:
                tok_concol.append(question_toks[idx: idx + 1])
                type_concol.append(["NONE"])
                idx += 1
                continue

            # if everything else fail, link using BERT
            question_output = last_hidden[:, idx+1, :]
            align_scores = args.cos_sim(question_output, entities_output)
            max_idx = torch.argmax(align_scores).item()
            if db_entity_idx[max_idx-1] == [-1]:
                ent_type = 'table'
            else:
                ent_type = 'col'
            #print("     BERT linking: %s->%s (%s)" % (question_toks[idx], 
            #                                          db_entities[max_idx], 
            #                                          ent_type))

            tok_concol.append([question_toks[idx]])
            type_concol.append([ent_type])
            idx += 1
            continue

        entry['question_arg'] = tok_concol
        entry['question_arg_type'] = type_concol
        entry['nltk_pos'] = nltk_result


    return datas



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--bert', dest='bert_linking', action='store_true')
    arg_parser.add_argument('--output', type=str, help='output data')
    
    args = arg_parser.parse_args()
    args.conceptNet = './data/conceptNet'
    print("args.bert_linking = ", args.bert_linking)
    stop_words = set(stopwords.words('english') + [".", ",", "?", "\'"])
    args.stop_words = stop_words

    # loading dataSets
    datas, tables = load_dataSets(args)
    
    if args.bert_linking:
        # load BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained(
            'bert-base-uncased',
            output_hidden_states=True
        )
        model.eval()
        cos_sim = CosineSimilarity(dim=1)
        args.model = model
        args.tokenizer = tokenizer
        args.cos_sim = cos_sim

        # tokenize schema entities for each database (table names and columns)
        db_tokenization = {}
        for db_name, db_dict in tables.items():
            out_dict = tokenize_db(db_dict, tokenizer)
            db_dict["entities"] = out_dict["entities"]
            db_dict["wordpieces"] = out_dict["wordpieces"]
            db_dict["entity_idx"] = out_dict["entity_idx"]
        args.tables = tables

        process_result = process_datas_bert_linking(datas, args)
    else:
        process_result = process_datas(datas, args)

    
    """
    with open(args.output, 'w') as f:
        json.dump(datas, f, indent=4)
    """
