import os
import re
import csv
import json
import src.commons.globals as glb

from itertools import groupby
from typing import List, Dict
from collections import defaultdict


label_schema = ["B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC", "B-ORG", "I-ORG", "B-GPE", "I-GPE", 
            "B-LOC", "I-LOC", "B-PRODUCT", "I-PRODUCT", "B-EVENT", "I-EVENT", "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW", 
            "B-LANGUAGE", "I-LANGUAGE", "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-PERCENT", "I-PERCENT","B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY", 
            "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL"]


def ner2domain(corpus_dir, save_dir, colnames, datasets=['train', 'dev', 'test']):
    """
    corpus_dir: str
    save_dir: str
    colnames: List[str]
    datasets: List[str]
    """

    corpus_dir = os.path.join(glb.DATA_DIR, corpus_dir)
    save_dir = os.path.join(glb.DATA_DIR, save_dir)

    for split in datasets:
        data_path = os.path.join(corpus_dir, split + '.txt')
        save_path = os.path.join(save_dir, split + '.txt')
        save_domain(data_path, save_path, colnames + '.txt')

    for split in datasets:
        data = load_from_json(os.path.join(save_dir, split + '.json'), json_by_line=True)
        print('[{: >6}]: {:,}'.format(split.upper(), len(data)))


def domain2ner(corpus_dir, save_dir, datasets=['train', 'dev', 'test']):
    """
    corpus_dir: str
    save_dir: str
    datasets: List[str]
    """

    corpus_dir = os.path.join(glb.DATA_DIR, corpus_dir)
    save_dir = os.path.join(glb.DATA_DIR, save_dir)

    data_stats = defaultdict(list)
    data_stats['entities'] = defaultdict(int)

    for split in datasets:
        data_path = os.path.join(corpus_dir, split + '.json')
        data = load_from_json(data_path, json_by_line=True)

        save_path = os.path.join(save_dir, split + '.txt')
        save_data = defaultdict(list)
        for i in range(len(data)):
            sentence = data[i]['tokens']
            tokens, labels = unlinearize_sentence(sentence)
            if len(tokens) and len(labels):
                save_data['tokens'].append(tokens)
                save_data['labels'].append(labels)

                data_stats['consistency'].append(float(data[i]['consistency']) if 'consistency' in data[i] else 0)
                data_stats['adequacy'].append(float(data[i]['adequacy']) if 'adequacy' in data[i] else 0)
                data_stats['fluency'].append(float(data[i]['fluency']) if 'fluency' in data[i] else 0)
                data_stats['diversity'].append(float(data[i]['diversity']) if 'diversity' in data[i] else 0)

                for entity_type in label_schema:
                    if entity_type.startswith('B-'):
                        data_stats['entities'][entity_type[2:]] += labels.count(entity_type)
        
        write_conll(save_path, save_data)

        data_stats['num_samples'] = len(save_data['tokens'])
        data_stats['num_samples_with_entities'] = len([label for label in save_data['labels'] if any(x in label_schema for x in label)])
        data_stats['num_samples_no_entities'] = len([label for label in save_data['labels'] if all(x not in label_schema for x in label)])
        data_stats['num_entities'] = len([x for label in save_data['labels'] for x in label if x.startswith('B-')])
        data_stats['num_entity_tokens'] = len([x for label in save_data['labels'] for x in label if x.startswith('B-') or x.startswith('I-')])

        data_stats['consistency'] = round(sum(data_stats['consistency']) / len(data_stats['consistency']), 3)
        data_stats['adequacy'] = round(sum(data_stats['adequacy']) / len(data_stats['adequacy']), 3)
        data_stats['fluency'] = round(sum(data_stats['fluency']) / len(data_stats['fluency']), 3)
        data_stats['diversity'] = round(sum(data_stats['diversity']) / len(data_stats['diversity']), 3)

        print(json.dumps(data_stats, sort_keys=False, indent=4))


def linearize_sentence(tokens, labels, linearize=True):
    sentence = []

    is_entity = False
    for i in range(len(tokens)):
        if linearize:
            if labels[i].startswith('B-'):
                sentence.append('<START_' + labels[i][2:].upper() + '>')
                sentence.append(tokens[i])
                is_entity = True
                if is_entity and (i == len(tokens) - 1 or not labels[i+1].startswith('I-')):
                    sentence.append('<END_' + labels[i][2:].upper() + '>')
                    is_entity = False
            elif labels[i].startswith('I-'):
                sentence.append(tokens[i])
                if is_entity and (i == len(tokens) - 1 or not labels[i+1].startswith('I-')):
                    sentence.append('<END_' + labels[i][2:].upper() + '>')
                    is_entity = False
            else:
                sentence.append(tokens[i])
        else:
            sentence.append(tokens[i])
    
    assert sum([x.startswith('<START_') for x in sentence]) == sum([x.startswith('<END_') for x in sentence]), '{} {}'.format(sum([x.startswith('<START_') for x in sentence]), sum([x.startswith('<END_') for x in sentence]))
    assert len(sentence) - len(tokens) == sum([x.startswith('B-') for x in labels]) * 2, '{} {} {}'.format(len(sentence), len(tokens), sum([x.startswith('B-') for x in labels]))

    return sentence


def unlinearize_sentence(sentence):
    tokens, labels = [], []
    num_b_entity, num_i_entity = 0, 0

    b_entity, i_entity, entity_type = False, False, None
    for token in sentence:
        if token.startswith('<START_') and token.endswith('>'):
            b_entity, i_entity, entity_type = True, False, token[7:-1]
            num_b_entity += 1
        elif token.startswith('<END_') and token.endswith('>'):
            b_entity, i_entity, entity_type = False, False, None
            num_i_entity += 1
        else:
            tokens.append(token)
            if b_entity:
                labels.append('B-' + entity_type)
                b_entity, i_entity = False, True
            elif i_entity:
                labels.append('I-' + entity_type)
            else:
                labels.append('O')
        
        if (num_b_entity > num_i_entity + 1) or (num_b_entity < num_i_entity):
            return ([], [])

    assert len(tokens) == len(labels), "tokens: {} {} \n labels: {} {}".format(len(tokens), tokens, len(labels), labels)

    return (tokens, labels) if num_b_entity == num_i_entity else ([], [])


def save_domain(data_path, save_path, colnames, sentence_linearization=True, json_format=True):
    """
    data_path: str
    save_path: str
    colnames: List[str]
    """
    
    data = read_conll(data_path, colnames)

    zipped = list(zip(*data.values()))

    save_data = []
    for i in range(len(zipped)):
        if sentence_linearization:
            tokens, labels = zipped[i]
            lineazed_tokens = linearize_sentence(tokens, labels)
            tmp_data = {'tokens': lineazed_tokens}
        else:
            tmp_data = {'tokens': zipped[i][0]}

        save_data.append(tmp_data)
    
    if json_format:
        save_path = save_path.replace('.txt', '.json')

    save_as_json(save_path, save_data, json_by_line=True)


def save_ner(data_path, save_path, data_colnames, save_colnames):
    """
    data_path: str
    save_path: str
    data_colnames: List[str]
    save_colnames: List[str]
    """

    dataset = read_conll(data_path, data_colnames)

    save_data = {}
    for colname in save_colnames:
        save_data[colname] = dataset[colname]

    write_conll(save_path, save_data)


def load_from_json(filepath, max_length=None, json_by_line=False):
    with open(filepath, 'r') as fp:
        if json_by_line:
            data = []
            for line in fp:
                json_data = json.loads(line.strip())
                json_data['tokens'] = list(filter(None, re.split(',|!|\?|\'|\.|\s', json_data['tokens']))) if type(json_data['tokens']) == str else json_data['tokens']
                if (not max_length) or (len(json_data['tokens']) <= max_length):
                    data.append(json_data)
        else:
            data = json.load(fp)
    return data


def save_as_json(filepath, data, json_by_line=False):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as fp:
        if json_by_line:
            for sample in data:
                fp.write(json.dumps(sample) + '\n')
        else:
            json.dump(data, fp)


def read_conll(filename, columns, delimiter='\t'):
    def is_empty_line(line_pack):
        return all(field.strip() == '' for field in line_pack)

    data = []
    with open(filename) as fp:
        reader = csv.reader(fp, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        groups = groupby(reader, is_empty_line)

        for is_empty, pack in groups:
            if is_empty is False:
                data.append([list(field) for field in zip(*pack)])
    
    data = list(zip(*data))
    dataset = {colname: list(data[columns[colname]]) for colname in columns}

    return dataset


def write_conll(filename, data, colnames: List[str] = None, delimiter='\t'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if colnames is None:
        colnames = list(data.keys())

    any_key = colnames[0]

    with open(filename, 'w') as fp:
        for sample_i in range(len(data[any_key])):
            for token_i in range(len(data[any_key][sample_i])):
                row = [data[col][sample_i][token_i] for col in colnames]
                fp.write(delimiter.join(row) + '\n')
            fp.write('\n')


if __name__ == '__main__':
    # ============== convert conll to json ================
    # corpus_dir = 'ner/nw'
    # save_dir = 'linearized_domain/nw'
    # colnames = {'tokens': 0, 'labels': 1}
    # datasets = ['train', 'dev', 'test']
    
    # ner2domain(corpus_dir, save_dir, colnames, datasets)
    # =====================================================
    
    # ============== convert json to conll ================
    # datasets = ['train', 'dev', 'test']
    # corpus_dir = 'linearized_pseudo/nw/sm'
    # save_dir = 'ner_pseudo/nw/sm'

    # domain2ner(corpus_dir, save_dir, datasets)
    # =====================================================
    
    exit()
