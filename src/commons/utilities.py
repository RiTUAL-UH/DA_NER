import os
import csv
import json
import signal

from itertools import groupby
from typing import List, Dict


def read_json(filename, columns, json_by_line=True):
    with open(filename, 'r') as fp:
        if json_by_line:
            dataset = {colname: [] for colname in columns}
            for line in fp:
                data = json.loads(line.strip())
                for colname in columns:
                    dataset[colname].append(' '.join(data[colname]))
        else:
            dataset = json.load(fp)
    return dataset


def save_as_json(filepath, data, json_by_line=True):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w+') as fp:
        if json_by_line:
            for sample in data:
                fp.write(json.dumps(sample) + '\n')
        else:
            json.dump(data, fp)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels


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


def input_with_timeout(prompt, timeout, default=''):
    def alarm_handler(signum, frame):
        raise Exception("Time is up!")
    try:
        # set signal handler
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout)  # produce SIGALRM in `timeout` seconds

        return input(prompt)
    except Exception as ex:
        return default
    finally:
        signal.alarm(0)  # cancel alarm

def flatten(l):
    return [i for sublist in l for i in sublist]
