import os
import torch
import src.commons.utilities as utils

from typing import List, Dict
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


class DomainDatasetBase(Dataset):
    def __init__(self,
                partition: str,
                data_file: str,
                data_cols: List[str],
                tokenizer: AutoTokenizer,
                label_pad_token_id: int):

        self.partition = partition

        self.data_file = data_file
        self.data_cols = data_cols

        self.tokenizer = tokenizer

        self.label_pad_token_id = label_pad_token_id

    def _init_data_fields(self, dataset=None):
        if dataset is None:
            dataset = utils.read_json(self.data_file, self.data_cols)

        self.inputs = self.dataset['tokens']
        self.labels = self.dataset['labels'] if 'labels' in self.dataset else None

    def _prepare_encoding_fields_from_start(self, max_length=64):
        """
        Only call this method if you want everything tokenized and encoded from the very beginning.
        """
        self.dataset = utils.read_json(self.data_file, self.data_cols)

        self.inputs = self.dataset['tokens']
        self.labels = self.dataset['labels'] if 'labels' in self.dataset else None

        self.tokenized = self.tokenizer(self.inputs)
        self.input_ids = self.tokenized['input_ids']
        self.input_msk = self.tokenized['attention_mask']
        self.label_ids = self.tokenizer(self.labels).input_ids if self.labels is not None else self.tokenizer(self.inputs).input_ids

        # filter out long tokenized sentences
        indexes = [i for i, (x, y) in enumerate(zip(self.input_ids, self.label_ids)) if (len(x) <= max_length and len(y) <= max_length)]

        self.inputs = [self.inputs[i] for i in indexes]
        self.labels = [self.labels[i] for i in indexes] if self.labels is not None else None

        self.input_ids = [self.input_ids[i] for i in indexes]
        self.input_msk = [self.input_msk[i] for i in indexes]
        self.label_ids = [self.label_ids[i] for i in indexes]

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def collate_fn(self, batch):
        raise NotImplementedError()


class DomainDataset(DomainDatasetBase):
    """
    This class encodes the data from the beginning.
    """
    def __init__(self,
                partition: str,
                data_file: str,
                data_cols: List[str],
                tokenizer: AutoTokenizer,
                label_pad_token_id: int):

        super().__init__(partition, data_file, data_cols, tokenizer, label_pad_token_id)

        # Always encodes the data from the beginning, regardless the partition
        self._prepare_encoding_fields_from_start()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        input_msk = self.input_msk[index]
        label_ids = self.label_ids[index]

        return input_ids, input_msk, label_ids

    def collate_fn(self, batch, pad_to_multiple_of=8):
        # Unwrap the batch into every field
        input_ids, input_msk, label_ids = map(list, zip(*batch))

        # Padded variables
        p_input_ids, p_input_msk, p_label_ids = [], [], []

        # How much padding do we need?
        # When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        input_max_length = (max(map(len, input_ids)) // pad_to_multiple_of + 1) * pad_to_multiple_of
        label_max_length = (max(map(len, label_ids)) // pad_to_multiple_of + 1) * pad_to_multiple_of

        for i in range(len(input_ids)):
            input_padding_length = input_max_length - len(input_ids[i])
            label_padding_length = label_max_length - len(label_ids[i])

            p_input_ids.append(input_ids[i] + [self.tokenizer.pad_token_id] * input_padding_length)
            p_input_msk.append(input_msk[i] + [0] * input_padding_length)
            p_label_ids.append(label_ids[i] + [self.label_pad_token_id] * label_padding_length)

        batch_dict = {
            'input_ids': torch.tensor(p_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(p_input_msk, dtype=torch.long),
            'labels': torch.tensor(p_label_ids, dtype=torch.long)
        }

        return batch_dict


def get_dataloaders(dargs, targs, oargs, data_dir, data_cols, tokenizer):
    corpus = dict()
    for split in dargs:
        splits, fnames = [split], [dargs[split]]
            
        for split, fname in zip(splits, fnames):
            data_file = os.path.join(data_dir, fname)
            dataset = DomainDataset(split, data_file, data_cols, tokenizer, targs.label_pad_token_id)

            if split == 'train':
                oargs.train_batch_size = oargs.per_gpu_train_batch_size
                batch_size = oargs.train_batch_size
                sampler = RandomSampler(dataset)
            else:
                oargs.eval_batch_size = oargs.per_gpu_eval_batch_size
                batch_size = oargs.eval_batch_size
                sampler = SequentialSampler(dataset)

            corpus[split] = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate_fn)

    return corpus
