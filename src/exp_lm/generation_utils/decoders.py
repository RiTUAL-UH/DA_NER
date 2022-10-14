import torch
import collections

from src.exp_lm.generation_utils.trie import Trie
from src.exp_lm.generation_utils.data_selection import Consistency, Adequacy, Fluency, Diversity

from src.exp_lm.main import accelerator


class DecoderBase():
    
    def __init__(self, model, gen_params, tokenizer):
        self.model = model
        self.gen_params = gen_params
        self.tokenizer = tokenizer
    
    def generate(self, task_prefix, input_ids, attention_mask=None, num_return_sequences=1):
        raise NotImplementedError('The DecoderBase class should never execute generate!')


class Decoder(DecoderBase):
    
    def __init__(self, model, gen_params, tokenizer):
        super().__init__(model, gen_params, tokenizer)
    
    def generate(self, task_prefix, input_ids, attention_mask=None):
        device = input_ids.device

        task_prefix = task_prefix.repeat(1, 1).to(device)
        task_mask = torch.ones_like(task_prefix).to(device)
        task_encoder_outputs = accelerator.unwrap_model(self.model).t5.encoder(task_prefix, task_mask)

        encoder_outputs = accelerator.unwrap_model(self.model).t5.encoder(input_ids, attention_mask)
        
        encoder_outputs.last_hidden_state = torch.cat([task_encoder_outputs[0], encoder_outputs[0]], dim=1)
        encoder_attention_mask = torch.cat([task_mask, attention_mask], dim=1)

        generated_outputs = accelerator.unwrap_model(self.model).t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            do_sample=self.gen_params.do_sample,
            max_length=self.gen_params.max_length,
            min_length=self.gen_params.min_length,
            top_k=self.gen_params.top_k,
            top_p=self.gen_params.top_p,
            temperature=self.gen_params.temperature,
            early_stopping=self.gen_params.early_stopping,
            num_return_sequences=self.gen_params.num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        sents = generated_outputs.sequences
        probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1).argmax(-1)
        
        return {'sequences': sents, 'probs': probs}


class ConstrainedDecoderBase(DecoderBase):
    
    def __init__(self, model, gen_params, tokenizer):
        super().__init__(model, gen_params, tokenizer)
        
        self.trie = Trie(tokenizer)
    
    def collect_all_candidates(self, dataloaders=None):
        # remove indexes for '<END_ENTITY_TYPE>' tokens to make sure the it can only appear after the corresponding '<START_ENTITY_TYPE>' token.
        remove_candidates = [self.tokenizer.convert_tokens_to_ids(x) for x in self.tokenizer.new_tokens if x.startswith('<END_')]

        if dataloaders:
            all_candidates = []
            for split in dataloaders:
                input_ids = dataloaders[split].dataset.input_ids
                all_candidates += [x for input in input_ids for x in input]
            self.candidates = list(set(all_candidates) - set(remove_candidates))
        else:
            self.candidates = list(set(range(len(self.tokenizer))) - set(remove_candidates))
    
    def build_trie(self, sentences):
        entities = collections.defaultdict(list)

        for sent in sentences:
            sent = sent.split()
            sent_idx = 0
            while sent_idx < len(sent):
                if sent[sent_idx].startswith('<START_') and sent[sent_idx].endswith('>'):
                    entity_type = sent[sent_idx]

                    entity = []
                    while sent_idx < len(sent):
                        entity_token = sent[sent_idx]
                        entity.append(entity_token)

                        if sent[sent_idx].startswith('<END_') and sent[sent_idx].endswith('>'):
                            break

                        sent_idx += 1

                    entities[entity_type].append(entity)

                sent_idx += 1

        for entity_type in entities:
            for entity in entities[entity_type]:
                self.trie.insert(entity)

    def print_trie_by_entity_type(self, entity_type):
        start_entity_type = '<START_' + entity_type.upper() + '>'
        self.trie.print_trie(self.trie.root.children[self.tokenizer.convert_tokens_to_ids([start_entity_type])[0]])

    def prefix_allowed_tokens_fn(self, batch_id, input_ids):
        self.trie.tmp_node.setdefault(batch_id, self.trie.root)
        
        x = input_ids[-1].item()
        if x in self.trie.tmp_node[batch_id].children:
            self.trie.tmp_node[batch_id] = self.trie.tmp_node[batch_id].children[x]
            candidates = list(self.trie.tmp_node[batch_id].children)

            if len(candidates) == 0:
                self.trie.tmp_node[batch_id] = self.trie.root
                candidates = self.candidates
        else:
            candidates = self.candidates

        return candidates
    
    def generate(self, task_prefix, input_ids, attention_mask=None):
        raise NotImplementedError('The ConstrainedDecoderBase class should never execute generate!')


class ConstrainedDecoder(ConstrainedDecoderBase):
    
    def __init__(self, model, gen_params, tokenizer):
        super().__init__(model, gen_params, tokenizer)
    
    def generate(self, task_prefix, input_ids, attention_mask=None):
        device = input_ids.device

        task_prefix = task_prefix.repeat(1, 1).to(device)
        task_mask = torch.ones_like(task_prefix).to(device)
        task_encoder_outputs = accelerator.unwrap_model(self.model).t5.encoder(task_prefix, task_mask)

        encoder_outputs = accelerator.unwrap_model(self.model).t5.encoder(input_ids, attention_mask)
        
        encoder_outputs.last_hidden_state = torch.cat([task_encoder_outputs[0], encoder_outputs[0]], dim=1)
        encoder_attention_mask = torch.cat([task_mask, attention_mask], dim=1)

        generated_outputs = accelerator.unwrap_model(self.model).t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            do_sample=self.gen_params.do_sample,
            max_length=self.gen_params.max_length,
            min_length=self.gen_params.min_length,
            top_k=self.gen_params.top_k,
            top_p=self.gen_params.top_p,
            temperature=self.gen_params.temperature,
            early_stopping=self.gen_params.early_stopping,
            num_return_sequences=self.gen_params.num_return_sequences,
            prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
            output_scores=True,
            return_dict_in_generate=True
        )

        sents = generated_outputs.sequences
        probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1).argmax(-1)
        
        return {'sequences': sents, 'probs': probs}


class ConstrainedDecoderWithSelection(ConstrainedDecoderBase):
    
    def __init__(self, model, gen_params, tokenizer):
        super().__init__(model, gen_params, tokenizer)
        
        self.consistency_score = Consistency(tokenizer, model)
        self.adequacy_score = Adequacy()
        self.fluency_score = Fluency()
        self.diversity_score = Diversity()
    
    def generate(self, task_prefix, input_ids, attention_mask=None):
        device = input_ids.device

        task_prefix = task_prefix.repeat(1, 1).to(device)
        task_mask = torch.ones_like(task_prefix).to(device)
        task_encoder_outputs = accelerator.unwrap_model(self.model).t5.encoder(task_prefix, task_mask)

        encoder_outputs = accelerator.unwrap_model(self.model).t5.encoder(input_ids, attention_mask)
        
        encoder_outputs.last_hidden_state = torch.cat([task_encoder_outputs[0], encoder_outputs[0]], dim=1)
        encoder_attention_mask = torch.cat([task_mask, attention_mask], dim=1)

        generated_outputs = accelerator.unwrap_model(self.model).t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            do_sample=self.gen_params.do_sample,
            max_length=self.gen_params.max_length,
            min_length=self.gen_params.min_length,
            top_k=self.gen_params.top_k,
            top_p=self.gen_params.top_p,
            temperature=self.gen_params.temperature,
            early_stopping=self.gen_params.early_stopping,
            num_return_sequences=self.gen_params.num_return_sequences,
            prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
            output_scores=True,
            return_dict_in_generate=True
        )

        decoded_sequences = []
        for sequence in generated_outputs.sequences:
            decoded = self.tokenizer.decode(sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_sequences.append(decoded)
        
        probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1).argmax(-1)
        
        input_sequence    = self.tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        consistency_score = self.consistency_score.score(input_sequence, decoded_sequences, device=device)
        adequacy_score    = self.adequacy_score.score(input_sequence, decoded_sequences, device=device)
        fluency_score     = self.fluency_score.score(decoded_sequences, device=device)
        diversity_score   = self.diversity_score.rank(input_sequence, decoded_sequences, device=device)
        
        scores = [[consistency_score[s], adequacy_score[s], fluency_score[s], diversity_score[s]] for s in decoded_sequences]
        
        indexes = sorted(
            range(len(decoded_sequences)),
            key=lambda i: (self.gen_params.lambda_consistency * scores[i][0] + self.gen_params.lambda_adequacy * scores[i][1] + \
                self.gen_params.lambda_fluency * scores[i][2] + self.gen_params.lambda_diversity * scores[i][3]), 
            reverse=True
        )

        sents = torch.cat([generated_outputs.sequences[i].unsqueeze(0) for i in indexes], dim=0)
        probs = torch.cat([probs[i] for i in indexes], dim=0)
        scores = torch.cat([torch.tensor(scores[i]).unsqueeze(0).to(device) for i in indexes], dim=0)

        return {'sequences': sents, 'probs': probs, 'scores': scores}

