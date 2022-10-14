import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5PreTrainedModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from undecorated import undecorated
from types import MethodType


class DAT5Base(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # preload weights for generator
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)
        if config.vocab_size != self.t5.config.vocab_size:
            self.t5.resize_token_embeddings(config.vocab_size)

        # preload weights for style classifier
        self.disc = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)
        self.disc = self._remove_embedding_and_encoder_layers(self.disc)

    def _remove_embedding_and_encoder_layers(self, model):
        if hasattr(model, 'shared'):
            del model.shared
        if hasattr(model, 'encoder'):
            del model.encoder
        return model
    
    def prepare_params_and_task_prefix(self, config):
        # hyper-parameters for coefficients and generation
        self.coef_params = config.coef_params if hasattr(config, 'coef_params') else None
        self.gen_params  = config.gen_params if hasattr(config, 'gen_params') else None

        src_task_prefix = torch.tensor(config.src_task_prefix).view(1, -1, 1) if hasattr(config, 'src_task_prefix') else None
        tgt_task_prefix = torch.tensor(config.tgt_task_prefix).view(1, -1, 1) if hasattr(config, 'tgt_task_prefix') else None

        # create hard_token_ids as task prefix using token indexes
        self.src_hard_task_prefix = src_task_prefix.view(1, -1) if src_task_prefix is not None else None
        self.tgt_hard_task_prefix = tgt_task_prefix.view(1, -1) if tgt_task_prefix is not None else None

        # create soft_token_ids as task prefix using token embeddings
        self.src_soft_task_prefix = torch.zeros((1, src_task_prefix.shape[1], self.t5.config.vocab_size)).scatter(-1, src_task_prefix, 1.0) if src_task_prefix is not None else None
        self.tgt_soft_task_prefix = torch.zeros((1, tgt_task_prefix.shape[1], self.t5.config.vocab_size)).scatter(-1, tgt_task_prefix, 1.0) if tgt_task_prefix is not None else None

    def freeze_classifier(self):
        for param in self.disc.parameters():
            param.requires_grad = False

    def pararephrase(self, task_prefix, batch):
        assert len(task_prefix.shape) == 2

        batch_size, device = batch['input_ids'].shape[0], batch['input_ids'].device

        task_input_ids = task_prefix.repeat(batch_size, 1).to(device)
        task_attention_mask = torch.ones_like(task_input_ids).to(device)
        task_encoder_outputs = self.t5.encoder(input_ids=task_input_ids, attention_mask=task_attention_mask)

        encoder_outputs = self.t5.encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

        concat_last_hidden_state = torch.cat([task_encoder_outputs[0], encoder_outputs[0]], dim=1)
        concat_encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(concat_last_hidden_state)
        concat_attention_mask = torch.cat([task_attention_mask, batch['attention_mask']], dim=1)

        outputs = self.t5(attention_mask=concat_attention_mask, encoder_outputs=concat_encoder_outputs, labels=batch['labels'])

        return {'loss': outputs.loss, 'hidden_state': encoder_outputs[0], 'attention_mask': batch['attention_mask']}

    def style_classification(self, hidden_state, attention_mask=None, labels=None):
        batch_size, device = hidden_state.shape[0], hidden_state.device

        # create `decoder_input_ids`
        decoder_input_ids = torch.tensor([self.disc.config.pad_token_id]).unsqueeze(0)
        decoder_input_ids = decoder_input_ids.repeat(batch_size, 1).to(device)

        # get decoder outputs
        decoder_outputs = self.disc.decoder(input_ids=decoder_input_ids, encoder_hidden_states=hidden_state, encoder_attention_mask=attention_mask)

        # get style representations
        sequence_output = decoder_outputs[0]

        # use the logits for the first token as sentence representation
        # select logits from 4727 (formal) and 15347 (informal) to calculate losses
        lm_logits = self.disc.lm_head(sequence_output).squeeze(1)
        selected_logits = lm_logits[:, [4727, 15347]]

        probs = F.softmax(selected_logits, dim=-1)

        # calculate losses and accuracy for style classification
        loss, accu = None, None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            labels = torch.full((batch_size, 1), 0 if labels == 'formal' else 1, dtype=torch.int64).to(device)
            loss = loss_fct(selected_logits.view(-1, selected_logits.size(-1)), labels.view(-1))
            accu = torch.sum(selected_logits.argmax(-1).view(-1) == labels.view(-1)) / batch_size * 100

        return {'loss': loss, 'accu': accu, 'probs': probs, 'style_reps': sequence_output}

    def forward(self, src_batch, tgt_batch=None):
        raise NotImplementedError('The AutoReconstructionModelBase class should never execute forward')


class DAT5PreTrainedModel(DAT5Base):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, src_batch, tgt_batch):
        """
        For pretraining, only paraphrase generation and style classification are involved to 
        train the generator and discrminator.
        """
        device = src_batch['input_ids'].device

        gen_loss = torch.tensor(0., device=device).float()
        dis_loss = torch.tensor(0., device=device).float()
        dis_accu = torch.tensor(0., device=device).float()

        src_gen_outs = self.pararephrase(self.src_hard_task_prefix, src_batch)
        tgt_gen_outs = self.pararephrase(self.tgt_hard_task_prefix, tgt_batch)

        src_gen_dis_outs = self.style_classification(src_gen_outs['hidden_state'], src_batch['attention_mask'], labels='formal')
        tgt_gen_dis_outs = self.style_classification(tgt_gen_outs['hidden_state'], tgt_batch['attention_mask'], labels='informal')

        gen_loss = (src_gen_outs['loss'] + tgt_gen_outs['loss']) / 2
        dis_loss = (src_gen_dis_outs['loss'] + tgt_gen_dis_outs['loss']) / 2
        dis_accu = (src_gen_dis_outs['accu'] + tgt_gen_dis_outs['accu']) / 2

        return {'loss': gen_loss + dis_loss, 'gen_loss': gen_loss, 'dis_loss': dis_loss, 'dis_accu': dis_accu}


class DAT5ForTextTransfer(DAT5Base):
    def __init__(self, config):
        super().__init__(config)
    
    def make_soft_mask(self, soft_input_ids):
        soft_attention_mask = torch.cumsum(soft_input_ids.argmax(-1) == self.t5.config.eos_token_id, dim=1)
        soft_attention_mask = (soft_attention_mask == 0).long().to(soft_input_ids.device)
        return soft_attention_mask
    
    def cycle_reconst(self, fw_task_prefix, bw_task_prefix, batch):
        assert len(fw_task_prefix.shape) == 2 and len(bw_task_prefix.shape) == 3

        batch_size, device = batch['input_ids'].shape[0], batch['input_ids'].device

        # forward paraphrase generation with hard task prefix
        fw_task_input_ids = fw_task_prefix.repeat(batch_size, 1).to(device)
        fw_task_attention_mask = torch.ones_like(fw_task_input_ids).to(device)
        fw_task_encoder_outputs = self.t5.encoder(input_ids=fw_task_input_ids, attention_mask=fw_task_attention_mask)

        fw_encoder_outputs = self.t5.encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

        fw_concat_last_hidden_state = torch.cat([fw_task_encoder_outputs[0], fw_encoder_outputs[0]], dim=1)
        fw_concat_encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(fw_concat_last_hidden_state)
        fw_concat_encoder_attention_mask = torch.cat([fw_task_attention_mask, batch['attention_mask']], dim=1)

        # enable differentiable decoding
        generate_with_grad = undecorated(self.t5.generate)
        self.t5.generate_with_grad = MethodType(generate_with_grad, self.t5)

        # translate sentences to the target style
        generated_outputs = self.t5.generate_with_grad(
            encoder_outputs=fw_concat_encoder_outputs,
            attention_mask=fw_concat_encoder_attention_mask,
            do_sample=self.gen_params.do_sample,
            max_length=self.gen_params.max_length,
            top_k=self.gen_params.top_k,
            top_p=self.gen_params.top_p,
            early_stopping=self.gen_params.early_stopping,
            num_return_sequences=1,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            forced_eos_token_id=True
        )

        # scores doesn't include the feature vector for the first token <PAD>
        probs = torch.stack(generated_outputs.scores, dim=1)

        # use gumbel softmax to approximate categorical distributions
        soft_input_ids = F.gumbel_softmax(probs)
        soft_attention_mask = self.make_soft_mask(soft_input_ids)

        assert soft_input_ids.requires_grad == True

        # backward paraphrase generation with soft task prefix
        bw_task_input_ids = bw_task_prefix.repeat(batch_size, 1, 1).to(device)
        bw_task_attention_mask = torch.ones((batch_size, bw_task_input_ids.shape[1])).to(device)
        bw_task_inputs_embeds = torch.matmul(bw_task_input_ids, self.t5.shared.weight)
        bw_task_encoder_outputs = self.t5.encoder(inputs_embeds=bw_task_inputs_embeds, attention_mask=bw_task_attention_mask)

        bw_inputs_embeds = torch.matmul(soft_input_ids, self.t5.shared.weight)
        bw_encoder_outputs = self.t5.encoder(inputs_embeds=bw_inputs_embeds, attention_mask=soft_attention_mask)

        # concatenate backward task representations and text representations
        bw_concat_last_hidden_state = torch.cat([bw_task_encoder_outputs[0], bw_encoder_outputs[0]], dim=1)
        bw_concat_encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(bw_concat_last_hidden_state)
        bw_concat_encoder_attention_mask = torch.cat([bw_task_attention_mask, soft_attention_mask], dim=1)

        # back-translate sentences to their original style
        outputs = self.t5(attention_mask=bw_concat_encoder_attention_mask, encoder_outputs=bw_concat_encoder_outputs, labels=batch['labels'])

        return {
            'loss': outputs.loss, 
            'fw_hidden_state': fw_encoder_outputs[0], 'fw_attention_mask': batch['attention_mask'],
            'bw_hidden_state': bw_encoder_outputs[0], 'bw_attention_mask': soft_attention_mask
        }

    def forward(self, src_batch, tgt_batch=None, mode='pg'):
        """
        For style transfer, paraphrase generation, cycle-consistent reconstruction, and style 
        classification are involved to train the generator and discriminator.
        """
        device = src_batch['input_ids'].device

        gen_loss = torch.tensor(0., device=device).float()
        dis_loss = torch.tensor(0., device=device).float()
        dis_accu = torch.tensor(0., device=device).float()

        # paraphrase generation with style classifcation
        if mode == 'pg':
            src_gen_outs = self.pararephrase(self.src_hard_task_prefix, src_batch)
            src_gen_dis_outs = self.style_classification(src_gen_outs['hidden_state'], src_batch['attention_mask'], labels='formal')

            gen_loss = self.coef_params.lambda_para * src_gen_outs['loss']
            dis_loss = self.coef_params.lambda_cls * src_gen_dis_outs['loss']
            dis_accu = src_gen_dis_outs['accu']

            if tgt_batch:
                tgt_gen_outs = self.pararephrase(self.tgt_hard_task_prefix, tgt_batch)
                tgt_gen_dis_outs = self.style_classification(tgt_gen_outs['hidden_state'], tgt_batch['attention_mask'], labels='informal')

                gen_loss = (gen_loss + self.coef_params.lambda_para * tgt_gen_outs['loss']) / 2
                dis_loss = (dis_loss + self.coef_params.lambda_cls * tgt_gen_dis_outs['loss']) / 2
                dis_accu = (dis_accu + tgt_gen_dis_outs['accu']) / 2

            return {'loss': gen_loss + dis_loss, 'gen_loss': gen_loss, 'dis_loss': dis_loss, 'dis_accu': dis_accu}

        # cycle-consistent reconstruction with style classifcation
        else:
            src_gen_outs = self.cycle_reconst(self.src_hard_task_prefix, self.tgt_soft_task_prefix, src_batch)
            src_gen_fw_dis_outs = self.style_classification(src_gen_outs['fw_hidden_state'], src_gen_outs['fw_attention_mask'], labels='formal')
            src_gen_bw_dis_outs = self.style_classification(src_gen_outs['bw_hidden_state'], src_gen_outs['bw_attention_mask'], labels='informal')

            gen_loss = self.coef_params.lambda_cycle * src_gen_outs['loss']
            dis_loss = self.coef_params.lambda_cls * (src_gen_fw_dis_outs['loss'] + src_gen_bw_dis_outs['loss']) / 2
            dis_accu = (src_gen_fw_dis_outs['accu'] + src_gen_bw_dis_outs['accu']) / 2

            if tgt_batch:
                tgt_gen_outs = self.cycle_reconst(self.tgt_hard_task_prefix, self.src_soft_task_prefix, tgt_batch)
                tgt_gen_fw_dis_outs = self.style_classification(tgt_gen_outs['fw_hidden_state'], tgt_gen_outs['fw_attention_mask'], labels='informal')
                tgt_gen_bw_dis_outs = self.style_classification(tgt_gen_outs['bw_hidden_state'], tgt_gen_outs['bw_attention_mask'], labels='formal')

                gen_loss = (gen_loss + self.coef_params.lambda_cycle * tgt_gen_outs['loss']) / 2
                dis_loss = (dis_loss + self.coef_params.lambda_cls * (tgt_gen_fw_dis_outs['loss'] + tgt_gen_bw_dis_outs['loss']) / 2 ) / 2
                dis_accu = (dis_accu + (tgt_gen_fw_dis_outs['accu'] + tgt_gen_bw_dis_outs['accu']) / 2 ) / 2

            return {'loss': gen_loss + dis_loss, 'gen_loss': gen_loss, 'dis_loss': dis_loss, 'dis_accu': dis_accu}

