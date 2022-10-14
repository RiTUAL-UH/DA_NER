import os
import math
import torch
import numpy as np
import src.commons.utilities as utils

from tqdm.auto import tqdm, trange
from transformers import get_linear_schedule_with_warmup

from src.exp_lm.main import accelerator, logger


def get_optimizer(args, model):
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                      betas=(args.beta_1, args.beta_2), eps=args.adam_epsilon)

    return optimizer


def print_stats(stats):
    for stat in stats:
        epoch_i = stat['train']['epoch']
        train_loss = round(stat['train']['loss'], 3)
        train_ppl = round(stat['train']['ppl'], 3)
        train_acc = round(stat['train']['acc'], 3)
        dev_loss = round(stat['dev']['loss'], 3)
        dev_ppl = round(stat['dev']['ppl'], 3)
        dev_acc = round(stat['dev']['acc'], 3)

        logger.info("  Epoch {: >2} - [TRAIN] loss: {:.3f} PPL: {:.3f} acc: {:.3f} [DEV] loss: {:.3f} PPL: {:.3f} acc: {:.3f}".format(epoch_i, train_loss, train_ppl, train_acc, dev_loss, dev_ppl, dev_acc))


def pretrain(args, model, dataloaders):
    oargs = args.optim

    optimizer= get_optimizer(oargs, model)

    model, optimizer, dataloaders['labeled']['src']['train'], dataloaders['labeled']['src']['dev'], dataloaders['labeled']['tgt']['train'], dataloaders['labeled']['tgt']['dev'] = accelerator.prepare(
        model, optimizer, dataloaders['labeled']['src']['train'], dataloaders['labeled']['src']['dev'], dataloaders['labeled']['tgt']['train'], dataloaders['labeled']['tgt']['dev']
    )

    labled_src_dataloaders = {'train': dataloaders['labeled']['src']['train'], 'dev': dataloaders['labeled']['src']['dev']}
    labled_tgt_dataloaders = {'train': dataloaders['labeled']['tgt']['train'], 'dev': dataloaders['labeled']['tgt']['dev']}

    num_training_samples = min(len(labled_src_dataloaders['train']), len(labled_tgt_dataloaders['train']))

    if oargs.max_steps > 0:
        t_total = oargs.max_steps
        oargs.num_train_epochs = oargs.max_steps // (num_training_samples // oargs.gradient_accumulation_steps) + 1
    else:
        t_total = num_training_samples // oargs.gradient_accumulation_steps * oargs.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=oargs.warmup_steps, num_training_steps=t_total)

    epoch_i, epoch_loss, epoch_ppl, epoch_acc, global_step = 0, 0, 0, 0, 0
    stats = []

    progress_desc = "Epoch [{}/{}] (Loss: {:.3f} PPL: {:.3f} Acc: {:.3f})"
    progress_bar = trange(t_total, desc=progress_desc.format(epoch_i, oargs.num_train_epochs, epoch_loss, epoch_ppl, epoch_acc), disable=not accelerator.is_local_main_process)

    for epoch_i in range(1, oargs.num_train_epochs + 1):
        epoch_stat = {}

        for split in ['train', 'dev']:
            if split == 'train':
                model.train()
                model.zero_grad()
            else:
                model.eval()

            epoch_loss, epoch_ppl, epoch_acc = 0, 0, 0
            # =================================================================================================
            for step, (src_batch, tgt_batch) in enumerate(zip(labled_src_dataloaders[split], labled_tgt_dataloaders[split])):
                outputs = model(src_batch, tgt_batch)

                loss = outputs['loss'] / oargs.gradient_accumulation_steps

                epoch_loss += outputs['loss'].item()
                epoch_ppl  += outputs['gen_loss'].item()
                epoch_acc  += outputs['dis_accu'].item()
                
                if split == 'train':
                    accelerator.backward(loss)

                    if (step + 1) % oargs.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), oargs.max_grad_norm)

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        model.zero_grad()
                        global_step += 1
                        progress_bar.update(1)

                    if oargs.max_steps > 0 and global_step > oargs.max_steps:
                        break

            epoch_loss = epoch_loss / (step + 1)
            epoch_ppl  = math.exp(epoch_ppl / (step + 1))
            epoch_acc  = epoch_acc / (step + 1)
            # =================================================================================================
            epoch_stat[split] = {'epoch': epoch_i, 'loss': epoch_loss, 'ppl': epoch_ppl, 'acc': epoch_acc}
    
        stats.append(epoch_stat)

        progress_bar.set_description(progress_desc.format(epoch_i, oargs.num_train_epochs, epoch_stat['dev']['loss'], epoch_stat['dev']['ppl'], epoch_stat['dev']['acc']), refresh=True)

    progress_bar.close()

    print_stats(stats)
    
    os.makedirs(args.experiment.checkpoint_dir, exist_ok=True)
    logger.info(f"  Saving model to the directory at {args.experiment.checkpoint_dir}...")
    accelerator.wait_for_everyone()
    model_to_save = accelerator.unwrap_model(model)
    model_to_save.save_pretrained(args.experiment.checkpoint_dir, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))


def train(args, model, dataloaders):
    oargs = args.optim

    optimizer_G = get_optimizer(oargs, model.t5)
    optimizer_D = get_optimizer(oargs, model.disc)

    model, optimizer_G, optimizer_D, dataloaders['labeled']['src']['train'], dataloaders['labeled']['src']['dev'], dataloaders['labeled']['tgt']['train'], dataloaders['labeled']['tgt']['dev'], dataloaders['unlabeled']['src']['train'], dataloaders['unlabeled']['src']['dev'], dataloaders['unlabeled']['tgt']['train'], dataloaders['unlabeled']['tgt']['dev'] = accelerator.prepare(
        model, optimizer_G, optimizer_D, dataloaders['labeled']['src']['train'], dataloaders['labeled']['src']['dev'], dataloaders['labeled']['tgt']['train'], dataloaders['labeled']['tgt']['dev'], dataloaders['unlabeled']['src']['train'], dataloaders['unlabeled']['src']['dev'], dataloaders['unlabeled']['tgt']['train'], dataloaders['unlabeled']['tgt']['dev']
    )

    labled_src_dataloaders   = {'train': dataloaders['labeled']['src']['train'], 'dev': dataloaders['labeled']['src']['dev']}
    labled_tgt_dataloaders   = {'train': dataloaders['labeled']['tgt']['train'], 'dev': dataloaders['labeled']['tgt']['dev']}
    unlabled_src_dataloaders = {'train': dataloaders['unlabeled']['src']['train'], 'dev': dataloaders['unlabeled']['src']['dev']}
    unlabled_tgt_dataloaders = {'train': dataloaders['unlabeled']['tgt']['train'], 'dev': dataloaders['unlabeled']['tgt']['dev']}

    num_training_samples = min(len(labled_src_dataloaders['train']), len(labled_tgt_dataloaders['train']), len(unlabled_src_dataloaders['train']), len(unlabled_tgt_dataloaders['train']))

    if oargs.max_steps > 0:
        t_total = oargs.max_steps
        oargs.num_train_epochs = oargs.max_steps // (num_training_samples // oargs.gradient_accumulation_steps)
    else:
        t_total = num_training_samples // oargs.gradient_accumulation_steps * oargs.num_train_epochs

    scheduler_G = get_linear_schedule_with_warmup(optimizer_G, num_warmup_steps=oargs.warmup_steps, num_training_steps=t_total)
    scheduler_D = get_linear_schedule_with_warmup(optimizer_D, num_warmup_steps=oargs.warmup_steps, num_training_steps=t_total)

    epoch_i, epoch_loss, epoch_ppl, epoch_acc, global_step = 0, 0, 0, 0, 0
    stats = []

    progress_desc = "Epoch [{}/{}] (Loss: {:.3f} PPL: {:.3f} Acc: {:.3f})"
    progress_bar = trange(t_total, desc=progress_desc.format(epoch_i, oargs.num_train_epochs, epoch_loss, epoch_ppl, epoch_acc), disable=not accelerator.is_local_main_process)
    
    for epoch_i in range(1, oargs.num_train_epochs + 1):
        epoch_stat = {}
        
        for split in ['train', 'dev']:
            if split == 'train':
                model.train()
                model.zero_grad()
            else:
                model.eval()
            
            epoch_loss, epoch_ppl, epoch_acc = 0, 0, 0
            # =================================================================================================
            zipped_dataloaders = zip(labled_src_dataloaders[split], labled_tgt_dataloaders[split], unlabled_src_dataloaders[split], unlabled_tgt_dataloaders[split])
            for step, (labeled_src_batch, labeled_tgt_batch, unlabeled_src_batch, unlabeled_tgt_batch) in enumerate(zipped_dataloaders):
                pg_outputs = model(labeled_src_batch, labeled_tgt_batch, mode='pg')
                cr_outputs = model(unlabeled_src_batch, unlabeled_tgt_batch, mode='cr')

                loss_G = (pg_outputs['gen_loss'] + cr_outputs['gen_loss']) / oargs.gradient_accumulation_steps
                loss_D = (pg_outputs['dis_loss'] + cr_outputs['dis_loss']) / oargs.gradient_accumulation_steps

                epoch_loss += (pg_outputs['loss'] + cr_outputs['loss']).item() / 2
                epoch_ppl  += (pg_outputs['gen_loss'] + cr_outputs['gen_loss']).item() / 2
                epoch_acc  += (pg_outputs['dis_accu'] + cr_outputs['dis_accu']).item() / 2

                if split == 'train':
                    accelerator.backward(loss_D, retain_graph=True)

                    if (step + 1) % oargs.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), oargs.max_grad_norm)

                        optimizer_D.step()
                        scheduler_D.step()
                        optimizer_D.zero_grad()
                        model.zero_grad()

                    accelerator.backward(loss_G)

                    if (step + 1) % oargs.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), oargs.max_grad_norm)

                        optimizer_G.step()
                        scheduler_G.step()
                        optimizer_G.zero_grad()
                        model.zero_grad()
                        global_step += 1
                        progress_bar.update(1)

                    if oargs.max_steps > 0 and global_step > oargs.max_steps:
                        break

            epoch_loss = epoch_loss / (step + 1)
            epoch_ppl  = math.exp(epoch_ppl / (step + 1))
            epoch_acc  = epoch_acc / (step + 1)
            # =================================================================================================
            epoch_stat[split] = {'epoch': epoch_i, 'loss': epoch_loss, 'ppl': epoch_ppl, 'acc': epoch_acc}

        stats.append(epoch_stat)

        progress_bar.set_description(progress_desc.format(epoch_i, oargs.num_train_epochs, epoch_stat['dev']['loss'], epoch_stat['dev']['ppl'], epoch_stat['dev']['acc']), refresh=True)
    
    progress_bar.close()

    print_stats(stats)
    
    os.makedirs(args.experiment.checkpoint_dir, exist_ok=True)
    logger.info(f"  Saving model to the directory at {args.experiment.checkpoint_dir}...")
    accelerator.wait_for_everyone()
    model_to_save = accelerator.unwrap_model(model)
    model_to_save.save_pretrained(args.experiment.checkpoint_dir, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))


def evaluate(args, model, dataloaders):
    model, dataloaders['labeled']['src']['dev'], dataloaders['labeled']['src']['test'], dataloaders['labeled']['tgt']['dev'], dataloaders['labeled']['tgt']['test'], dataloaders['unlabeled']['src']['dev'], dataloaders['unlabeled']['src']['test'], dataloaders['unlabeled']['tgt']['dev'], dataloaders['unlabeled']['tgt']['test'] = accelerator.prepare(
        model, dataloaders['labeled']['src']['dev'], dataloaders['labeled']['src']['test'], dataloaders['labeled']['tgt']['dev'], dataloaders['labeled']['tgt']['test'], dataloaders['unlabeled']['src']['dev'], dataloaders['unlabeled']['src']['test'], dataloaders['unlabeled']['tgt']['dev'], dataloaders['unlabeled']['tgt']['test']
    )

    labled_src_dataloaders   = {'dev': dataloaders['labeled']['src']['dev'], 'test': dataloaders['labeled']['src']['test']}
    labled_tgt_dataloaders   = {'dev': dataloaders['labeled']['tgt']['dev'], 'test': dataloaders['labeled']['tgt']['test']}
    unlabled_src_dataloaders = {'dev': dataloaders['unlabeled']['src']['dev'], 'test': dataloaders['unlabeled']['src']['test']}
    unlabled_tgt_dataloaders = {'dev': dataloaders['unlabeled']['tgt']['dev'], 'test': dataloaders['unlabeled']['tgt']['test']}
    
    model.eval()
    
    for split in ['dev', 'test']:
        t_total = min(len(labled_src_dataloaders[split]), len(labled_tgt_dataloaders[split]), len(unlabled_src_dataloaders[split]), len(unlabled_tgt_dataloaders[split]))

        progress_desc = "{} [{}/{}])"
        progress_bar = trange(t_total, desc=progress_desc.format(split.upper(), 0, t_total), disable=not accelerator.is_local_main_process)

        epoch_loss, pg_loss, cr_loss, epoch_ppl, epoch_acc = 0, 0, 0, 0, 0
        # =================================================================================================
        zipped_dataloaders = zip(labled_src_dataloaders[split], labled_tgt_dataloaders[split], unlabled_src_dataloaders[split], unlabled_tgt_dataloaders[split])
        for step, (labeled_src_batch, labeled_tgt_batch, unlabeled_src_batch, unlabeled_tgt_batch) in enumerate(zipped_dataloaders):
            pg_outputs = model(labeled_src_batch, labeled_tgt_batch, mode='pg')
            cr_outputs = model(unlabeled_src_batch, unlabeled_tgt_batch, mode='cr')

            epoch_loss += (pg_outputs['loss'] + cr_outputs['loss']).item() / 2
            epoch_ppl  += (pg_outputs['gen_loss'] + cr_outputs['gen_loss']).item() / 2
            epoch_acc  += (pg_outputs['dis_accu'] + cr_outputs['dis_accu']).item() / 2
            pg_loss  += pg_outputs['gen_loss'].item()
            cr_loss += cr_outputs['gen_loss'].item()

            progress_bar.update(1)
            progress_bar.set_description(progress_desc.format(split.upper(), step + 1, t_total), refresh=True)

        epoch_loss = epoch_loss / (step + 1)
        epoch_ppl  = math.exp(epoch_ppl / (step + 1))
        epoch_acc  = epoch_acc / (step + 1)
        pg_loss    = pg_loss / (step + 1)
        cr_loss    = cr_loss / (step + 1)
        # =================================================================================================
        progress_bar.close()

        logger.info('  [{: >4}] loss: {:.3f} pg: {:.3f} cr: {:.3f} ppl: {:.3f} acc: {:.3f}'.format(split.upper(), epoch_loss, pg_loss, cr_loss, epoch_ppl, epoch_acc))


def generate(args, tokenizer, task_prefix, decoder, dataloader, filepath):
    decoder.model, dataloader = accelerator.prepare(decoder.model, dataloader)

    progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)

    save_data = [[] for _ in range(decoder.gen_params.num_return_sequences)]

    for _, batch in enumerate(dataloader):
        with torch.no_grad():
            generated_sequences, generated_scores = [], []
            for i in range(len(batch['input_ids'])):
                generated_outputs = decoder.generate(task_prefix, batch['input_ids'][i:i+1], attention_mask=batch['attention_mask'][i:i+1])

                # gather generated outputs from different devices
                generated_sequences = accelerator.pad_across_processes(generated_outputs['sequences'], dim=1, pad_index=tokenizer.pad_token_id)
                generated_sequences = accelerator.gather(generated_sequences).cpu().numpy()
                if isinstance(generated_sequences, tuple):
                    generated_sequences = generated_sequences[0]

                # gather 
                generated_scores = accelerator.gather(generated_outputs['scores']).cpu().numpy() if 'scores' in generated_outputs else None

                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"][i:i+1], dim=1, pad_index=tokenizer.pad_token_id)
                labels = accelerator.gather(labels).cpu().numpy()
                if args.tokenizer.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                decoded_preds = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = utils.postprocess_text(decoded_preds, decoded_labels)

                print(decoder.gen_params.num_return_sequences)

                for j in range(len(decoded_labels)):
                    for k in range(j * decoder.gen_params.num_return_sequences, (j + 1) * decoder.gen_params.num_return_sequences):
                        if generated_scores is not None:
                            save_data[k - j * decoder.gen_params.num_return_sequences].append({'tokens': decoded_preds[k], 'labels': decoded_labels[j], 'consistency': str(generated_scores[k][0]), \
                                'adequacy': str(generated_scores[k][1]), 'fluency': str(generated_scores[k][2]), 'diversity': str(generated_scores[k][3])})
                        else:
                            save_data[k - j * decoder.gen_params.num_return_sequences].append({'tokens': decoded_preds[k], 'labels': decoded_labels[j]})
                
            # Update progress bar
            progress_bar.update(1)
    
    progress_bar.close()

    save_data = utils.flatten(save_data)
    
    save_path = os.path.join(args.experiment.pseudo_dir, filepath, 'train.json')
    utils.save_as_json(save_path, save_data)
    logger.info(f"  Saved predictions in {save_path}.\n")
