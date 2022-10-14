import os
import torch
import src.commons.utilities as utils

from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup
from src.exp_ner.evaluation.metric import EpochStats


def get_optimizer(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      betas=(args.beta_1, args.beta_2),
                      eps=args.adam_epsilon)
    return optimizer


def track_best_model(args, model, dev_stats, best_f1, best_step, global_step):
    curr_f1, _, _ = dev_stats.metrics(args.data.label_scheme)
    if best_f1 > curr_f1:
        return best_f1, best_step

    # Save model checkpoint
    os.makedirs(args.experiment.checkpoint_dir, exist_ok=True)
    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.experiment.checkpoint_dir)
    meta = {
        'args': args,
        'f1': curr_f1,
        'global_step': global_step
    }
    torch.save(meta, os.path.join(args.experiment.checkpoint_dir, "training_meta.bin"))
    return curr_f1, global_step


def print_stats(stats, label_scheme):
    for i in range(len(stats['train'])):
        print(f"Epoch {str(i+1):>2s} -", end=" ")
        for split in ['train', 'dev']:
            epoch_stats = stats[split][i]
            f1, _, _ = epoch_stats.metrics(label_scheme)
            loss = sum(epoch_stats.losses) / len(epoch_stats.losses)
            print(f"[{split.upper()}] Loss: {loss:.3f} F1: {f1 :6.3f}", end=' ')
        print()
    print()


def train(args, model, dataloaders):
    oargs = args.optim

    if oargs.max_steps > 0:
        t_total = oargs.max_steps
        oargs.num_train_epochs = oargs.max_steps // (len(dataloaders['train']) // oargs.gradient_accumulation_steps) + 1
    else:
        t_total = len(dataloaders['train']) // oargs.gradient_accumulation_steps * oargs.num_train_epochs

    optimizer = get_optimizer(oargs, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=oargs.warmup_steps, num_training_steps=t_total)

    model.to(args.optim.device)

    # multi-gpu training
    if oargs.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.zero_grad()

    epoch_i, best_f1, best_step, global_step = 0, 0, 0, 0

    stats = {'train': [], 'dev': []}

    # Only show the progress bar once on each machine.
    progress_desc = "Epoch [{}/{}] (Dev F1: {:.3f} at step {})"
    progress_bar = trange(t_total, desc=progress_desc.format(epoch_i, args.optim.num_train_epochs, best_f1, best_step))

    for epoch_i in range(args.optim.num_train_epochs):
        for split in ['train', 'dev']:
            epoch_stats = EpochStats()
            # ====================================================================
            for step, batch in enumerate(dataloaders[split]):
                if split == 'train':
                    model.train()
                    model.zero_grad()
                else:
                    model.eval()

                for field in batch.keys():
                    if batch[field] is not None:
                        batch[field] = batch[field].to(oargs.device)

                outputs = model(**batch, wrap_scalars=oargs.n_gpu > 1)
                loss, ner_loss, lm_loss = outputs[0]

                if oargs.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    ner_loss = ner_loss.mean()
                    lm_loss = lm_loss.mean()

                if oargs.gradient_accumulation_steps > 1:
                    loss = loss / oargs.gradient_accumulation_steps
                    ner_loss = ner_loss / oargs.gradient_accumulation_steps
                    lm_loss = lm_loss / oargs.gradient_accumulation_steps

                epoch_stats.step(scores=outputs[1], target=batch['labels'], mask=batch['label_mask'],
                                 loss=loss.item(), ner_loss=ner_loss.item(), lm_loss=lm_loss.item())

                if split == 'train':
                    loss.backward()

                    if (step + 1) % oargs.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), oargs.max_grad_norm)

                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1
                        progress_bar.update(1)

                if oargs.max_steps > 0 and global_step > oargs.max_steps:
                    progress_bar.close()
                    break
            # ====================================================================
            stats[split].append(epoch_stats)

            if split == 'dev':
                best_f1, best_step = track_best_model(args, model, epoch_stats, best_f1, best_step, global_step)
        
        # Update progress bar description
        progress_bar.set_description(progress_desc.format(epoch_i + 1, args.optim.num_train_epochs, best_f1, best_step), refresh=True)

        if oargs.max_steps > 0 and global_step > oargs.max_steps:
            progress_bar.close()
            break
    
    progress_bar.close()

    return stats, best_f1, best_step


def predict(args, model, dataloader):
    model.eval()
    stats = EpochStats()

    oargs = args.optim

    # multi-gpu evaluate
    if oargs.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            for field in batch:
                if batch[field] is not None:
                    batch[field] = batch[field].to(oargs.device)

            outputs = model(**batch, wrap_scalars=oargs.n_gpu > 1)
            loss, ner_loss, lm_loss = outputs[0]

            if oargs.n_gpu > 1:
                ner_loss = ner_loss.mean()
                lm_loss = lm_loss.mean()
                loss = loss.mean()

            stats.step(scores=outputs[1], target=batch['labels'], mask=batch['label_mask'],
                    loss=loss.item(), ner_loss=ner_loss.item(), lm_loss=lm_loss.item())
                    
    return stats

