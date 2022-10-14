import os
import torch
import src.exp_ner.data.datasets as ds
import src.exp_ner.modeling.nets as nets
import src.commons.utilities as utils
import src.commons.globals as glb
import src.exp_ner.modeling.trainer as trainer

from transformers import AutoConfig, AutoTokenizer


def prepare_model_config(args):
    config = AutoConfig.from_pretrained(args.config.config_name_or_path)

    config.model_name_or_path = args.model.model_name_or_path
    config.pretrained_frozen = args.model.pretrained_frozen
    config.num_labels = len(args.data.label_scheme)
    config.label_pad_token_id = args.tokenizer.label_pad_token_id
    config.output_hidden_states = False
    config.output_attentions = False

    if args.model.model_class == 'ner_devlin':
        config.output_attentions = False
        config.lstm_dropout = args.model.lstm_dropout
        config.lstm_layers = args.model.lstm_layers
        config.lstm_bidirectional = args.model.lstm_bidirectional
        config.use_lstm = args.model.use_lstm

    elif args.model.model_class == 'ner':
        pass

    else:
        raise NotImplementedError(f"Unexpected model name: {args.model.model_class}")

    return config


def get_model_class(model_name):
    if model_name == 'ner':
        model_class = nets.NERModel

    elif model_name == 'ner_devlin':
        model_class = nets.NERDevlinModel

    else:
        raise NotImplementedError(f'Unknown model name: {model_name}')

    return model_class


def main(args):
    config = prepare_model_config(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer.tokenizer_name_or_path, 
        do_lower_case=args.tokenizer.do_lowercase,
        use_fast=args.tokenizer.use_fast
    )

    dataloaders = ds.get_dataloaders(args, tokenizer)

    print(f"[LOG] Reading dataset from '{args.data.directory.replace(glb.PROJ_DIR, '$PROJECT')}'")
    for split in dataloaders:
        print(" [{: >8}]: {:>6,d}, ({})".format(split.upper(), len(dataloaders[split].dataset), dataloaders[split].dataset.dataset_file))
    
    print("[LOG] {}".format('=' * 40))
    print()

    model = get_model_class(args.model.model_class)(config)

    if args.mode == 'train':
        if args.model.resume_from_checkpoint:
            if os.path.exists(args.experiment.checkpoint_dir):
                print(f"Loading model from pretrained checkpoint at {args.experiment.checkpoint_dir}")
                model = get_model_class(args.model.model_class).from_pretrained(args.experiment.checkpoint_dir)
                model.to(args.optim.device)
            else:
                print(f"[WARNING] No checkpoint found in {args.experiment.checkpoint_dir} and will train the model from scratch...")

        stats, f1, global_step = trainer.train(args, model, dataloaders)
        trainer.print_stats(stats, args.data.label_scheme)

        print(f"\nBest dev F1: {f1:.3f}")
        print(f"Best global step: {global_step}")

    # Load the best checkpoint according to dev
    if os.path.exists(args.experiment.checkpoint_dir):
        print(f"Loading model from pretrained checkpoint at {args.experiment.checkpoint_dir}")
        model = get_model_class(args.model.model_class).from_pretrained(args.experiment.checkpoint_dir)
        model.to(args.optim.device)
    else:
        print(f"[WARNING] No checkpoint found in {args.experiment.checkpoint_dir}.")

    if utils.input_with_timeout("Do you want to evaluate the model? [y/n]:", 3, "y").strip() == 'y':
        # Perform evaluation over the dev and test sets with the best checkpoint
        for split in dataloaders.keys():
            if split == 'train' or split == 'dev':
                continue

            stats = trainer.predict(args, model, dataloaders[split])
            # torch.save(stats, os.path.join(args.experiment.output_dir, f'{split}_best_preds.bin'))

            f1, prec, recall = stats.metrics(args.data.label_scheme)
            loss, _, _ = stats.loss()
            ner_loss, _, _ = stats.loss(loss_type='ner')
            lm_loss, _, _ = stats.loss(loss_type='lm')

            report = stats.get_classification_report(args.data.label_scheme)
            classes = sorted(set([label[2:] for label in args.data.label_scheme if label != 'O']))

            print(f"\n********** {split.upper()} RESULTS **********\n")
            print('\t'.join(["LLoss", "NLoss", "Loss"] + ["Prec", "Recall", "F1"]), end='\n')
            print('\t'.join([f"{l:.4f}" for l in [lm_loss, ner_loss, loss, prec, recall, f1]]), end='\t')
            print()

            if utils.input_with_timeout("Print class-level results? [y/n]:", 5, "n").strip() == 'y':
                stats.print_classification_report(report=report)
                
        print()

if __name__ == '__main__':
    main()

