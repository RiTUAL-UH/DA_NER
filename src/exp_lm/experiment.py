import os
import re

import src.exp_lm.data.datasets as ds
import src.exp_lm.modeling.trainer as trainer

from collections import namedtuple
from transformers import AutoConfig, AutoTokenizer
from src.exp_lm.modeling.models import DAT5PreTrainedModel, DAT5ForTextTransfer
from src.exp_lm.generation_utils.decoders import Decoder, ConstrainedDecoder, ConstrainedDecoderWithSelection

from src.exp_lm.main import logger


def prepare_config_and_tokenizer(args):
    config = AutoConfig.from_pretrained(args.config.model_name_or_path)
    config.model_name_or_path = args.config.model_name_or_path
    config.coef_params = namedtuple('coef_params', args.model.coef_params.keys())(*args.model.coef_params.values()) if hasattr(args.model, 'coef_params') else None
    config.gen_params = namedtuple('gen_params', args.model.generation.keys())(*args.model.generation.values()) if hasattr(args.model, 'generation') else None

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer.model_name_or_path, use_fast=args.tokenizer.use_fast, model_max_length=args.tokenizer.model_max_length)
    if len(args.tokenizer.new_tokens) > 0:
        tokenizer.new_tokens = args.tokenizer.new_tokens
        tokenizer.add_tokens(args.tokenizer.new_tokens)
        config.vocab_size = len(tokenizer)
        logger.info(f"  Add new tokens. New vocab size: {len(tokenizer)}")
        assert len(tokenizer) <= config.vocab_size
    
    # tokenize task prefixes and remove the eos_token_id: [1, seq_len, 1]
    config.src_task_prefix = tokenizer(args.config.src_task_prefix)['input_ids'][:-1] if hasattr(args.config, 'src_task_prefix') else None
    config.tgt_task_prefix = tokenizer(args.config.tgt_task_prefix)['input_ids'][:-1] if hasattr(args.config, 'tgt_task_prefix') else None
    
    return config, tokenizer


def get_model_class(model_args):
    if model_args.model_class == 'DAT5PreTrainedModel':
        model_class = DAT5PreTrainedModel
    elif model_args.model_class == 'DAT5ForTextTransfer':
        model_class = DAT5ForTextTransfer
    else:
        raise ValueError(f'{model_args.model_class} is not supported.')
    return model_class


def get_decoder_class(generation_args):
    if generation_args.decoder_class == 'Decoder':
        decoder_class = Decoder
    elif generation_args.decoder_class == 'ConstrainedDecoder':
        decoder_class = ConstrainedDecoder
    elif generation_args.decoder_class == 'ConstrainedDecoderWithSelection':
        decoder_class = ConstrainedDecoderWithSelection
    else:
        raise ValueError(f'{generation_args.decoder_class} is not supported.')
    return decoder_class


def main(args):
    config, tokenizer = prepare_config_and_tokenizer(args)
    
    if args.mode == 'pretrain':
        dataloaders = {
            'labeled': {
                'src': ds.get_dataloaders(args.data.labeled.source, args.tokenizer, args.optim, args.data.directory, args.data.labeled.columns, tokenizer),
                'tgt': ds.get_dataloaders(args.data.labeled.target, args.tokenizer, args.optim, args.data.directory, args.data.labeled.columns, tokenizer)
            }
        }

        model = get_model_class(args.model)(config)

        if args.model.resume_from_checkpoint:
            logger.info(f"  Loading model from a pretrained checkpoint at {args.experiment.checkpoint_dir}\n")
            model = model.from_pretrained(args.experiment.checkpoint_dir)
        
        model.prepare_params_and_task_prefix(config)
    
        logger.info("  Start pretraining...")
        trainer.pretrain(args, model, dataloaders)
        logger.info("  Done pretraining!\n")

    else:
        dataloaders = {
            'labeled': {
                'src': ds.get_dataloaders(args.data.labeled.source, args.tokenizer, args.optim, args.data.directory, args.data.labeled.columns, tokenizer),
                'tgt': ds.get_dataloaders(args.data.labeled.target, args.tokenizer, args.optim, args.data.directory, args.data.labeled.columns, tokenizer)
            },
            'unlabeled': {
                'src': ds.get_dataloaders(args.data.unlabeled.source, args.tokenizer, args.optim, args.data.directory, args.data.unlabeled.columns, tokenizer),
                'tgt': ds.get_dataloaders(args.data.unlabeled.target, args.tokenizer, args.optim, args.data.directory, args.data.unlabeled.columns, tokenizer)
            }
        }

        if args.mode == 'train':
            model = get_model_class(args.model)(config)

            if args.model.resume_from_pretraining and not args.model.resume_from_checkpoint:
                logger.info(f"  Loading model from a pretrained checkpoint at {args.experiment.pretraining_dir}\n")
                model = model.from_pretrained(args.experiment.pretraining_dir)
            elif args.model.resume_from_checkpoint:
                logger.info(f"  Loading model from a pretrained checkpoint at {args.experiment.checkpoint_dir}\n")
                model = model.from_pretrained(args.experiment.checkpoint_dir)
            
            model.prepare_params_and_task_prefix(config)

            if args.model.freeze_classifier:
                logger.info("  Freezing classifer...")
                model.freeze_classifier()

            logger.info("  Start training...")
            trainer.train(args, model, dataloaders)
            logger.info("  Done training!\n")
        else:
            logger.info("  Skipping training!\n")
            
            model = get_model_class(args.model)(config)
            # Load the best checkpoint according to dev
            if os.path.exists(args.experiment.checkpoint_dir):
                logger.info(f"  Loading model from a pretrained checkpoint at {args.experiment.checkpoint_dir}\n")
                model = model.from_pretrained(args.experiment.checkpoint_dir)
            
            model.prepare_params_and_task_prefix(config)

            if args.mode == 'eval':
                logger.info("  Start evaluating...")
                # Perform evaluation over the dev and test sets with the best checkpoint
                trainer.evaluate(args, model, dataloaders)
            
            if args.mode == 'generate':
                src_path, tgt_path = re.split('-|\.', args.experiment.id)[-2:]

                # ========================================================================================
                decoder = get_decoder_class(args.model.generation)(model, args.model.generation, tokenizer)

                if isinstance(decoder, ConstrainedDecoder) or isinstance(decoder, ConstrainedDecoderWithSelection):
                    logger.info("  Collecting all candidates...")
                    decoder.collect_all_candidates(dataloaders['unlabeled']['tgt'])
                    
                    logger.info("  Building trie...")
                    decoder.build_trie(dataloaders['unlabeled']['src']['train'].dataset.inputs)

                logger.info("  Start generating...")
                trainer.generate(args, tokenizer, model.src_hard_task_prefix, decoder, dataloaders['unlabeled']['src']['train'], src_path + '/' + tgt_path)


if __name__ == '__main__':
    main()

