"""Supervised fine-tuning on the item-format rx_counsel datasets.

One record = one chat turn: user = input_md/<split>/<id>.md, assistant =
output_items/<split>/<id>.json. Loss is computed on the assistant part only.
Defaults reproduce checkpoints/ax4_light_7b_sft_v3sel_best (training_args.bin):
lr 1e-5, 2 epochs, micro batch 1, grad accum 4, cosine_with_min_lr (min_lr_rate
0.1), warmup 3%, bf16, gradient checkpointing, ZeRO-2, YaRN x2 (32,768 tokens),
eval every 50 steps; the lowest eval_loss model is kept in <output_dir>/best
with BEST.json, and the final model in <output_dir>/last with LAST.json.

    torchrun --nproc_per_node 8 src/train_sft.py --data_dir <dataset> --output_dir <dir>
"""
import argparse
import glob
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data_dir', required=True, help='item-format dataset (input_md/, output_items/)')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--model_path', default=str(PROJECT_ROOT / 'models/AX-4.0-Light'))
    p.add_argument('--train_split', default='train')
    p.add_argument('--eval_split', default='val')
    p.add_argument('--max_len', type=int, default=32768, help='records longer than this are dropped')
    p.add_argument('--yarn_factor', type=float, default=2.0, help='0 disables rope scaling')
    p.add_argument('--attn', default='sdpa', choices=['sdpa', 'flash_attention_2', 'eager'])
    p.add_argument('--deepspeed', default=str(PROJECT_ROOT / 'configs/deepspeed/zero2.json'))
    p.add_argument('--learning_rate', type=float, default=1e-5)
    p.add_argument('--num_train_epochs', type=float, default=2.0)
    p.add_argument('--per_device_train_batch_size', type=int, default=1)
    p.add_argument('--gradient_accumulation_steps', type=int, default=4)
    p.add_argument('--warmup_ratio', type=float, default=0.03)
    p.add_argument('--min_lr_rate', type=float, default=0.1)
    p.add_argument('--eval_steps', type=int, default=50)
    p.add_argument('--logging_steps', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    # dry runs
    p.add_argument('--limit_train', type=int, default=0)
    p.add_argument('--limit_eval', type=int, default=0)
    p.add_argument('--max_steps', type=int, default=-1)
    return p.parse_args()


def load_pairs(data_dir, split, limit):
    """(id, user text, assistant text) for every record that has both files."""
    pairs = []
    for md in sorted(glob.glob(f'{data_dir}/input_md/{split}/*.md')):
        rid = Path(md).stem
        out = f'{data_dir}/output_items/{split}/{rid}.json'
        pairs.append({'id': rid,
                      'user': Path(md).read_text(encoding='utf-8'),
                      'assistant': Path(out).read_text(encoding='utf-8').rstrip('\n')})
        if limit and len(pairs) >= limit:
            break
    if not pairs:
        raise RuntimeError(f'no records in {data_dir}/input_md/{split}')
    return pairs


def tokenize(example, tokenizer):
    user = [{'role': 'user', 'content': example['user']}]
    prompt = tokenizer.apply_chat_template(user, add_generation_prompt=True, tokenize=True)
    full = tokenizer.apply_chat_template(user + [{'role': 'assistant', 'content': example['assistant']}],
                                         tokenize=True)
    assert full[:len(prompt)] == prompt, example['id']
    return {'input_ids': full, 'labels': [-100] * len(prompt) + full[len(prompt):], 'length': len(full)}


def build_dataset(args, split, limit, tokenizer, training_args):
    """Tokenize once (main process first) and cache next to the run."""
    cache = Path(args.output_dir) / f'cache_{split}_{limit or "all"}'
    with training_args.main_process_first(desc=f'tokenize {split}'):
        if not (cache / 'dataset_info.json').exists():
            ds = Dataset.from_list(load_pairs(args.data_dir, split, limit))
            ds = ds.map(tokenize, fn_kwargs={'tokenizer': tokenizer}, remove_columns=['user', 'assistant'],
                        num_proc=min(16, os.cpu_count() or 1))
            ds.save_to_disk(str(cache))
        ds = load_from_disk(str(cache))
    kept = ds.filter(lambda x: x['length'] <= args.max_len)
    if training_args.process_index == 0:
        print(f'[DATA] {split}: {len(ds)} records, {len(ds) - len(kept)} longer than {args.max_len} dropped, '
              f'max length {max(kept["length"])}')
    return kept.remove_columns(['id', 'length'])


def collate(features):
    n = max(len(f['input_ids']) for f in features)
    ids = torch.full((len(features), n), 0, dtype=torch.long)
    labels = torch.full((len(features), n), -100, dtype=torch.long)
    mask = torch.zeros((len(features), n), dtype=torch.long)
    for i, f in enumerate(features):
        k = len(f['input_ids'])
        ids[i, :k] = torch.tensor(f['input_ids'])
        labels[i, :k] = torch.tensor(f['labels'])
        mask[i, :k] = 1
    return {'input_ids': ids, 'labels': labels, 'attention_mask': mask}


class BestTrainer(Trainer):
    """Keep only the lowest eval_loss model in <output_dir>/best, with BEST.json."""

    best = None

    def evaluate(self, *a, **kw):
        metrics = super().evaluate(*a, **kw)
        loss = metrics.get('eval_loss')
        if loss is not None and (self.best is None or loss < self.best):
            previous, self.best = self.best, loss
            best_dir = Path(self.args.output_dir) / 'best'
            self.save_model(str(best_dir))
            if self.args.process_index == 0:
                self.processing_class.save_pretrained(str(best_dir))
                (best_dir / 'BEST.json').write_text(json.dumps(
                    {'step': self.state.global_step, 'epoch': self.state.epoch, 'eval_loss': loss,
                     'previous_best': previous}, indent=2))
        return metrics


def main():
    args = parse_args()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type='cosine_with_min_lr',
        lr_scheduler_kwargs={'min_lr_rate': args.min_lr_rate},
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim='adamw_torch',
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed or None,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        prediction_loss_only=True,
        save_strategy='no',
        logging_steps=args.logging_steps,
        report_to='tensorboard',
        logging_dir=str(Path(args.output_dir) / 'tensorboard'),
        dataloader_num_workers=4,
        seed=args.seed,
        disable_tqdm=True,
        remove_unused_columns=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train = build_dataset(args, args.train_split, args.limit_train, tokenizer, training_args)
    evals = build_dataset(args, args.eval_split, args.limit_eval, tokenizer, training_args)

    config = AutoConfig.from_pretrained(args.model_path)
    if args.yarn_factor:
        original = config.max_position_embeddings
        config.rope_scaling = {'rope_type': 'yarn', 'factor': args.yarn_factor,
                               'original_max_position_embeddings': original}
        config.max_position_embeddings = int(original * args.yarn_factor)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, dtype=torch.bfloat16,
                                                 attn_implementation=args.attn)

    trainer = BestTrainer(model=model, args=training_args, train_dataset=train, eval_dataset=evals,
                          data_collator=collate, processing_class=tokenizer)
    trainer.train()
    trainer.evaluate()  # the last steps may fall between eval_steps
    last_dir = Path(args.output_dir) / 'last'
    trainer.save_model(str(last_dir))
    if training_args.process_index == 0:
        tokenizer.save_pretrained(str(last_dir))
        (last_dir / 'LAST.json').write_text(json.dumps(
            {'step': trainer.state.global_step, 'epoch': trainer.state.epoch,
             'eval_loss': trainer.state.log_history[-1].get('eval_loss')}, indent=2))


if __name__ == '__main__':
    main()
