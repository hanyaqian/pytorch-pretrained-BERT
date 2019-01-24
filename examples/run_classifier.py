# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc.
# team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import os
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import classifier_args
import classifier_data as data
from logger import logger


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def main():
    # Arguments
    parser = classifier_args.get_base_parser()
    classifier_args.training_args(parser)
    classifier_args.fp16_args(parser)
    classifier_args.eval_args(parser)
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda
            else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info(
        f"device: {device} n_gpu: {n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.fp16}"
    )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: "
            f"{args.gradient_accumulation_steps}, should be >= 1"
        )

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_prune:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_prune` must be True."
        )
    out_dir_exists = os.path.exists(args.output_dir) and \
        os.listdir(args.output_dir)
    if out_dir_exists and args.do_train:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not "
            "empty."
        )
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in data.processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = data.processors[task_name]()
    num_labels = data.num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train or args.do_prune:
        # Prepare training data
        train_examples = processor.get_train_examples(args.data_dir)
        train_data = data.prepare_tensor_dataset(
            train_examples,
            label_list,
            args.max_seq_length,
            tokenizer
        )
        # Prepare data loader
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size
        )
        # Number of training steps
        num_train_steps = int(
            len(train_examples)
            / args.train_batch_size
            / args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(
        args.bert_model,
        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /
        f"distributed_{args.local_rank}",
        num_labels=num_labels
    )
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex "
                "to use distributed and fp16 training."
            )

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex "
                "to use distributed and fp16 training."
            )

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    # ==== TRAIN ====
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            train_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(train_iterator):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * \
                        warmup_linear(global_step/t_total,
                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    # Save a trained model
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    model = BertForSequenceClassification.from_pretrained(
        args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
    model.to(device)

    n_layers = model.bert.config.num_hidden_layers
    n_heads = model.bert.config.num_attention_heads
    is_main = args.local_rank == -1 or torch.distributed.get_rank() == 0

    # ==== PRUNE ====
    if args.do_prune and is_main:
        # Disable dropout
        model.eval()
        n_prune_steps = len(train_examples) // args.train_batch_size
        logger.info("***** Running pruning *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Batch size = {args.train_batch_size}")
        logger.info(f"  Num steps = {n_prune_steps}")
        prune_iterator = tqdm(train_dataloader, desc="Iteration")
        head_importance = torch.zeros(n_layers, n_heads).cuda()
        tot_tokens = 0

        for step, batch in enumerate(prune_iterator):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids).sum()

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            for layer in range(model.bert.config.num_hidden_layers):
                self_att = model.bert.encoder.layer[layer].attention.self
                ctx = self_att.context_layer_val
                grad_ctx = ctx.grad
                # Take the dot
                dot = torch.einsum("bhli,bhlj->bhl", [grad_ctx, ctx])
                head_importance[layer] += dot.abs().sum(-1).sum(0).detach()

            tot_tokens += input_mask.float().detach().sum().data
        head_importance /= tot_tokens
        # Layerwise importance normalization
        head_importance /= (head_importance ** 2).sum(-1).sqrt().unsqueeze(-1)
        print("Head importance scores")
        for layer in range(n_layers):
            layer_importance = head_importance[layer].cpu().data
            print("\t".join(f"{x:.5f}" for x in layer_importance))
    # ==== EVALUATE ====
    if args.do_eval and is_main:
        # Prepare data
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_data = data.prepare_tensor_dataset(
            eval_examples,
            label_list,
            args.max_seq_length,
            tokenizer
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(eval_examples)}")
        logger.info(f"  Batch size = {args.eval_batch_size}")

        model.eval()
        # Prune heads
        for descriptor in args.attention_mask_heads:
            layer, heads = descriptor.split(":")
            layer = int(layer) - 1
            heads = [int(head) - 1 for head in heads.split(",")]
            self_att = model.bert.encoder.layer[layer].attention.self
            if args.reverse_head_mask:
                excluded_heads = set(heads)
                heads = [head for head in range(self_att.num_attention_heads)
                         if head not in excluded_heads]
            self_att.mask_heads = heads
            self_att._head_mask = None

        # Run prediction for full data
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        if args.save_attention_probs != "":
            example_idx = 0
            attn_partition = 1
            attns_to_save = {}

        def entropy(p):
            plogp = p * torch.log(p)
            plogp[p == 0] = 0
            return -plogp.sum(dim=-1)

        def pairwise_kl(p):
            # p has shape bsz x nheads x L x L and is normalized in the last
            # dim
            logp = torch.log(p)
            logp[p == 0] = 0
            H_pq = -torch.einsum(
                "blij,bljk->blik",
                [p.permute(0, 2, 1, 3), logp.permute(0, 2, 3, 1)]
            ).permute(0, 2, 3, 1)
            H_p = entropy(p).unsqueeze(-2)
            KL = H_pq - H_p
            return KL

        attn_entropy = torch.zeros(n_layers, n_heads).cuda()
        attn_kl = torch.zeros(n_layers, n_heads, n_heads).cuda()
        tot_tokens = 0

        eval_iterator = tqdm(eval_dataloader, desc="Evaluating")
        for input_ids, input_mask, segment_ids, label_ids in eval_iterator:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(
                    input_ids, segment_ids, input_mask, label_ids)
                logits, attns = model(
                    input_ids, segment_ids, input_mask, return_att=True)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

            # Record attention entropy
            for layer, attn in enumerate(attns):
                mask = input_mask.float()
                # Entropy
                masked_entropy = entropy(attn) * mask.unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()
                # KL
                masked_kl = pairwise_kl(attn) * mask.unsqueeze(1).unsqueeze(1)
                attn_kl[layer] += masked_kl.sum(-1).sum(0).detach()
                # Number of tokens
                tot_tokens += mask.detach().sum().data

            if args.save_attention_probs != "":
                attns = [attn.detach().cpu() for attn in attns]
                for batch_idx in range(input_ids.size(0)):
                    attns_to_save[eval_examples[example_idx]] = [
                        attn[batch_idx] for attn in attns]
                    example_idx += 1
                    if (example_idx + 1) % 100 == 0:
                        file = f"{args.save_attention_probs}.{attn_partition}"
                        torch.save(attns_to_save, file)
                        attns_to_save = {}
                        attn_partition += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': loss}

        # Print layer/headwise entropy
        print("Head entropy")
        attn_entropy /= tot_tokens.float()
        for layer in range(len(attn_entropy)):
            print(
                "\t".join(f"{H:.5f}" for H in attn_entropy[layer].cpu().data))

        # Print pairwise layer kl
        print("Pairwise head KL")
        attn_kl /= tot_tokens.float()
        for layer in range(len(attn_kl)):
            print("Layer", layer)
            for head in range(len(attn_kl[layer])):
                head_kl = attn_kl[layer, head].cpu().data
                print("\t".join(f"{kl:.5f}" for kl in head_kl))

        if args.save_attention_probs != "":
            torch.save(attns_to_save,
                       f"{args.save_attention_probs}.{attn_partition}")

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
