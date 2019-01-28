from tqdm import tqdm
from itertools import islice

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from logger import logger
from util import head_entropy, head_pairwise_kl


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def evaluate(
    eval_data,
    model,
    eval_batch_size,
    save_attention_probs=False,
    print_head_entropy=False,
    device=None,
    result=None,
    verbose=True,
):
    """Evaluate the model's accuracy"""
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    if verbose:
        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(eval_data)}")
        logger.info(f"  Batch size = {eval_batch_size}")

    model.eval()
    device = device or model.device

    # Run prediction for full data
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    if save_attention_probs != "":
        example_idx = 0
        attn_partition = 1
        attns_to_save = []

    # Save entropy maybe
    if print_head_entropy:
        n_layers = model.bert.config.num_hidden_layers
        n_heads = model.bert.config.num_attention_heads
        attn_entropy = torch.zeros(n_layers, n_heads).to(device)
        attn_kl = torch.zeros(n_layers, n_heads, n_heads).to(device)

    tot_tokens = 0

    eval_iterator = tqdm(
        eval_dataloader, desc="Evaluating", disable=not verbose)
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
            masked_entropy = head_entropy(attn) * mask.unsqueeze(1)
            attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()
            # KL
            masked_kl = head_pairwise_kl(attn) * mask.unsqueeze(1).unsqueeze(1)
            attn_kl[layer] += masked_kl.sum(-1).sum(0).detach()
            # Number of tokens
            tot_tokens += mask.detach().sum().data

        if save_attention_probs != "":
            attns = [attn.detach().cpu() for attn in attns]
            for batch_idx in range(input_ids.size(0)):
                attns_to_save.append([attn[batch_idx] for attn in attns])
                example_idx += 1
                if (example_idx + 1) % 100 == 0:
                    file = f"{save_attention_probs}.{attn_partition}"
                    torch.save(attns_to_save, file)
                    attns_to_save = []
                    attn_partition += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    result = result or {}
    result["eval_loss"] = eval_loss
    result["eval_accuracy"] = eval_accuracy

    if print_head_entropy and verbose:
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

    if save_attention_probs != "":
        torch.save(attns_to_save,
                   f"{save_attention_probs}.{attn_partition}")

    return result


def calculate_head_importance(
        model,
        data,
        batch_size,
        device,
        normalize_scores_by_layer=True,
        verbose=True,
        subset_size=1.0,
):
    """Calculate head importance scores"""
    # Disable dropout
    model.eval()
    n_prune_steps = int(np.ceil(
        len(data)
        / batch_size
        * subset_size
    ))
    if verbose:
        logger.info("***** Calculating head importance *****")
        logger.info(f"  Num examples = {len(data)}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Num steps = {n_prune_steps}")

    # Prepare data loader
    sampler = RandomSampler(data)
    dataloader = islice(DataLoader(
        data,
        sampler=sampler,
        batch_size=batch_size
    ), n_prune_steps)
    prune_iterator = tqdm(dataloader, desc="Iteration",
                          disable=not verbose, total=n_prune_steps)
    # Head importance tensor
    n_layers = model.bert.config.num_hidden_layers
    n_heads = model.bert.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    tot_tokens = 0

    for step, batch in enumerate(prune_iterator):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        # Compute gradients
        loss = model(input_ids, segment_ids, input_mask, label_ids).sum()
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
    if normalize_scores_by_layer:
        norm_by_layer = (head_importance ** 2).sum(-1).sqrt()
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    return head_importance


def predict(
    predict_data,
    model,
    predict_batch_size,
    device=None,
    verbose=True,
):
    """Predict labels on a dataset"""

    predict_sampler = SequentialSampler(predict_data)
    predict_dataloader = DataLoader(
        predict_data, sampler=predict_sampler, batch_size=predict_batch_size)

    if verbose:
        logger.info("***** Running prediction *****")
        logger.info(f"  Num examples = {len(predict_data)}")
        logger.info(f"  Batch size = {predict_batch_size}")

    model.eval()
    device = device or model.device

    predict_iterator = tqdm(
        predict_dataloader, desc="Analizing", disable=not verbose)

    # Compute model predictions
    predictions = []
    for input_ids, input_mask, segment_ids, label_ids in predict_iterator:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        # Compute logits
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
        # Track predictions
        batch_predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        for pred in batch_predictions:
            predictions.append(batch_predictions)

    return np.asarray(predictions, dtype=int)


def analyze_nli(anal_examples, predictions, labels_list):
    report = {
        "label": {},
        "lex_sem": {},
        "pred_arg_struct": {},
        "logic": {},
        "knowledge": {},
        "domain": {},
    }
    normalizers = {k: {} for k in report}
    for example, pred in zip(anal_examples, predictions):
        correct = float(anal_examples.label == labels_list[predictions])
        for feature in report:
            value = getattr(anal_examples, feature)
            if value is not None:
                # Record whether the model was correct on this particular
                # value of the feature
                if value not in report[feature]:
                    report[feature][value] = 0
                    normalizers[feature][value] = 0
                report[feature][value] += correct
                normalizers[feature][value] += 1
    # Normalize report
    for feature in report:
        Ns = normalizers[feature]
        report[feature] = {k: v / Ns[k] for k, v in report[feature].items()}

    return report
