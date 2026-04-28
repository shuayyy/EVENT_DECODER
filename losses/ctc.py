import math

import torch
import torch.nn as nn


def recover_target_strings(targets, target_lengths, tokenizer):
    target_strings = []
    offset = 0

    for length in target_lengths.tolist():
        target_slice = targets[offset : offset + length].tolist()
        target_strings.append(tokenizer.decode(target_slice))
        offset += length

    return target_strings


def greedy_ctc_decode(logits, tokenizer):
    predicted_ids = logits.argmax(dim=-1).detach().cpu()
    decoded_predictions = []

    for row in predicted_ids.tolist():
        collapsed_tokens = []
        previous_token = None

        for token_id in row:
            if token_id != previous_token:
                collapsed_tokens.append(token_id)
            previous_token = token_id

        bit_string = "".join(
            "0" if token_id == tokenizer.zero_id
            else "1" if token_id == tokenizer.one_id
            else ""
            for token_id in collapsed_tokens
            if token_id != tokenizer.blank_id
        )
        decoded_predictions.append(bit_string)

    return decoded_predictions


def _stable_logaddexp(left, right):
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    if left > right:
        return left + math.log1p(math.exp(right - left))
    return right + math.log1p(math.exp(left - right))


def ctc_prefix_beam_search_decode(
    logits,
    tokenizer,
    beam_width=25,
    target_bit_length=70,
    length_penalty=2.0,
    exact_length_preferred=True,
):
    log_probs = torch.log_softmax(logits.detach().cpu(), dim=-1)
    blank_id = tokenizer.blank_id
    token_to_symbol = {
        tokenizer.zero_id: "0",
        tokenizer.one_id: "1",
    }
    max_prefix_length = target_bit_length + 8
    decoded_predictions = []

    for sample_log_probs in log_probs:
        beams = {"": (0.0, -math.inf)}

        for time_step in sample_log_probs:
            next_beams = {}
            blank_log_prob = float(time_step[blank_id].item())

            for prefix, (prefix_blank, prefix_nonblank) in beams.items():
                total_prefix_score = _stable_logaddexp(prefix_blank, prefix_nonblank)
                next_blank, next_nonblank = next_beams.get(
                    prefix,
                    (-math.inf, -math.inf),
                )
                next_blank = _stable_logaddexp(
                    next_blank,
                    total_prefix_score + blank_log_prob,
                )
                next_beams[prefix] = (next_blank, next_nonblank)

                for token_id, symbol in token_to_symbol.items():
                    token_log_prob = float(time_step[token_id].item())
                    last_symbol = prefix[-1] if prefix else None

                    if symbol == last_symbol:
                        same_blank, same_nonblank = next_beams.get(
                            prefix,
                            (-math.inf, -math.inf),
                        )
                        same_nonblank = _stable_logaddexp(
                            same_nonblank,
                            prefix_nonblank + token_log_prob,
                        )
                        next_beams[prefix] = (same_blank, same_nonblank)

                    if len(prefix) >= max_prefix_length:
                        continue

                    extended_prefix = prefix + symbol
                    extend_score = prefix_blank + token_log_prob
                    if symbol != last_symbol:
                        extend_score = _stable_logaddexp(
                            extend_score,
                            prefix_nonblank + token_log_prob,
                        )

                    extended_blank, extended_nonblank = next_beams.get(
                        extended_prefix,
                        (-math.inf, -math.inf),
                    )
                    extended_nonblank = _stable_logaddexp(
                        extended_nonblank,
                        extend_score,
                    )
                    next_beams[extended_prefix] = (
                        extended_blank,
                        extended_nonblank,
                    )

            scored_beams = sorted(
                next_beams.items(),
                key=lambda item: _stable_logaddexp(item[1][0], item[1][1]),
                reverse=True,
            )
            beams = dict(scored_beams[:beam_width])

        final_scored_beams = [
            (prefix, _stable_logaddexp(prefix_blank, prefix_nonblank))
            for prefix, (prefix_blank, prefix_nonblank) in beams.items()
        ]

        if exact_length_preferred:
            exact_length_candidates = [
                item
                for item in final_scored_beams
                if len(item[0]) == target_bit_length
            ]
            if exact_length_candidates:
                decoded_predictions.append(
                    max(exact_length_candidates, key=lambda item: item[1])[0]
                )
                continue

        decoded_predictions.append(
            max(
                final_scored_beams,
                key=lambda item: item[1] - length_penalty * abs(len(item[0]) - target_bit_length),
            )[0]
        )

    return decoded_predictions


def compute_sequence_metrics(predictions, targets, target_bit_length):
    exact_matches = 0
    matching_bits = 0
    total_target_bits = 0

    for prediction, target in zip(predictions, targets):
        exact_matches += int(prediction == target)

        compare_length = min(len(prediction), target_bit_length)
        overlap_matches = sum(
            pred_bit == target_bit
            for pred_bit, target_bit in zip(
                prediction[:compare_length],
                target[:compare_length],
            )
        )
        overlap_wrong = compare_length - overlap_matches
        length_penalty = abs(len(prediction) - target_bit_length)
        total_wrong_bits = overlap_wrong + length_penalty
        correct_bits = max(target_bit_length - total_wrong_bits, 0)

        matching_bits += correct_bits
        total_target_bits += target_bit_length

    return exact_matches, matching_bits, total_target_bits


def ctc_loss(logits, targets, input_lengths, target_lengths, criterion=None):
    if criterion is None:
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    log_probs = nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)
    return criterion(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
    )
