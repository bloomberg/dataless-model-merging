from collections import Counter, defaultdict

import numpy as np


def get_metric(p_num: int, total_num: int, total_predicted_num: int):
    """
    Return the metrics of precision, recall and f-score, based on the number
    (We make this small piece of function in order to reduce the code effort and less possible to have typo error)
    :param p_num:
    :param total_num:
    :param total_predicted_num:
    :return:
    """
    precision = (
        p_num * 1.0 / total_predicted_num * 100 if total_predicted_num != 0 else 0
    )
    recall = p_num * 1.0 / total_num * 100 if total_num != 0 else 0
    fscore = (
        2.0 * precision * recall / (precision + recall)
        if precision != 0 or recall != 0
        else 0
    )
    return precision, recall, fscore


class Span:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """

    def __init__(self, left: int, right: int, type: str):
        """
        A span compose of left, right (inclusive) and its entity label.
        :param left:
        :param right: inclusive.
        :param type:
        """
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return (
            self.left == other.left
            and self.right == other.right
            and self.type == other.type
        )

    def __hash__(self):
        return hash((self.left, self.right, self.type))


def get_ner_metrics_func(idx2label):
    def ner_metrics(p):
        batch_pred_ids = p.predictions.argmax(-1)
        batch_gold_ids = p.label_ids
        seq_len = np.array([batch_gold_ids.shape[1]] * batch_gold_ids.shape[0])

        p_dict, total_predict_dict, total_entity_dict = evaluate_batch_insts(
            batch_pred_ids, batch_gold_ids, seq_len, idx2label
        )
        total_p = sum(list(p_dict.values()))
        total_predict = sum(list(total_predict_dict.values()))
        total_entity = sum(list(total_entity_dict.values()))
        precision, recall, fscore = get_metric(total_p, total_entity, total_predict)
        result = {"precision": precision, "recall": recall, "f1": fscore}
        result["key_score"] = result["f1"]
        return result

    return ner_metrics


def evaluate_batch_insts(batch_pred_ids, batch_gold_ids, word_seq_lens, idx2label):
    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, number of all positive, number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, number of entities predicted, number of entities in the dataset)
    """
    batch_p_dict = defaultdict(int)
    batch_total_entity_dict = defaultdict(int)
    batch_total_predict_dict = defaultdict(int)
    all_preds, all_outputs = [], []

    idx2label = {k: v for k, v in idx2label.items()}
    idx2label[-100] = "<PAD>"
    word_seq_lens = word_seq_lens.tolist()
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        output = [idx2label[lbl] for lbl in output]
        prediction = [idx2label[lbl] for lbl in prediction]
        # convert to span
        output_spans = set()
        start = -1
        started = False
        prev_i, prev_output = None, None
        for i in range(len(output)):
            if output[i] == "<PAD>":
                continue

            if started and not output[i].startswith("I-"):
                output_spans.add(Span(start, prev_i, prev_output[prev_i][2:]))
                batch_total_entity_dict[prev_output[prev_i][2:]] += 1
                started = False

            if output[i].startswith("B-"):
                start = i
                started = True
            elif output[i].startswith("I-") and not started:  # single word
                start = i
                output_spans.add(Span(start, start, output[i][2:]))
                batch_total_entity_dict[output[i][2:]] += 1
                started = False

            prev_i, prev_output = i, output

        if started:
            output_spans.add(Span(start, prev_i, prev_output[prev_i][2:]))
            batch_total_entity_dict[prev_output[prev_i][2:]] += 1
            started = False

        predict_spans = set()
        start = -1
        started = False

        prev_i, prev_prediction = None, None

        for i in range(len(prediction)):
            if output[i] == "<PAD>":
                continue

            if started and not prediction[i].startswith("I-"):
                predict_spans.add(Span(start, prev_i, prev_prediction[prev_i][2:]))
                batch_total_predict_dict[prev_prediction[prev_i][2:]] += 1
                started = False

            if prediction[i].startswith("B-"):
                start = i
                started = True
            elif prediction[i].startswith("I-") and not started:  # single word
                start = i
                predict_spans.add(Span(start, start, prediction[i][2:]))
                batch_total_predict_dict[prediction[i][2:]] += 1
                started = False

            prev_i, prev_prediction = i, prediction

        if started:
            predict_spans.add(Span(start, prev_i, prev_prediction[prev_i][2:]))
            batch_total_predict_dict[prev_prediction[prev_i][2:]] += 1
            started = False

        correct_spans = predict_spans.intersection(output_spans)
        for span in correct_spans:
            batch_p_dict[span.type] += 1

        all_preds.append(predict_spans)
        all_outputs.append(output_spans)

    return (
        Counter(batch_p_dict),
        Counter(batch_total_predict_dict),
        Counter(batch_total_entity_dict),
    )
