# SOURCE:
# https://www.youtube.com/watch?v=dzyDHMycx_c

import json
import os

import evaluate
import numpy as np
import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import (
    pipeline,
    AutoModelForTokenClassification,
    BertTokenizerFast,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from text_parsing_methods_using_nlp.config import (
    ANNOTATED_DATA_FOLDER,
    INVERTED_NER_LABELS,
    MODEL_OUTPUT_FOLDER,
    NER_LABELS,
    NER_LABELS_LIST,
)

from pprint import pprint


class SlovakBertNerModel:

    def __init__(self) -> None:
        # TODO - Docstring
        self._tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        self._metric = evaluate.load('seqeval')

        self._data = self._load_data()

    def _load_data(self) -> Dataset:
        # TODO - Docstring

        modeling_columns = ['text_tokens', 'fixed_ner_tags']

        data = pd.read_csv(
            os.path.join(
                ANNOTATED_DATA_FOLDER,
                'NBS_sentence_450_Annotated_18_03_2023__14_44_55.csv'
            ),
            usecols=modeling_columns
        )

        data.drop_duplicates(inplace=True)

        for col in modeling_columns:
            data[col] = data[col].apply(eval)

        data.columns = ['tokens', 'ner_tags']

        train_test = Dataset.from_pandas(data).map(
            self.tokenize_and_align_labels,
            batched=True
        ).train_test_split(train_size=0.8, test_size=0.2)

        train_valid = train_test['train'].train_test_split(
            train_size=0.8, 
            test_size=0.2
        )

        return DatasetDict({
            'train': train_valid['train'],
            'test': train_test['test'],
            'validation': train_valid['test']}
        )         

    def tokenize_and_align_labels(
        self,
        dataset_row,
        label_all_tokens=True
    ):
        tokenized_input = self._tokenizer(
            dataset_row['tokens'],
            truncation=True,
            is_split_into_words=True
        )
        labels = []

        for i, label in enumerate(dataset_row['ner_tags']):
            word_ids = tokenized_input.word_ids(batch_index=i)
            # word_ids() => Return a list mapping the tokens to their actual
            # word in the initial sentence. It returns a list indicating the
            # word corresponding to each token.
            previous_word_idx = None

            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    #  Set -100 as the label for these special tokens
                    label_ids.append(-100)

                elif word_idx != previous_word_idx:
                    # If current word_idx is not equal to the previous
                    # word_idx, then it is the most regular case and add the
                    # corresponding token
                    label_ids.append(label[word_idx])

                else:
                    # To take care of the sub-words which have the same
                    # word_idx set -100 as well for them but only if
                    # label_all_tokens = False
                    label_ids.append(
                        label[word_idx] if label_all_tokens else -100
                    )

                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_input['labels'] = labels

        return tokenized_input

    def compute_metrics(self, eval_preds):
        pred_logits, labels = eval_preds

        pred_logits = np.argmax(pred_logits, axis=2)

        predictions = [
            [
                NER_LABELS_LIST[eval_preds] for (eval_preds, l) in
                zip(prediction, label) if l != -100
            ] for prediction, label in zip(pred_logits, labels)
        ]

        true_labels = [
            [
                NER_LABELS_LIST[l] for (eval_preds, l) in
                zip(prediction, label) if l != -100
            ] for prediction, label in zip(pred_logits, labels)
        ]

        results = self._metric.compute(
            predictions=predictions, references=true_labels
        )

        return {
            'precision': results['overall_precision'],
            'recall': results['overall_recall'],
            'accuracy': results['overall_accuracy'],
            'f1': results['overall_f1'],
        }

    def func(self) -> None:


        model = AutoModelForTokenClassification.from_pretrained(
            'gerulata/slovakbert',
            num_labels=14,
        )

        args = TrainingArguments(
            'ner_model',
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
        )

        data_collator = DataCollatorForTokenClassification(self._tokenizer)

        trainer = Trainer(
            model,
            args, 
            train_dataset=self._data['train'],
            eval_dataset=self._data['validation'],
            data_collator=data_collator,
            tokenizer=self._tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

        model.save_pretrained('ner_model')

        self._tokenizer.save_pretrained('tokenizer')

        config = json.load(open('ner_model/config.json'))

        config['id2label'] = NER_LABELS
        config['label2id'] = INVERTED_NER_LABELS

        json.dump(config, open('ner_model/config.json', 'w'))

        model_fine_tuned = AutoModelForTokenClassification.from_pretrained('ner_model')

        nlp = pipeline('ner', model=model_fine_tuned, tokenizer=self._tokenizer)

        example = 'Začiatkom roka 2021 sa objavili nezhody medzi Richardom Sulíkom a šéfom hnutia OĽANO Igorom Matovičom, ktoré v istej miere pretrvávajú aj dodnes.'

        pprint(nlp(example))
