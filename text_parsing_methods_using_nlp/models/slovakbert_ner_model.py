# SOURCE:
# https://www.youtube.com/watch?v=dzyDHMycx_c

import json
import os
from copy import deepcopy

import evaluate
import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.features import Sequence, Value
from transformers import (
    pipeline,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    RobertaTokenizerFast,
    TrainingArguments,
    Trainer,
    TrainerCallback, 
    TrainerControl,
    TrainerState
)

from text_parsing_methods_using_nlp.config import (
    ANNOTATED_DATA_FOLDER,
    INVERTED_NER_LABELS,
    NER_LABELS,
    NER_LABELS_LIST,
    SLOVAKBERT_NER_MODEL_CONFIG,
    SLOVAKBERT_NER_MODEL_OUTPUT_FOLDER,
    SLOVAKBERT_NER_MODEL_TOKENIZER_OUTPUT_FOLDER,
    SLOVAKBERT_NER_MODEL_TRAINER_STATE,
)
from text_parsing_methods_using_nlp.utils.utils import process_training_history

from pprint import pprint

class CustomCallback(TrainerCallback):
    # SOURCE: https://stackoverflow.com/a/70564009/14319439

    # TODO - Docstring
    
    def __init__(self, trainer, data) -> None:
        # TODO - Docstring

        super().__init__()
        self._trainer = trainer
        self._data = data
    
    def on_step_end(
        self,
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ) -> (TrainerControl | None):
        # TODO - Docstring

        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, 
                metric_key_prefix="train"
            )
            return control_copy

    def on_train_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ) -> None:
        # TODO - Docstring

        self._trainer.evaluate(
            eval_dataset=self._data['test'], 
            metric_key_prefix = 'test'
        )
        return super().on_train_end(args, state, control, **kwargs)


class SlovakBertNerModel:

    # TODO - Docstring

    def __init__(self) -> None:
        # TODO - Docstring

        self._tokenizer = RobertaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path='crabz/slovakbert-ner'
            
        )

        self._model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path='gerulata/slovakbert',
            num_labels=len(NER_LABELS_LIST),
        )

        self._training_args = TrainingArguments(
            output_dir=SLOVAKBERT_NER_MODEL_OUTPUT_FOLDER,
            evaluation_strategy='steps',
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            lr_scheduler_type='linear',
            num_train_epochs=15,
            # num_train_epochs=1,   # Only for testing purposes
            seed=42,
            save_total_limit=2,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-08,
        )

        self._data = self._load_data()

        self._data_collator = DataCollatorForTokenClassification(
            tokenizer=self._tokenizer
        )

        self._trainer = Trainer(
            model=self._model,
            args=self._training_args,
            train_dataset=self._data['train'],
            eval_dataset=self._data['validation'],
            data_collator=self._data_collator,
            tokenizer=self._tokenizer,
            compute_metrics=self._compute_metrics,
        )
        
        self._trainer.add_callback(
            CustomCallback(trainer=self._trainer, data=self._data)
        ) 

        self._metric = evaluate.load(path='seqeval')

    def _load_data(self, concat_with_wikiann=True) -> DatasetDict:
        # TODO - Docstring

        modeling_columns = ['text_tokens', 'fixed_ner_tags']

        if concat_with_wikiann:
            wikiann_data = load_dataset('wikiann', 'sk')

            for split in wikiann_data.keys():
                wikiann_data[split] = wikiann_data[split].remove_columns(
                    ['langs', 'spans']
                )

            tokenized_wikiann_data = wikiann_data.map(
                self._tokenize_and_align_labels,
                batched=True
            )

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
            self._tokenize_and_align_labels,
            batched=True
        ).train_test_split(train_size=0.8, test_size=0.2)

        train_valid = train_test['train'].train_test_split(
            train_size=0.8,
            test_size=0.2
        )

        if concat_with_wikiann:
            train_dataset = concatenate_datasets(
                [
                    train_valid['train'],
                    tokenized_wikiann_data['train'].cast_column(
                        'ner_tags',
                        Sequence(feature=Value(dtype='int64', id=None))
                    )
                ]
            )

            validation_dataset = concatenate_datasets(
                [
                    train_valid['train'],
                    tokenized_wikiann_data['validation'].cast_column(
                        'ner_tags',
                        Sequence(feature=Value(dtype='int64', id=None))
                    )
                ]
            )

            test_dataset = concatenate_datasets(
                [
                    train_test['test'],
                    tokenized_wikiann_data['test'].cast_column(
                        'ner_tags',
                        Sequence(feature=Value(dtype='int64', id=None))
                    )
                ]
            )

            return DatasetDict({
                'train': train_dataset,
                'test': validation_dataset,
                'validation': test_dataset
            })

        return DatasetDict({
            'train': train_valid['train'],
            'test': train_test['test'],
            'validation': train_valid['test']
        })

    def _tokenize_and_align_labels(
        self,
        dataset_row,
        label_all_tokens=True
    ):
        # TODO - Docstring

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

    def _compute_metrics(self, eval_preds):
        # TODO - Docstring

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

    def _save_model(self):
        # TODO - Docstring

        self._model.save_pretrained(SLOVAKBERT_NER_MODEL_OUTPUT_FOLDER)
        # self._tokenizer.save_pretrained(
        #     SLOVAKBERT_NER_MODEL_TOKENIZER_OUTPUT_FOLDER
        # )

        config = json.load(open(SLOVAKBERT_NER_MODEL_CONFIG))

        config['id2label'] = NER_LABELS
        config['label2id'] = INVERTED_NER_LABELS

        json.dump(config, open(SLOVAKBERT_NER_MODEL_CONFIG, 'w'))

        self._trainer.save_state()

    def _plot_model_metrics(self) -> None:
        # TODO - Docstring

        training_history = process_training_history(
            training_history_path=SLOVAKBERT_NER_MODEL_TRAINER_STATE
        )

        print(training_history.head())
        
        # TODO - Finish function

    def train(self) -> None:
        # TODO - Docstring

        self._trainer.train()

    def evaluate(self) -> None:
        # TODO - Docstring

        # self._save_model()
        self._plot_model_metrics()

    def predict(self) -> None:
        model_fine_tuned = AutoModelForTokenClassification.from_pretrained(
            SLOVAKBERT_NER_MODEL_OUTPUT_FOLDER
        )

        nlp = pipeline(
            'ner',
            model=model_fine_tuned,
            tokenizer=self._tokenizer
        )

        example = 'Začiatkom roka 2021 sa objavili nezhody medzi Richardom Sulíkom a šéfom hnutia OĽANO Igorom Matovičom, ktoré v istej miere pretrvávajú aj dodnes.'

        pprint(nlp(example))

    def __call__(self) -> None:
        # self.train()
        self.evaluate()
        # self.predict()
