# SOURCE: https://www.youtube.com/watch?v=dzyDHMycx_c

import csv
import json
import os
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import ray
import pandas as pd
from datasets import concatenate_datasets, load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.features import Sequence, Value
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from transformers import (
    pipeline,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    IntervalStrategy,
    RobertaTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from text_parsing_methods_using_nlp.annotation.annotator import Annotator
from text_parsing_methods_using_nlp.config import (
    ANNOTATED_DATA_FOLDER,
    INVERTED_NER_LABELS,
    MODEL_OUTPUT_FOLDER,
    MODEL_TEST_DATASET_FOLDER,
    NER_LABELS,
    NER_LABELS_LIST,
    TRAINING_HISTORIES_OUTPUT_FOLDER,
)
from text_parsing_methods_using_nlp.ops.plotter import Plotter
from text_parsing_methods_using_nlp.utils.utils import process_training_history

from pprint import pprint


class SlovakBertNerModelCallback(TrainerCallback):
    # SOURCE: https://stackoverflow.com/a/70564009/14319439

    """Represents custom SlovakBERT NER Model Callbacks.

    These Callbacks are used during and at the end of Model training for 
    additional evaluation metric generation.
    """

    def __init__(self, trainer: Trainer, data: DatasetDict) -> None:
        """Initializes the SlovakBertNerModelCallback Class.

        Args:
            trainer (Trainer): SlovakBERT NER Model Trainer.
            data (DatasetDict): Datasets used during training, validation
            and testing.
        """
        super().__init__()
        self._trainer = trainer
        self._data = data

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: dict[str, Any]
    ) -> (TrainerControl | None):
        """Triggers 'On Step End' evaluation.

        Args:
            args (TrainingArguments): Model Training Arguments.
            state (TrainerState): Current Model Trainer State.
            control (TrainerControl): Current Model Trainer Controller.
            **kwargs (dict[str, Any]): Additional Keyword Arguments.

        Returns:
            TrainerControl | None: TrainerControl if evaluation is needed,
            None otherwise.
        """
        if control.should_evaluate:
            control_copy = deepcopy(control)

            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset,
                metric_key_prefix='train'
            )

            return control_copy

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: dict[str, Any]
    ) -> None:
        """Triggers 'On Train End' evaluation.

        Args:
            args (TrainingArguments): Model Training Arguments.
            state (TrainerState): Current Model Trainer State.
            control (TrainerControl): Current Model Trainer Controller.
            **kwargs (dict[str, Any]): Additional Keyword Arguments.
        """

        self._trainer.evaluate(
            eval_dataset=self._data['test'],
            metric_key_prefix='test'
        )

        return super().on_train_end(args, state, control, **kwargs)


class SlovakBertNerModel:

    """Represents the custom SlovakBERT NER Model."""

    def __init__(self, version: str) -> None:
        """Initializes the SlovakBertNerModel Class.

        Args:
            version (str): Current Model Version.
        """
        self._model_name = f'SlovakBERT_NER_Model_V{version}'

        self._hyperparam_tuning = int(version.split('.')[-1]) == 4

        #######################################################################
        #                            Path Variables                           #
        #######################################################################

        self._timestamp = datetime.now().strftime('%d_%m_%Y__%H_%M_%S')

        self._model_output_folder = os.path.join(
            MODEL_OUTPUT_FOLDER,
            next(
                (
                    s for s in os.listdir(MODEL_OUTPUT_FOLDER)
                    if f'{version}_' in s
                ),
                None
            )
        )

        self._model_config = os.path.join(
            self._model_output_folder,
            'config.json'
        )

        self._model_trainer_state_path = os.path.join(
            self._model_output_folder,
            'trainer_state.json'
        )

        self._training_history_output_path = os.path.join(
            TRAINING_HISTORIES_OUTPUT_FOLDER,
            self._model_name
        )

        self._test_dataset_file_name = 'Test_Dataset'

        self._test_dataset_length = 0

        self._test_dataset_output_path = os.path.join(
            MODEL_TEST_DATASET_FOLDER,
            f'{self._test_dataset_file_name}_{self._timestamp}.csv'
        )

        #######################################################################
        #                            Model Variables                          #
        #######################################################################

        self._seed = 42

        self._tokenizer = RobertaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path='crabz/slovakbert-ner'
        )

        self._model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path='gerulata/slovakbert',
            num_labels=len(NER_LABELS_LIST),
        )

        self._training_args = TrainingArguments(
            output_dir=self._model_output_folder,
            evaluation_strategy=IntervalStrategy.STEPS,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-08,
            num_train_epochs=15,
            lr_scheduler_type='linear',
            save_total_limit=2,
            seed=self._seed,
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model='f1'
        )

        self._data_collator = DataCollatorForTokenClassification(
            tokenizer=self._tokenizer
        )

        self._data = self._load_data()

        self._trainer = Trainer(
            model=self._model,
            args=self._training_args,
            data_collator=self._data_collator,
            train_dataset=self._data['train'],
            eval_dataset=self._data['validation'],
            tokenizer=self._tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        self._trainer.add_callback(
            SlovakBertNerModelCallback(trainer=self._trainer, data=self._data)
        )

        self._metric = evaluate.load(path='seqeval')

        #######################################################################
        #                                Utils                                #
        #######################################################################

        self._plotter = Plotter()

    ###########################################################################
    #                              Data Processing                            #
    ###########################################################################

    @staticmethod
    def filter_numeric_data_from_wikiann_dataset(
        dataset_row: Dict[str, List[Union[str, int]]]
    ) -> bool:
        """Filters rows containing numeric values in Wikiann Dataset.

        Args:
            dataset_row (Dict[str, List[Union[str, int]]]): Row from Wikiann
            Dataset.

        Returns:
            bool: True if row contains no numeric value, False otherwise.
        """
        for token in dataset_row['tokens']:
            if any(re.findall(r'\d+', token)):
                return False
        return True

    def _save_test_data_to_csv(self, test_dataset: Dataset) -> None:
        """Saves Test Dataset to CSV file.

        Args:
            test_dataset (Dataset): Test Dataset in Arrow Dataset format.
        """
        # Convert Arrow Dataset to Pandas DataFrame
        test_data_pd = test_dataset.to_pandas(batch_size=32)

        # Filter out rows containing apostrophes
        regex = re.compile(r'(.*\'.*)')
        regex_match = np.vectorize(lambda x: bool(regex.match(x)))

        test_data_pd = test_data_pd[
            test_data_pd['tokens'].apply(lambda x: not any(regex_match(x)))
        ]

        # Save Dataset length
        self._test_dataset_length = len(test_data_pd)

        # Write DataFrame to CSV File
        test_data_pd[['tokens', 'ner_tags']].to_csv(
            path_or_buf=self._test_dataset_output_path,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
        )

    def _load_data(
        self,
        concat_with_wikiann: bool = True,
        filter_numeric_wikiann_rows: bool = True
    ) -> DatasetDict:
        """Loads NBS Dataset (and adds Wikiann Dataset if needed) to 
        DatasetDict.

        Args:
            concat_with_wikiann (bool, optional): Option to add Wikiann 
            Dataset. Defaults to True.
            filter_numeric_wikiann_rows (bool, optional): Option to filter 
            numeric values from Wikiann Dataset. Defaults to True.

        Returns:
            DatasetDict: Train, Validation and Test Datasets.
        """
        modeling_columns = ['text_tokens', 'fixed_ner_tags']

        if concat_with_wikiann:
            # Load Wikiann Dataset
            wikiann_data = load_dataset('wikiann', 'sk')

            # For each subset (Train, Validation and Test)
            for split in wikiann_data.keys():

                # Remove unused columns and cast 'ner_tags' column to match
                # NBS Dataset format
                wikiann_data[split] = wikiann_data[split].remove_columns(
                    ['langs', 'spans']
                ).cast_column(
                    'ner_tags', Sequence(feature=Value(dtype='int64', id=None))
                )

                if filter_numeric_wikiann_rows:
                    # Filter numeric values from Wikiann Dataset
                    wikiann_data[split] = wikiann_data[split].filter(
                        lambda row:
                        self.filter_numeric_data_from_wikiann_dataset(
                            dataset_row=row
                        )
                    )

        # Load NBS Dataset
        data = pd.read_csv(
            os.path.join(
                ANNOTATED_DATA_FOLDER,
                'Crabz_-_SlovakBERT_NER_Model',
                'NBS_sentence_450_Annotated_18_03_2023__14_44_55.csv'
            ),
            usecols=modeling_columns
        )

        # Remove duplicates
        data.drop_duplicates(inplace=True)

        # Convert string List representation to actual List of values
        for col in modeling_columns:
            data[col] = data[col].apply(eval)

        # Rename columns
        data.columns = ['tokens', 'ner_tags']

        # Split NBS Dataset to Train (80%) and Test (20%) subset
        train_test = Dataset.from_pandas(data).train_test_split(
            train_size=0.8,
            test_size=0.2,
            seed=self._seed
        )

        # Split Train Dataset to Train (80%) and Validation (20%) subset
        train_valid = train_test['train'].train_test_split(
            train_size=0.8,
            test_size=0.2,
            seed=self._seed
        )

        if concat_with_wikiann:
            # Create concatenated Test Dataset
            test_dataset = concatenate_datasets(
                [train_test['test'], wikiann_data['test']]
            )

            # Save Test Dataset to CSV file
            self._save_test_data_to_csv(test_dataset)

            # Create DatasetDict of 3 Dataset and apply tokenizer to them
            return DatasetDict(
                {
                    'train': concatenate_datasets(
                        [train_valid['train'], wikiann_data['train']]
                    ),
                    'validation': concatenate_datasets(
                        [train_valid['train'], wikiann_data['validation']]
                    ),
                    'test': test_dataset,
                }
            ).map(
                self._tokenize_and_align_labels,
                batched=True
            )

        # Create DatasetDict of 3 (not concatenated) Dataset and apply
        # tokenizer to them
        return DatasetDict({
            'train': train_valid['train'],
            'validation': train_valid['test'],
            'test': train_test['test']
        }).map(
            self._tokenize_and_align_labels,
            batched=True
        )

    def _tokenize_and_align_labels(
        self,
        dataset_row: Dict[str, List[Union[str, int]]],
        label_all_tokens: bool = True
    ) -> Dict[str, List[Union[str, int]]]:
        """Applies Tokenizer to given Dataset row.

        Args:
            dataset_row (Dict[str, List[Union[str, int]]]): Dataset row to 
            tokenize.
            label_all_tokens (bool, optional): Option to label every token. 
            Defaults to True.

        Returns:
            Dict[str, List[Union[str, int]]]: Tokenized Dataset row. 
        """

        # SOURCE: https://www.youtube.com/watch?v=dzyDHMycx_c

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

    def _compute_metrics(self, eval_preds) -> Dict[str, float]:
        """Computes metrics on predictions.

        Args:
            eval_preds (_type_): Predictions on the Evaluation set.

        Returns:
            Dict[str, float]: Computed metrics.
        """
        # TODO - Type Hint

        # SOURCE: https://www.youtube.com/watch?v=dzyDHMycx_c

        pred_logits, labels = eval_preds

        print('Eval Preds:', type(eval_preds))

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

    ###########################################################################
    #                            Model Training                               #
    ###########################################################################

    @staticmethod
    def _model_init():
        return AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path='gerulata/slovakbert',
            num_labels=len(NER_LABELS_LIST),
        )

    def _tune_ner_model(self, number_of_trials: int = 10) -> TrainingArguments:
        """Executes Hyperparameter Search.

        Args:
            number_of_trials (int, optional): Number of trials. Defaults to 10.

        Returns:
            TrainingArguments: Training Arguments of best model.
        """

        # SOURCE:
        # https://docs.ray.io/en/latest/tune/examples/pbt_transformers.html#tune-huggingface-example

        hyperparam_tuner_training_args = TrainingArguments(
            output_dir=self._model_output_folder,
            do_train=True,
            do_eval=True,
            evaluation_strategy=IntervalStrategy.STEPS,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-08,
            num_train_epochs=15,
            max_steps=-1,
            lr_scheduler_type='linear',
            warmup_steps=0,
            save_total_limit=2,
            seed=self._seed,
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model='f1'
        )

        hyperparam_tuner_trainer = Trainer(
            model_init=self._model_init,
            args=hyperparam_tuner_training_args,
            train_dataset=self._data['train'],
            eval_dataset=self._data['validation'],
            compute_metrics=self._compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        hyperparam_tuner_tune_config = {
            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 8,
            'num_train_epochs': tune.choice([5, 10, 15, 20]),
        }

        hyperparam_tuner_scheduler = PopulationBasedTraining(
            time_attr='training_iteration',
            metric='eval_f1',
            mode='max',
            perturbation_interval=1,
            hyperparam_mutations={
                'learning_rate': tune.uniform(1e-5, 5e-5),
                'weight_decay': tune.uniform(0.0, 0.3),
                'adam_beta1': tune.uniform(0.8, 0.999),
                'adam_beta2': tune.uniform(0.999, 1.0),
                'adam_epsilon': tune.uniform(1e-7, 1e-10),
            },
        )

        hyperparam_tuner_reporter = CLIReporter(
            parameter_columns={
                'weight_decay': 'w_decay',
                'learning_rate': 'lr',
                'num_train_epochs': 'num_epochs',
            },
            metric_columns=[
                'eval_acc', 'eval_loss', 'epoch', 'training_iteration'
            ],
        )

        best_hyperparams = hyperparam_tuner_trainer.hyperparameter_search(
            hp_space=lambda _: hyperparam_tuner_tune_config,
            n_trials=number_of_trials,
            backend='ray',
            resources_per_trial={'cpu': -1, 'gpu': -1},
            scheduler=hyperparam_tuner_scheduler,
            keep_checkpoints_num=1,
            checkpoint_score_attr='training_iteration',
            stop=None,
            progress_reporter=hyperparam_tuner_reporter,
            local_dir='~/ray_results/',
            name='tune_transformer_pbt',
            log_to_file=True,
        ).hyperparameters

        print('Best Hyperparams')
        pprint(best_hyperparams)

        return best_hyperparams

    def train(self) -> None:
        """Runs Model Training Process."""
        if self._hyperparam_tuning:
            self._trainer.args = self._tune_ner_model(
                number_of_trials=1
            )

        self._trainer.train()
        self._save_model()

    ###########################################################################
    #                            Model Evaluation                             #
    ###########################################################################

    def _plot_model_metrics(self) -> None:
        """Plots simple Model (Training History) Metrics"""
        self._plotter.display_training_history(
            model_name=' '.join(self._model_name.split('_')),
            history=process_training_history(
                training_history_path=self._model_trainer_state_path
            ),
            index_col='epoch',
            path=self._training_history_output_path,
            timestamp=self._timestamp
        )

    def evaluate(self) -> None:
        """Evaluates Model performance on Test Dataset."""
        # Plot Model (Training History) Metrics
        self._plot_model_metrics()

        # Send Model and Test Data to Annotator for additional Classification
        # evaluation
        annotator = Annotator(
            input_data_filepath=self._test_dataset_output_path,
            input_data_filename=self._test_dataset_file_name,
            dataset_size=self._test_dataset_length,
            model_test_dataset_evaluation=True,
            timestamp=self._timestamp,
            model=AutoModelForTokenClassification.from_pretrained(
                self._model_output_folder
            ).to('cpu'),
            model_name=self._model_name,
            tokenizer=self._tokenizer
        )

        annotator()

    ###########################################################################
    #                            Model Prediction                             #
    ###########################################################################

    def predict(self, prediction_input: str) -> None:
        """Generates Model predictions for given input.

        Args:
            prediction_input (str): Prediction input string.
        """

        # TODO - Refactor Prediction

        ner_pipeline = pipeline(
            'ner',
            model=AutoModelForTokenClassification.from_pretrained(
                self._model_output_folder
            ),
            tokenizer=self._tokenizer
        )

        classifications = ner_pipeline(prediction_input)

        pprint(classifications)

    ###########################################################################
    #                             Model Utils                                 #
    ###########################################################################

    def _update_model_config(self) -> None:
        """Updates Model Configuration file.

        Saves NER Category IDs and Labels to Model Configuration.
        """
        config = json.load(open(self._model_config))

        config['id2label'] = NER_LABELS
        config['label2id'] = INVERTED_NER_LABELS

        json.dump(config, open(self._model_config, 'w'))

    def _save_model(self) -> None:
        """Saves Model."""
        # Save Model Weights
        self._model.save_pretrained(self._model_output_folder)

        # Save IDs and Labels to Model Configuration
        self._update_model_config()

        # Save Trainer State
        self._trainer.save_state()

    def __call__(self) -> None:
        """Make SlovakBertNerModel object callable."""
        self.train()
        self.evaluate()
        # self.predict(prediction_input='Začiatkom roka 2021 sa objavili nezhody medzi Richardom Sulíkom a šéfom hnutia OĽANO Igorom Matovičom, ktoré v istej miere pretrvávajú aj dodnes.')
