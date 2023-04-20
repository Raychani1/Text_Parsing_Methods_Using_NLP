# SOURCE: https://www.youtube.com/watch?v=dzyDHMycx_c

import csv
import json
import os
import re
import random
from copy import deepcopy
from datetime import datetime
from pprint import pprint
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import pandas as pd
import wandb
from datasets import concatenate_datasets, load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.features import Sequence, Value
from natsort import natsorted
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from transformers import (
    pipeline,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    IntervalStrategy,
    PreTrainedModel,
    RobertaForTokenClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from text_parsing_methods_using_nlp.annotation.annotator import Annotator
from text_parsing_methods_using_nlp.config import (
    ANNOTATED_NBS_SENTENCES_DATASET,
    BASE_MODEL,
    BASE_TOKENIZER,
    DEFAULT_MODELLING_PARAMETERS,
    INVERTED_NER_LABELS,
    MODEL_OUTPUT_FOLDER,
    MODEL_TEST_DATASET_FOLDER,
    NER_LABELS,
    NER_LABELS_LIST,
    TRAINING_HISTORIES_OUTPUT_FOLDER,
)
from text_parsing_methods_using_nlp.ops.plotter import Plotter
from text_parsing_methods_using_nlp.utils.utils import (
    process_training_history,
    setup_folders,
    setup_weights_and_biases_env_variables
)


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

    def __init__(
        self,
        version: str,
        **kwargs
    ) -> None:
        """Initializes the SlovakBertNerModel Class.

        Args:
            version (str): Model version.
            hidden_dropout_prob (float, optional): Base Model Hidden Layers 
            Dropout probability.
            attention_probs_dropout_prob (float, optional): Base Model 
            Attention Layers Dropout probability.
            seed (int, optional): Model seed. Defaults to 42.
            filter_numeric_wikiann_rows (bool, optional): Option to filter 
            numeric Wikiann Data rows. Defaults to True.
            strategy (IntervalStrategy, optional): Strategy used for model 
            logging, evaluation and saving. Defaults to IntervalStrategy.EPOCH.
            per_device_train_batch_size(int, optional): Training data batch 
            size.
            per_device_eval_batch_size(int, optional): Evaluation data batch 
            size.
            learning_rate (float, optional): Model learning rate. Defaults to 
            5e-05.
            weight_decay (float, optional): Model weight decay. Defaults to 0.
            num_train_epochs (int, optional): Number of Training Epochs. 
            Defaults to 15.
            early_stopping_patience (int, optional): Early Stopping Patience. 
            Defaults to None.
            hyperparameter_tuning (bool, optional): Option to run 
            hyperparameter tuning. Defaults to True.
        """
        # Set attributes to default values and update the ones set using kwargs
        self.__dict__.update(DEFAULT_MODELLING_PARAMETERS)
        self.__dict__.update(kwargs)

        # Set up Weights and Biases Environmental variables
        setup_weights_and_biases_env_variables()

        self._model_name = f'SlovakBERT_NER_Model_V{version}'

        # region Path Variables

        self._timestamp = datetime.now().strftime('%d_%m_%Y__%H_%M_%S')

        self._model_output_folder = os.path.join(
            MODEL_OUTPUT_FOLDER,
            next(
                (
                    s for s in sorted(os.listdir(MODEL_OUTPUT_FOLDER))
                    if s.startswith(self._model_name)
                ),
                self._model_name
            )
        )

        self._training_history_output_path = os.path.join(
            TRAINING_HISTORIES_OUTPUT_FOLDER,
            self._model_name
        )

        self._model_config_path = None
        self._model_trainer_state_path = None
        self._test_dataset_file_name = None

        self._update_folder_paths()

        self._test_dataset_length = 0

        self._test_dataset_output_path = os.path.join(
            MODEL_TEST_DATASET_FOLDER,
            f'{self._test_dataset_file_name}_{self._timestamp}.csv'
        )

        setup_folders(
            folders=[
                self._model_output_folder,
                self._training_history_output_path
            ]
        )

        # endregion

        # region Model Variables

        self._tokenizer = RobertaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=BASE_TOKENIZER
        )

        self._model_config = AutoConfig.from_pretrained(BASE_MODEL)

        self._model_config.hidden_dropout_prob = self.hidden_dropout_prob
        self._model_config.attention_probs_dropout_prob = (
            self.attention_probs_dropout_prob
        )
        self._model_config.classifier_dropout = self.classifier_dropout_value
        self._model_config.num_labels=len(NER_LABELS_LIST)

        self._model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=BASE_MODEL,
            config = self._model_config
        )

        self._training_args = TrainingArguments(
            output_dir=self._model_output_folder,
            overwrite_output_dir=self.overwrite_output_dir,
            do_train=self.do_train,
            do_eval=self.do_eval,
            do_predict=self.do_predict,
            evaluation_strategy=self.evaluation_strategy,
            prediction_loss_only=self.prediction_loss_only,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            per_gpu_train_batch_size=self.per_gpu_train_batch_size,
            per_gpu_eval_batch_size=self.per_gpu_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            eval_accumulation_steps=self.eval_accumulation_steps,
            eval_delay=self.eval_delay,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            max_grad_norm=self.max_grad_norm,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            warmup_steps=self.warmup_steps,
            log_level=self.log_level,
            log_level_replica=self.log_level_replica,
            log_on_each_node=self.log_on_each_node,
            logging_dir=self.logging_dir,
            logging_strategy=self.logging_strategy,
            logging_first_step=self.logging_first_step,
            logging_steps=self.logging_steps,
            logging_nan_inf_filter=self.logging_nan_inf_filter,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            save_on_each_node=self.save_on_each_node,
            no_cuda=self.no_cuda,
            use_mps_device=self.use_mps_device,
            seed=self.seed,
            data_seed=self.data_seed,
            jit_mode_eval=self.jit_mode_eval,
            use_ipex=self.use_ipex,
            bf16=self.bf16,
            fp16=self.fp16,
            fp16_opt_level=self.fp16_opt_level,
            half_precision_backend=self.half_precision_backend,
            bf16_full_eval=self.bf16_full_eval,
            fp16_full_eval=self.fp16_full_eval,
            tf32=self.tf32,
            local_rank=self.local_rank,
            xpu_backend=self.xpu_backend,
            tpu_num_cores=self.tpu_num_cores,
            tpu_metrics_debug=self.tpu_metrics_debug,
            debug=self.debug,
            dataloader_drop_last=self.dataloader_drop_last,
            eval_steps=self.eval_steps,
            dataloader_num_workers=self.dataloader_num_workers,
            past_index=self.past_index,
            run_name=(
                self._model_name if self.run_name is None
                else self.self.run_name
            ),
            disable_tqdm=self.disable_tqdm,
            remove_unused_columns=self.remove_unused_columns,
            label_names=self.label_names,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            ignore_data_skip=self.ignore_data_skip,
            sharded_ddp=self.sharded_ddp,
            fsdp=self.fsdp,
            fsdp_min_num_params=self.fsdp_min_num_params,
            fsdp_transformer_layer_cls_to_wrap=(
                self.fsdp_transformer_layer_cls_to_wrap
            ),
            deepspeed=self.deepspeed,
            label_smoothing_factor=self.label_smoothing_factor,
            optim=self.optim,
            optim_args=self.optim_args,
            adafactor=self.adafactor,
            group_by_length=self.group_by_length,
            length_column_name=self.length_column_name,
            report_to=self.report_to,
            ddp_find_unused_parameters=self.ddp_find_unused_parameters,
            ddp_bucket_cap_mb=self.ddp_bucket_cap_mb,
            dataloader_pin_memory=self.dataloader_pin_memory,
            skip_memory_metrics=self.skip_memory_metrics,
            use_legacy_prediction_loop=self.use_legacy_prediction_loop,
            push_to_hub=self.push_to_hub,
            resume_from_checkpoint=self.resume_from_checkpoint,
            hub_model_id=self.hub_model_id,
            hub_strategy=self.hub_strategy,
            hub_token=self.hub_token,
            hub_private_repo=self.hub_private_repo,
            gradient_checkpointing=self.gradient_checkpointing,
            include_inputs_for_metrics=self.include_inputs_for_metrics,
            fp16_backend=self.fp16_backend,
            push_to_hub_model_id=self.push_to_hub_model_id,
            push_to_hub_organization=self.push_to_hub_organization,
            push_to_hub_token=self.push_to_hub_token,
            mp_parameters=self.mp_parameters,
            auto_find_batch_size=self.auto_find_batch_size,
            full_determinism=self.full_determinism,
            torchdynamo=self.torchdynamo,
            ray_scope=self.ray_scope,
            ddp_timeout=self.ddp_timeout,
            torch_compile=self.torch_compile,
            torch_compile_backend=self.torch_compile_backend,
            torch_compile_mode=self.torch_compile_mode
        )

        self._data_collator = DataCollatorForTokenClassification(
            tokenizer=self._tokenizer
        )

        self._data = self._load_data(
            filter_numeric_wikiann_rows=self.filter_numeric_wikiann_rows
        )

        self._trainer = Trainer(
            model=self._model if not self.hyperparameter_tuning else None,
            args=self._training_args,
            data_collator=self._data_collator,
            train_dataset=self._data['train'],
            eval_dataset=self._data['validation'],
            tokenizer=self._tokenizer,
            model_init=(
                self._model_init if self.hyperparameter_tuning else None
            ),
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience
                )
            ] if self.early_stopping_patience is not None else None
        )

        self._trainer.add_callback(
            SlovakBertNerModelCallback(trainer=self._trainer, data=self._data)
        )

        self._metric = evaluate.load(path='seqeval')

        # endregion

        # region Utils

        self._plotter = Plotter()

        # endregion

    # region Data Processing

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
        dataset_path: str = ANNOTATED_NBS_SENTENCES_DATASET,
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
        data = pd.read_csv(dataset_path, usecols=modeling_columns)

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
            seed=self.seed
        )

        # Split Train Dataset to Train (80%) and Validation (20%) subset
        train_valid = train_test['train'].train_test_split(
            train_size=0.8,
            test_size=0.2,
            seed=self.seed
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
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
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

    # endregion

    # region Model Training

    def _model_init(self) -> PreTrainedModel:
        """Initializes the Base Model for Hyperparameter Tuning.

        Returns:
            PreTrainedModel: Loaded Pretrained Model.
        """
        return AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=BASE_MODEL,
            num_labels=len(NER_LABELS_LIST),
        )

    def _tune_ner_model(self, number_of_trials: int = 10) -> TrainingArguments:
        """Executes Hyperparameter Tuning.

        Args:
            number_of_trials (int, optional): Number of trials. Defaults to 10.

        Returns:
            TrainingArguments: Training Arguments of best model.
        """

        # SOURCE:
        # https://docs.ray.io/en/latest/tune/examples/pbt_transformers.html#tune-huggingface-example

        hyperparam_tuner_scheduler = PopulationBasedTraining(
            time_attr='training_iteration',
            metric='eval_f1',
            mode='max',
            perturbation_interval=1,
            hyperparam_mutations={
                'learning_rate': tune.uniform(2e-5, 4e-5),
                'weight_decay': tune.uniform(0.2, 0.4),
            },
        )

        hyperparam_tuner_reporter = CLIReporter(
            parameter_columns={
                'weight_decay': 'weight_decay',
                'learning_rate': 'learning_rate',
            },
            metric_columns=[
                'eval_loss', 'eval_precision', 'eval_recall',
                'eval_accuracy', 'eval_f1', 'epoch', 'training_iteration'
            ],
        )

        best_hyperparams = self._trainer.hyperparameter_search(
            n_trials=number_of_trials,
            backend='ray',
            resources_per_trial={'cpu': 24, 'gpu': 1},
            scheduler=hyperparam_tuner_scheduler,
            keep_checkpoints_num=1,
            checkpoint_score_attr='training_iteration',
            stop=None,
            progress_reporter=hyperparam_tuner_reporter,
            local_dir=os.path.join(self._model_output_folder, 'ray_results'),
            name='SlovakBERT_NER_Model_PBT',
            log_to_file=True,
        ).hyperparameters

        print('Best Hyperparams')
        pprint(best_hyperparams)

    def train(self) -> None:
        """Runs Model Training Process."""
        if self.hyperparameter_tuning:
            self._tune_ner_model(
                number_of_trials=1
            )
        else:
            self._trainer.train()
            self._save_model()

        wandb.finish()

    # endregion

    # region Model Evaluation

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

    def evaluate(
        self,
        input_data_filepath: str = None,
        dataset_size: int = None,
        model_input_folder: str = None,
        model_name: str = None
    ) -> None:
        """Evaluates Model performance on Test Dataset."""
        # if not self.hyperparameter_tuning:
        #     # Plot Model (Training History) Metrics
        #     self._plot_model_metrics()

        # Send Model and Test Data to Annotator for additional Classification
        # evaluation
        annotator = Annotator(
            input_data_filepath=(
                self._test_dataset_output_path if input_data_filepath is None
                else input_data_filepath
            ),
            input_data_filename='Test_Dataset',
            dataset_size=(
                self._test_dataset_length if dataset_size is None
                else dataset_size
            ),
            model_test_dataset_evaluation=True,
            timestamp=self._timestamp,
            model=AutoModelForTokenClassification.from_pretrained(
                self._model_output_folder if model_input_folder is None
                else model_input_folder
            ).to('cpu'),
            model_name=self._model_name if model_name is None else model_name,
            tokenizer=self._tokenizer
        )

        annotator()
    # endregion

    # region Model Prediction

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

    # endregion

    # region Model Utils

    def _update_folder_paths(self) -> None:
        # TODO - Docstring

        if not os.path.isdir(self._model_output_folder):
            self._model_name = f'{self._model_name}.1'
        else:
            new_version = int(
                natsorted(
                    list(
                        filter(
                            lambda x: x.startswith(self._model_name),
                            os.listdir(self._model_output_folder)
                        )
                    )
                )[-1].split('.')[-1]
            ) + 1

            self._model_name = f'{self._model_name}.{new_version}'

        self._model_output_folder = os.path.join(
            self._model_output_folder,
            self._model_name
        )

        self._model_config_path = os.path.join(
            self._model_output_folder,
            'config.json'
        )

        self._model_trainer_state_path = os.path.join(
            self._model_output_folder,
            'trainer_state.json'
        )

        self._training_history_output_path = os.path.join(
            self._training_history_output_path,
            self._model_name
        )

        self._test_dataset_file_name = f'{self._model_name}_Test_Dataset'

    def _compute_metrics(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        """Computes metrics on predictions.

        Args:
            eval_preds (EvalPrediction): Predictions on the Evaluation set.

        Returns:
            Dict[str, float]: Computed metrics.
        """

        # SOURCE: https://www.youtube.com/watch?v=dzyDHMycx_c

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

    def _update_model_config_path(self) -> None:
        """Updates Model Configuration file.

        Saves NER Category IDs and Labels to Model Configuration.
        """
        config = json.load(open(self._model_config_path))

        config['id2label'] = NER_LABELS
        config['label2id'] = INVERTED_NER_LABELS

        json.dump(config, open(self._model_config_path, 'w'))

    def _save_model(self) -> None:
        """Saves Model."""
        if not self.hyperparameter_tuning:
            # Save Model Weights
            self._model.save_pretrained(self._model_output_folder)

        # Save IDs and Labels to Model Configuration
        self._update_model_config_path()

        # Save Trainer State
        self._trainer.save_state()

    def __call__(self) -> None:
        """Make SlovakBertNerModel object callable."""
        self.train()
        self.evaluate()
        # self.predict(
        #     prediction_input='Začiatkom roka 2021 sa objavili nezhody medzi Richardom Sulíkom a šéfom hnutia OĽANO Igorom Matovičom, ktoré v istej miere pretrvávajú aj dodnes.'
        # )

    # endregion
