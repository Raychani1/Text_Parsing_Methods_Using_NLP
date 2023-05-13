# SOURCE: https://www.youtube.com/watch?v=dzyDHMycx_c

import csv
import itertools
import os
import random
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

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
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    PreTrainedModel,
    RobertaTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from text_parsing_methods_using_nlp.annotation.annotator import Annotator
from text_parsing_methods_using_nlp.config.config import (
    ANNOTATED_NBS_SENTENCES_DATASET,
    BASE_MODEL,
    BASE_TOKENIZER,
    DATASET_DISTRIBUTION_OUTPUT_FOLDER,
    DEFAULT_MODELLING_PARAMETERS,
    INVERTED_NER_LABELS,
    MODEL_OUTPUT_FOLDER,
    MODEL_TEST_DATASET_FOLDER,
    NER_LABELS,
    NER_LABELS_LIST,
    NER_LABELS_STR_KEY,
    TRAINING_HISTORIES_OUTPUT_FOLDER,
)
from text_parsing_methods_using_nlp.ops.plotter import Plotter
from text_parsing_methods_using_nlp.utils.utils import (
    display_ner_classification,
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
            `trainer` (Trainer): SlovakBERT NER Model Trainer.

            `data` (DatasetDict): Datasets used during training, validation
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
            `args` (TrainingArguments): Model Training Arguments.

            `state` (TrainerState): Current Model Trainer State.

            `control` (TrainerControl): Current Model Trainer Controller.

            `**kwargs` (dict[str, Any]): Additional Keyword Arguments.
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

        Some Default TrainingArguments were added to kwargs for additional 
        options.

        [DOCSTRING SOURCE](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py) 

        Args:
            `version` (str): Model version.

            `hidden_dropout_prob` (float, optional): Base Model Hidden Layer(s) 
            Dropout probability. Defaults to 0.1.

            `attention_probs_dropout_prob` (float, optional): Base Model 
            Attention Layer(s) Dropout probability. Defaults to 0.1.

            `classifier_dropout_value` (float, optional):Base Model 
            Classification Layer(s) Dropout probability. Defaults to None.

            `filter_numeric_wikiann_rows` (bool, optional): Option to filter 
            numeric Wikiann Data rows. Defaults to True.

            `concat_with_wikiann` (bool, optional): Option to concat input data 
            with Wikiann Data. Defaults to True.

            `early_stopping_patience` (int, optional): Early Stopping Patience. 
            Defaults to None.

            `hyperparameter_tuning` (bool, optional): Option to run 
            hyperparameter tuning. Defaults to False.

            `layers_to_freeze` (List[Union[str, int]], optional): List of 
            layers to freeze in model. Defaults to None.

            `overwrite_output_dir` (bool, optional): Option to overwrite the 
            content of the output directory. Defaults to False.

            `evaluation_strategy` (str or trainer_utils.IntervalStrategy, 
            optional): The evaluation strategy to adopt during training. 
            Defaults to 'no'.
            
                Possible values are:

                    - 'no': No evaluation is done during training.
                    - 'steps': Evaluation is done (and logged) every `eval_steps`.
                    - 'epoch': Evaluation is done at the end of each epoch.
            
            `prediction_loss_only` (bool, optional): When performing evaluation
            and generating predictions, only returns the loss. Defaults to 
            False.

            `per_device_train_batch_size` (int, optional): The batch size per 
            GPU/TPU core/CPU for training. Defaults to 8.

            `per_device_eval_batch_size` (int, optional): The batch size per 
            GPU/TPU core/CPU for evaluation. Defaults to 8.

            `gradient_accumulation_steps` (int, optional): Number of updates 
            steps to accumulate the gradients for, before performing a 
            backward/update pass. Defaults to 1.

                <Tip>

                When using gradient accumulation, one step is counted as one 
                step with backward pass. Therefore, logging, evaluation, save 
                will be conducted every 
                `gradient_accumulation_steps * xxx_step` training examples.

                </Tip>

            `eval_accumulation_steps` (int, optional): Number of 
            predictions steps to accumulate the output tensors for, before 
            moving the results to the CPU. If left unset, the whole predictions
            are accumulated on GPU/TPU before being moved to the CPU (faster 
            but requires more memory).

            `eval_delay` (float, optional): Number of epochs or steps to wait 
            for before the first evaluation can be performed, depending on the 
            evaluation_strategy.

            `learning_rate` (float, optional): The initial learning rate for 
            AdamW optimizer. Defaults to 5e-5.

            `weight_decay` (float, optional): The weight decay to apply (if not 
            zero) to all layers except all bias and LayerNorm weights in AdamW 
            optimizer. Defaults to 0.

            `adam_beta1` (float, optional): The beta1 hyperparameter for the 
            AdamW optimizer. Defaults to 0.9.

            `adam_beta2` (float, optional): The beta2 hyperparameter for the 
            AdamW optimizer. Defaults to 0.999.

            `adam_epsilon` (float, optional):The epsilon hyperparameter for the 
            AdamW optimizer. Defaults to 1e-8.

            `max_grad_norm` (float, optional): Maximum gradient norm (for 
            gradient clipping). Defaults to 1.0.

            `num_train_epochs`(float, optional): Total number of training 
            epochs to perform (if not an integer, will perform the decimal part
            percents of the last epoch before stopping training). Defaults to 
            3.0.

            `max_steps` (int, optional): If set to a positive number, the total
            number of training steps to perform. Overrides `num_train_epochs`. 
            In case of using a finite iterable dataset the training may stop 
            before reaching the set number of steps when all data is exhausted.
            Defaults to -1.

            `lr_scheduler_type` (str or SchedulerType, optional): The scheduler
            type to use. See the documentation of SchedulerType for all 
            possible values. Defaults to 'linear'.
            
            `warmup_ratio` (float, optional): Ratio of total training steps 
            used for a linear warmup from 0 to `learning_rate`. Defaults to 
            0.0.

            `warmup_steps` (int, optional): Number of steps used for a linear 
            warmup from 0 to `learning_rate`. Overrides any effect of 
            `warmup_ratio`. Defaults to 0.

            `log_level` (str, optional): Logger log level to use on the main 
            process. Possible choices are the log levels as strings: 'debug', 
            'info', 'warning', 'error' and 'critical', plus a 'passive' level 
            which doesn't set anything and lets the application set the level.
            Defaults to 'passive'.

            `log_level_replica` (str, optional): Logger log level to use on 
            replicas. Same choices as `log_level`.  Defaults to 'passive'.

            `log_on_each_node` (bool, optional): In multinode distributed 
            training, whether to log using `log_level` once per node, or only 
            on the main node. Defaults to True.

            `logging_dir` (str, optional): 
            [TensorBoard](https://www.tensorflow.org/tensorboard) log 
            directory. Will default to 
            'output_dir/runs/CURRENT_DATETIME_HOSTNAME'.

            `logging_strategy` (str or trainer_utils.IntervalStrategy, 
            optional): The logging strategy to adopt during training.  Defaults
            to 'steps'
                
                Possible values are:

                    - 'no': No logging is done during training.
                    - 'epoch': Logging is done at the end of each epoch.
                    - 'steps': Logging is done every `logging_steps`.

            `logging_first_step` (bool, optional): Whether to log and evaluate 
            the first `global_step` or not. Defaults to False.

            `logging_steps` (int, optional): Number of update steps between two 
            logs if `logging_strategy`='steps'. Defaults to 500.

            `logging_nan_inf_filter` (bool, optional): Whether to filter `nan` 
            and `inf` losses for logging. If set to True the loss of every step 
            that is `nan` or `inf` is filtered and the average loss of the 
            current logging window is taken instead. Defaults to True.

                <Tip>

                `logging_nan_inf_filter` only influences the logging of loss 
                values, it does not change the behavior the gradient is 
                computed or applied to the model.

                </Tip>

            `save_strategy` (str or trainer_utils.IntervalStrategy, optional): 
            The checkpoint save strategy to adopt during training. Defaults to 
            'steps'.
            
                Possible values are:

                    - 'no': No save is done during training.
                    - 'epoch': Save is done at the end of each epoch.
                    - 'steps': Save is done every `save_steps`.

            `save_steps` (int, optional): Number of updates steps before two 
            checkpoint saves if `save_strategy`='steps'. Defaults to 500.

            `save_total_limit` (int, optional): If a value is passed, will 
            limit the total amount of checkpoints. Deletes the older 
            checkpoints in `output_dir`.

            `save_on_each_node` (bool, optional): When doing multi-node 
            distributed training, whether to save models and checkpoints on 
            each node, or only on the main one. Defaults to False.

                This should not be activated when the different nodes use the 
                same storage as the files will be saved with the same names for 
                each node.

            `no_cuda` (bool, optional): Whether to not use CUDA even when it is 
            available or not. Defaults to False.

            `seed` (int, optional): Random seed that will be set at the 
            beginning of training. To ensure reproducibility across runs, use 
            the Trainer.model_init function to instantiate the model if it has 
            some randomly initialized parameters. Defaults to 42.

            `data_seed` (int, optional): Random seed to be used with data 
            samplers. If not set, random generators for data sampling will use 
            the same seed as `seed`. This can be used to ensure reproducibility 
            of data sampling, independent of the model seed.

            `jit_mode_eval` (bool, optional): Whether or not to use PyTorch jit 
            trace for inference. Defaults to False.

            `use_ipex` (bool, optional): Use Intel extension for PyTorch when 
            it is available. [IPEX installation](https://github.com/intel/intel-extension-for-pytorch).
            Defaults to False.

            `bf16` (bool, optional): Whether to use bf16 16-bit (mixed) 
            precision training instead of 32-bit training. Requires Ampere or 
            higher NVIDIA architecture or using CPU (no_cuda). This is an 
            experimental API and it may change. Defaults to False.

            `fp16` (bool, optional): Whether to use fp16 16-bit (mixed) 
            precision training instead of 32-bit training. Defaults to False.

            `fp16_opt_level` (str, optional): For `fp16` training, Apex AMP 
            optimization level selected in ['O0', 'O1', 'O2', and 'O3']. 
            Defaults to 'O1'. See details on the [Apex documentation](https://nvidia.github.io/apex/amp).

            `fp16_backend` (str, optional): This argument is deprecated. Use 
            `half_precision_backend` instead. Defaults to 'auto'.

            `half_precision_backend` (str, optional): The backend to use for 
            mixed precision training. Defaults to 'auto'. Must be one of 'auto', 
            'cuda_amp', 'apex', 'cpu_amp'. 'auto' will use CPU/CUDA AMP or APEX 
            depending on the PyTorch version detected, while the other choices 
            will force the requested backend.

            `bf16_full_eval` (bool, optional): Whether to use full bfloat16 
            evaluation instead of 32-bit. This will be faster and save memory 
            but can harm metric values. This is an experimental API and it may 
            change. Defaults to False.

            `fp16_full_eval` (bool, optional): Whether to use full float16 
            evaluation instead of 32-bit. This will be faster and save memory 
            but can harm metric values. Defaults to False.

            `tf32` (bool, optional): Whether to enable the TF32 mode, available 
            in Ampere and newer GPU architectures. The default value depends on 
            PyTorch's version default of `torch.backends.cuda.matmul.allow_tf32`. 
            For more details please refer to the [TF32](https://huggingface.co/docs/transformers/performance#tf32) 
            documentation. This is an experimental API and it may change.

            `local_rank` (int, optional): Rank of the process during 
            distributed training. Defaults to -1.

            `xpu_backend` (str, optional): The backend to use for xpu 
            distributed training. Must be one of 'mpi' or 'ccl' or 'gloo'.

            `tpu_num_cores` (int, optional): When training on TPU, the number
            of TPU cores (automatically passed by launcher script).

            `dataloader_drop_last` (bool, optional): Whether to drop the last 
            incomplete batch (if the length of the dataset is not divisible by 
            the batch size) or not. Defaults to False.

            `eval_steps` (int, optional): Number of update steps between two 
            evaluations if `evaluation_strategy`='steps'. Will default to the 
            same value as `logging_steps` if not set.

            `dataloader_num_workers` (int, optional): Number of subprocesses to 
            use for data loading (PyTorch only). Defaults to 0. 0 means that 
            the data will be loaded in the main process.

            `past_index` (int, optional): Some models like 
            [TransformerXL](https://github.com/huggingface/transformers/tree/main/src/transformers/models/transfo_xl) or 
            [XLNet](https://github.com/huggingface/transformers/tree/main/src/transformers/models/xlnet) can make use of the past hidden states
            for their predictions. If this argument is set to a positive int, 
            the `Trainer` will use the corresponding output (usually index 2) 
            as the past state and feed it to the model at the next training 
            step under the keyword argument `mems`. Defaults to -1.

            `run_name` (str, optional): A descriptor for the run. Typically 
            used for [wandb](https://www.wandb.com/) and [mlflow](https://www.mlflow.org/)
            logging.

            `disable_tqdm` (bool, optional): Whether or not to disable the tqdm
              progress bars and table of metrics produced by 
              notebook.NotebookTrainingTracker in Jupyter Notebooks. Will 
              default to True if the logging level is set to warn or lower 
              (default), False otherwise.

            `remove_unused_columns` (bool, optional): Whether or not to 
            automatically remove the columns unused by the model forward method. 
            Defaults to True.

                (Note that this behavior is not implemented for TFTrainer yet.)

            `label_names` (List[str], optional): The list of keys in your 
            dictionary of inputs that correspond to the labels.

                Will eventually default to ['labels'] except if the model used 
                is one of the `XxxForQuestionAnswering` in which case it will 
                default to ['start_positions', 'end_positions'].

            `load_best_model_at_end` (bool, optional): Whether or not to load 
            the best model found during training at the end of training.  
            Defaults to False.

                <Tip>

                When set to True, the parameters `save_strategy` needs to be 
                the same as `evaluation_strategy`, and in the case it is 
                'steps', `save_steps` must be a round multiple of `eval_steps`.

                </Tip>

            `metric_for_best_model` (str, optional): Use in conjunction with 
            `load_best_model_at_end` to specify the metric to use to compare 
            two different models. Must be the name of a metric returned by the 
            evaluation with or without the prefix 'eval_'. Will default to 
            'loss' if unspecified and `load_best_model_at_end=True` (to use the
            evaluation loss).

                If you set this value, `greater_is_better` will default to 
                True. Don't forget to set it to False if your metric is better 
                when lower.

            `greater_is_better` (bool, optional): Use in conjunction with 
            `load_best_model_at_end` and `metric_for_best_model` to specify if 
            better models should have a greater metric or not. Will default to:

                - True if `metric_for_best_model` is set to a value that isn't 'loss' or 'eval_loss'.
                - False if `metric_for_best_model` is not set, or set to 'loss' or 'eval_loss'.

            `ignore_data_skip` (bool, optional): When resuming training, 
            whether or not to skip the epochs and batches to get the data 
            loading at the same stage as in the previous training. Defaults to 
            False. If set to True, the training will begin faster (as that 
            skipping step can take a long time) but will not yield the same 
            results as the interrupted training would have.

            `sharded_ddp` (bool, str or list of trainer_utils.ShardedDDPOption, 
            optional): Use Sharded DDP training from 
            [FairScale](https://github.com/facebookresearch/fairscale) (in 
            distributed training only). This is an experimental feature. 
            Defaults to False.

                A list of options along the following:

                - 'simple': to use first instance of sharded DDP released by fairscale (`ShardedDDP`) similar to ZeRO-2.
                - 'zero_dp_2': to use the second instance of sharded DPP released by fairscale (`FullyShardedDDP`) in Zero-2 mode (with `reshard_after_forward=False`).
                - 'zero_dp_3': to use the second instance of sharded DPP released by fairscale (`FullyShardedDDP`) in Zero-3 mode (with `reshard_after_forward=True`).
                - 'offload': to add ZeRO-offload (only compatible with 'zero_dp_2' and 'zero_dp_3').

                If a string is passed, it will be split on space. If a bool is 
                passed, it will be converted to an empty list for False and 
                'simple' for True.

            `fsdp` (bool, str or list of trainer_utils.FSDPOption, optional): 
            Use PyTorch Distributed Parallel Training (in distributed training 
            only). Defaults to False.

                A list of options along the following:

                - 'full_shard': Shard parameters, gradients and optimizer states.
                - 'shard_grad_op': Shard optimizer states and gradients.
                - 'offload': Offload parameters and gradients to CPUs (only compatible with 'full_shard' and 'shard_grad_op').
                - 'auto_wrap': Automatically recursively wrap layers with FSDP using `default_auto_wrap_policy`.

            `fsdp_min_num_params` (int, optional): FSDP's minimum number of 
            parameters for Default Auto Wrapping. (useful only when `fsdp` 
            field is passed). Defaults to 0.

            `deepspeed` (str or dict, optional): Use 
            [Deepspeed](https://github.com/microsoft/deepspeed). This is an 
            experimental feature and its API may evolve in the future. The 
            value is either the location of DeepSpeed json config file (e.g., 
            ds_config.json) or an already loaded json file as a dict.

            `label_smoothing_factor` (float, optional): The label smoothing 
            factor to use.  Defaults to 0.0. Zero means no label smoothing, 
            otherwise the underlying onehot-encoded labels are changed from 
            0s and 1s to `label_smoothing_factor/num_labels` and 
            `1 - label_smoothing_factor + label_smoothing_factor/num_labels` 
            respectively.

            `debug` (str or list of debug_utils.DebugOption, optional): 
            Enable one or more debug features. This is an experimental feature.
            Defaults to ''.

                Possible options are:

                - 'underflow_overflow': detects overflow in model's input/outputs and reports the last frames that led to the event
                - 'tpu_metrics_debug': print debug metrics on TPU

                The options should be separated by whitespaces.

            `optim` (str or training_args.OptimizerNames, optional): The 
            optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, 
            adamw_anyprecision or adafactor.  Defaults to 'adamw_hf'.

            `optim_args` (str, optional): Optional arguments that are supplied 
            to AnyPrecisionAdamW.

            `adafactor` (bool, optional): This argument is deprecated. Use 
            `--optim adafactor` instead.  Defaults to False.

            `group_by_length` (bool, optional): Whether or not to group 
            together samples of roughly the same length in the training dataset
            (to minimize padding applied and be more efficient). Only useful if
            applying dynamic padding. Defaults to False.

            `length_column_name` (str, optional): Column name for precomputed 
            lengths. If the column exists, grouping by length will use these 
            values rather than computing them on train startup. Ignored unless 
            `group_by_length` is True and the dataset is an instance of 
            `Dataset`. Defaults to 'length'

            `report_to` (str or List[str], optional): The list of integrations 
            to report the results and logs to. Supported platforms are 
            'azure_ml', 'comet_ml', 'mlflow', 'neptune', 'tensorboard',
            'clearml' and 'wandb'. Use 'all' to report to all integrations 
            installed, 'none' for no integrations. Defaults to 'all'.

            `ddp_find_unused_parameters` (bool, optional): When using 
            distributed training, the value of the flag `find_unused_parameters` 
            passed to `DistributedDataParallel`. Will default to False if 
            gradient checkpointing is used, True otherwise.

            `ddp_bucket_cap_mb` (int, optional): When using distributed 
            training, the value of the flag `bucket_cap_mb` passed to 
            `DistributedDataParallel`.

            `dataloader_pin_memory` (bool, optional): Whether you want to pin 
            memory in data loaders or not. Defaults to True.

            `skip_memory_metrics` (bool, optional): Whether to skip adding of 
            memory profiler reports to metrics. This is skipped by default 
            because it slows down the training and evaluation speed. 
            Defaults to True.

            `push_to_hub` (bool, optional): Whether or not to push the model to
            the Hub every time the model is saved. Defaults to False. If this 
            is activated, `output_dir` will begin a git directory synced with 
            the repo (determined by `hub_model_id`) and the content will be 
            pushed each time a save is triggered (depending on your 
            `save_strategy`). Calling Trainer.save_model will also trigger a 
            push.

                <Tip>

                If `output_dir` exists, it needs to be a local clone of the 
                repository to which the Trainer will be pushed.

                </Tip>

            `resume_from_checkpoint` (str, optional): The path to a folder with
            a valid checkpoint for your model. This argument is not directly
            used by Trainer, it's intended to be used by your training/
            evaluation scripts instead. See the [example scripts](https://github.com/huggingface/transformers/tree/main/examples)
            for more details.

            `hub_model_id` (str, optional): The name of the repository to keep 
            in sync with the local *output_dir*. It can be a simple model ID in
            which case the model will be pushed in your namespace. Otherwise it
            should be the whole repository name, for instance 'user_name/model',
            which allows you to push to an organization you are a member of 
            with 'organization_name/model'. Will default to 
            `user_name/output_dir_name` with *output_dir_name* being the name
            of `output_dir`. Defaults to the name of `output_dir`.

            `hub_strategy` (str or trainer_utils.HubStrategy, optional): 
            Defines the scope of what is pushed to the Hub and when. Defaults 
            to 'every_save'. 
            
                Possible values are:

                - 'end': push the model, its configuration, the tokenizer (if passed along to the Trainer) and a draft of a model card when the ~Trainer.save_model method is called.
                - 'every_save': push the model, its configuration, the tokenizer (if passed along to the Trainer) and a draft of a model card each time there is a model save. The pushes are asynchronous to not block training, and in case the save are very frequent, a new push is only attempted if the previous one is finished. A last push is made with the final model at the end of training.
                - 'checkpoint': like 'every_save' but the latest checkpoint is also pushed in a subfolder named last-checkpoint, allowing you to resume training easily with `trainer.train(resume_from_checkpoint='last-checkpoint')`.
                - 'all_checkpoints': like 'checkpoint' but all checkpoints are pushed like they appear in the output folder (so you will get one checkpoint folder per folder in your final repository)

            `hub_token` (str, optional): The token to use to push the model to 
            the Hub. Will default to the token in the cache folder obtained 
            with `huggingface-cli login`.

            `hub_private_repo` (bool, optional): If True, the Hub repo will be 
            set to private. Defaults to False.

            `gradient_checkpointing` (bool, optional): If True, use gradient 
            checkpointing to save memory at the expense of slower backward 
            pass. Defaults to False.
            
            `include_inputs_for_metrics` (bool, optional): Whether or not the 
            inputs will be passed to the `compute_metrics` function. This is 
            intended for metrics that need inputs, predictions and references 
            for scoring calculation in Metric class. Defaults to False.

            `auto_find_batch_size` (bool, optional): Whether to find a batch 
            size that will fit into memory automatically through exponential 
            decay, avoiding CUDA Out-of-Memory errors. Requires accelerate to 
            be installed (`pip install accelerate`). Defaults to False.

            `full_determinism` (bool, optional): If True, 
            enable_full_determinism is called instead of set_seed to ensure 
            reproducible results in distributed training. Defaults to False.

            `torchdynamo` (str, optional): If set, the backend compiler for 
            TorchDynamo. Possible choices are 'eager', 'aot_eager', 'inductor',
            'nvfuser', 'aot_nvfuser', 'aot_cudagraphs', 'ofi', 'fx2trt', 
            'onnxrt' and 'ipex'.

            `ray_scope` (str, optional): The scope to use when doing 
            hyperparameter search with Ray. Defaults to 'last'. Ray will then 
            use the last checkpoint of all trials, compare those, and select 
            the best one. However, other options are also available. See the 
            [Ray documentation](https://docs.ray.io/en/latest/tune/index.html) 
            for more options.

            `ddp_timeout` (int, optional): The timeout for 
            `torch.distributed.init_process_group` calls, used to avoid GPU 
            socket timeouts when performing slow operations in distributed 
            runnings. Please refer the [PyTorch documentation](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) 
            for more information. Defaults to 1800.

            `use_mps_device` (bool, optional): Whether to use Apple Silicon 
            chip based `mps` device. Defaults to False.

            `torch_compile` (bool, optional): Whether or not to compile the 
            model using [PyTorch 2.0 torch.compile](https://pytorch.org/get-started/pytorch-2.0/) 
            (requires a nighlty install of PyTorch). Defaults to False. If set,
            the backend will default to 'inductor' (can be customized with 
            `torch_compile_backend`) and the mode will default to 'default' 
            (can be customized with `torch_compile_mode`).

            `torch_compile_backend` (str, optional): The backend to use in 
            `torch.compile`. If set to any value, `torch_compile` will be set 
            to True. Possible choices are 'eager', 'aot_eager', 'inductor', 
            'nvfuser', 'aot_nvfuser', 'aot_cudagraphs', 'ofi', 'fx2trt', 
            'onnxrt' and 'ipex'.

            `torch_compile_mode` (str, optional): The mode to use in 
            `torch.compile`. If set to any value, `torch_compile` will be set 
            to True. Possible choices are 'default', 'reduce-overhead' and 
            'max-autotune'.
        """

        # Set attributes to default values and update the ones set using kwargs
        self.__dict__.update(DEFAULT_MODELLING_PARAMETERS)
        self.__dict__.update(kwargs)

        if self.report_to == 'wandb':
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

        self._dataset_distribution_output_path = os.path.join(
            DATASET_DISTRIBUTION_OUTPUT_FOLDER,
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

        self._dataset_distribution_output = os.path.join(
            self._dataset_distribution_output_path,
            f'{self._model_name}_dataset_distribution_{self._timestamp}.csv'
        )

        setup_folders(
            folders=[
                self._training_history_output_path,
                self._dataset_distribution_output_path
            ]
        )

        # endregion

        # region Model Variables

        self._tokenizer = RobertaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=BASE_TOKENIZER
        )

        self._model_config = AutoConfig.from_pretrained(BASE_MODEL)

        self._modify_model_configuration()

        self._model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=BASE_MODEL,
            config=self._model_config
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
            seed=(
                self.seed if not self.hyperparameter_tuning else
                random.randint(0, 100)
            ),
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

        self._data = self._load_data()

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

        if self.layers_to_freeze is not None:
            self._freeze_model_layers()

        # endregion

        # region Utils

        self._plotter = Plotter()

        # endregion

    # region Data Processing

    @staticmethod
    def _filter_numeric_data_from_wikiann_dataset(
        dataset_row: Dict[str, List[Union[str, int]]]
    ) -> bool:
        """Filters rows containing numeric values in Wikiann Dataset.

        Args:
            `dataset_row` (Dict[str, List[Union[str, int]]]): Row from Wikiann
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
            `test_dataset` (Dataset): Test Dataset in Arrow Dataset format.
        """
        # Convert Arrow Dataset to Pandas DataFrame
        test_data_pd = test_dataset.to_pandas(batch_size=32)

        self._test_dataset_length = len(test_data_pd)

        # Write DataFrame to CSV File
        test_data_pd[['tokens', 'ner_tags']].to_csv(
            path_or_buf=self._test_dataset_output_path,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
        )

    def _concat_wikiann_data(
        self,
        train_and_validation_data: Dataset,
        train_and_test_data: Dataset,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Joins Wikiann data to Input Dataset. 

        Args:
            `train_and_validation_data` (Dataset): Train and Evaluation Dataset 
            split.

            `train_and_test_data` (Dataset): Train and Test Dataset split.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: Joint Datasets.
        """
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

            if self.filter_numeric_wikiann_rows:
                # Filter numeric values from Wikiann Dataset
                wikiann_data[split] = wikiann_data[split].filter(
                    lambda row:
                    self._filter_numeric_data_from_wikiann_dataset(
                        dataset_row=row
                    )
                )

        # Create concatenated Datasets
        train_dataset = concatenate_datasets(
            [train_and_validation_data['train'], wikiann_data['train']]
        )
        validation_dataset = concatenate_datasets(
            [train_and_validation_data['test'], wikiann_data['validation']]
        )
        test_dataset = concatenate_datasets(
            [train_and_test_data['test'], wikiann_data['test']]
        )

        # Filter out rows containing apostrophes
        regex = re.compile(r'(.*\'.*)')
        regex_match = np.vectorize(lambda x: bool(regex.match(x)))

        test_dataset_pd = test_dataset.to_pandas(batch_size=32)

        test_dataset_pd = test_dataset_pd[
            test_dataset_pd['tokens'].apply(lambda x: not any(regex_match(x)))
        ].drop('__index_level_0__', axis=1)

        test_dataset = Dataset.from_pandas(test_dataset_pd)

        return train_dataset, validation_dataset, test_dataset

    def _load_data(
        self,
        dataset_path: str = ANNOTATED_NBS_SENTENCES_DATASET,
    ) -> DatasetDict:
        """Loads NBS Dataset (and adds Wikiann Dataset if needed) to 
        DatasetDict.

        Args:
            `dataset_path` (str, optional): Annotated input file path. Defaults 
            to ANNOTATED_NBS_SENTENCES_DATASET.

        Returns:
            DatasetDict: Train, Validation and Test Datasets.
        """
        modeling_columns = ['text_tokens', 'fixed_ner_tags']

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

        if self.concat_with_wikiann:
            (
                train_dataset,
                validation_dataset,
                test_dataset
            ) = self._concat_wikiann_data(
                train_and_validation_data=train_valid,
                train_and_test_data=train_test,
            )

        else:
            train_dataset = train_valid['train']
            validation_dataset = train_valid['test'],
            validation_dataset = validation_dataset[0]
            test_dataset = train_test['test']

        self._generate_dataset_distribution_data(
            train=train_dataset,
            validation=validation_dataset,
            test=test_dataset
        )

        self._save_test_data_to_csv(test_dataset)

        # Create DatasetDict of 3 (not concatenated) Dataset and apply
        # tokenizer to them
        return DatasetDict(
            {
                'train': train_dataset,
                'validation': validation_dataset,
                'test': test_dataset,
            }
        ).map(
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
            `dataset_row` (Dict[str, List[Union[str, int]]]): Dataset row to 
            tokenize.

            `label_all_tokens` (bool, optional): Option to label every token. 
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

    def _tune_ner_model(self, number_of_trials: int = 10) -> None:
        """Executes Hyperparameter Tuning.

        Args:
            `number_of_trials` (int, optional): Number of trials. Defaults to 
            10.
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

        self._trainer.hyperparameter_search(
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
        )

    def train(self) -> None:
        """Runs Model Training Process."""
        setup_folders(folders=[self._model_output_folder])

        if self.hyperparameter_tuning:
            self._tune_ner_model(number_of_trials=1)
        else:
            self._trainer.train()
            self._save_model()

        wandb.finish()

    # endregion

    # region Model Evaluation

    def _plot_model_metrics(self) -> None:
        """Plots simple Model (Training History) Metrics"""
        if os.path.exists(self._model_trainer_state_path):
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
        model_name: str = None,
        plot_history: bool = False
    ) -> None:
        """Evaluates Model performance on Test Dataset."""
        if os.listdir(self._model_output_folder):
            if not self.hyperparameter_tuning and plot_history:
                # Plot Model (Training History) Metrics
                self._plot_model_metrics()

            # Send Model and Test Data to Annotator for additional Classification
            # evaluation
            annotator = Annotator(
                input_data_filepath=(
                    self._test_dataset_output_path
                    if input_data_filepath is None else input_data_filepath
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
                model_name=(
                    self._model_name if model_name is None else model_name,
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    self._model_output_folder if model_input_folder is None
                    else model_input_folder
                ),
            )

            annotator()
        else:
            os.rmdir(self._model_output_folder)

    # endregion

    # region Model Prediction

    def predict(self, prediction_input: str) -> None:
        """Generates Model predictions for given input.

        Args:
            `prediction_input` (str): Prediction input string.
        """
        ner_pipeline = pipeline(
            'ner',
            model=AutoModelForTokenClassification.from_pretrained(
                self._model_output_folder
            ),
            tokenizer=self._tokenizer
        )

        classifications = ner_pipeline(prediction_input)

        display_ner_classification(
            input_sentence=prediction_input,
            classifications=classifications
        )

    # endregion

    # region Model Utils

    def _update_folder_paths(self) -> None:
        """Updates path variables based on model version."""
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

        self._dataset_distribution_output_path = os.path.join(
            self._dataset_distribution_output_path,
            self._model_name
        )

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

    def _modify_model_configuration(self) -> None:
        """Applies settings to model configuration file."""
        self._model_config.hidden_dropout_prob = self.hidden_dropout_prob
        self._model_config.attention_probs_dropout_prob = (
            self.attention_probs_dropout_prob
        )
        self._model_config.classifier_dropout = self.classifier_dropout_value
        self._model_config.num_labels = len(NER_LABELS_LIST)
        self._model_config.id2label = NER_LABELS_STR_KEY
        self._model_config.label2id = INVERTED_NER_LABELS

    def _freeze_model_layers(self) -> None:
        """Freezes specific layers of base model."""
        print(f'Frozen Layers: {self.layers_to_freeze}\n')

        for layer in self.layers_to_freeze:
            if layer == 'embeddings':
                for param in self._model.roberta.embeddings.parameters():
                    param.requires_grad = False
            elif layer == 'classifier':
                for param in self._model.parameters():
                    param.requires_grad = False
            else:
                for param in (
                    self._model.roberta.encoder.layer[layer].parameters()
                ):
                    param.requires_grad = False

        for name, param in self._model.named_parameters():
            print(name, param.requires_grad)

    def _generate_dataset_distribution_data(
        self,
        train: Dataset,
        validation: Dataset,
        test: Dataset
    ) -> None:
        """Generates and saves Dataset value distribution information.

        Args:
            `train` (Dataset): Training Dataset to process.
            
            `validation` (Dataset): Validation Dataset to process.
            
            `test` (Dataset): Testing Dataset to process.
        """
        total_rows = sum([train.num_rows, validation.num_rows, test.num_rows])

        train_tokens = list(itertools.chain(*train['ner_tags']))
        validation_tokens = list(itertools.chain(*validation['ner_tags']))
        test_tokens = list(itertools.chain(*test['ner_tags']))

        distribution_data = pd.DataFrame(
            data={
                'Subset': ['train', 'validation', 'test'],
                'Num_of_Rows': [
                    train.num_rows, validation.num_rows, test.num_rows
                ],
                'Num_of_Tokens': [
                    len(train_tokens), len(validation_tokens), len(test_tokens)
                ],
                'Split': [
                    train.num_rows / total_rows,
                    validation.num_rows / total_rows,
                    test.num_rows / total_rows
                ],
                **{
                    NER_LABELS[key]: [
                        value,
                        Counter(validation_tokens)[key],
                        Counter(test_tokens)[key]
                    ] for key, value in Counter(train_tokens).items()
                }
            }
        ).reindex(
            columns=[
                'Subset',
                'Num_of_Rows',
                'Num_of_Tokens',
                'Split',
                *NER_LABELS_LIST
            ]
        )

        distribution_data.to_csv(
            self._dataset_distribution_output,
            index=False
        )

    def _compute_metrics(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        """Computes metrics on predictions.

        Args:
            `eval_preds` (EvalPrediction): Predictions on the Evaluation set.

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

    def _save_model(self) -> None:
        """Saves Model."""
        if not self.hyperparameter_tuning:
            # Save Model Weights
            self._model.save_pretrained(self._model_output_folder)

        # Save Tokenizer
        self._tokenizer.save_pretrained(self._model_output_folder)

        # Save Trainer State
        self._trainer.save_state()

    def __call__(self) -> None:
        """Make SlovakBertNerModel object callable."""
        self.train()
        self.evaluate(plot_history=True)

    # endregion
