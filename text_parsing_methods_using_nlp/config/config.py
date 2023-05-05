import os

# region Data

ROOT_DATA_FOLDER = os.path.join(os.getcwd(), 'data')

MANUAL_CORRECTION_DATA_FOLDER = os.path.join(
    ROOT_DATA_FOLDER,
    'manual_correction_files'
)

ANNOTATION_PROCESS_OUTPUT_FOLDER = os.path.join(
    MANUAL_CORRECTION_DATA_FOLDER,
    'annotation_process_output'
)

MODEL_TEST_DATASET_FOLDER = os.path.join(
    ROOT_DATA_FOLDER,
    'model_test_dataset'
)

ANNOTATED_DATA_FOLDER = os.path.join(ROOT_DATA_FOLDER, 'annotated')

ANNOTATED_DATA_FOLDER_SLOVAKBERT_NER_VERSION = os.path.join(
    ANNOTATED_DATA_FOLDER,
    'SlovakBERT_NER_Model'
)

ANNOTATED_NBS_SENTENCES_DATASET = os.path.join(
    ANNOTATED_DATA_FOLDER,
    'Crabz_-_SlovakBERT_NER_Model',
    'NBS_sentence_450_Annotated_18_03_2023__14_44_55.csv'
)

DATASET_DISTRIBUTION_OUTPUT_FOLDER = os.path.join(
    ROOT_DATA_FOLDER,
    'distribution'
)

DATA_CONFIG = {
    'NBS_sentence': {
        'raw_input_data_path': os.path.join(
            ROOT_DATA_FOLDER,
            'raw',
            'NBS_sentence.csv'
        ),
        'dataset_size': 8445,
        'file_name': 'NBS_sentence',
    }
}

# endregion

# region Data Pre-Processing

PREPROCESSING_REGEX_RULES = {

    # Add an extra space before colon which is not a time indicator
    r'([a-zA-Z]+):([a-zA-Z]+)': r'\1 : \2',

    # Remove reference indicators
    r'(\[\s*\d+\s*\])': ' ',

    # Split ranges
    r'(\d)\s*-\s*(\d)': r'\1 až \2',

    # Replace date separators
    r'(\d{1,2})\.(\d{1,2})\.(\d{4})': r'\1 \2 \3',

    # Remove special characters
    r'''['"`‘’„“”\(\)\[\]\/(\s\-|–\s))!?;…\|]|(\.|,)\s*(\.|,)''': ' ',

    # Remove redundant colons
    r'(\s*:\s+)': ' ',

    # Remove dots from text
    r'([a-zA-Z]{2,})(\s*\.\s*)': r'\1 ',

    # Remove dots from text
    r'([áäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ%]{1,}[a-zA-Z0-9]*)\s*(\.)\s*': r'\1 ',

    # Remove dots from text
    r'(\d{1,})(\s*\.\s*)': r'\1 ',

    # Replace floating point number commas with dot notation
    r'([+-]?\d+),(\d+)': r'\1.\2',

    # Add extra space to floating point percentages
    r'([+-]?\d+\.\d+|[+-]?\d)(%)': r'\1 \2 ',

    # Remove commas from text
    ',': ' ',

    # Replace Euro symbol
    '€': 'euro ',

    # Replace excessive whitespace characters
    r'\s+': ' ',

    # Merge larger number formats together
    r'( [+-]?\d{1,3}) (\d{3}) (\d{2,3})': r'\1\2\3',
    r'( [+-]?\d{1,3}) (\d{3}) ([a-z]*)': r'\1\2 \3',

    # Remove extra space after letters
    r'(\s+[a-zA-Záäčďéíĺľňóôŕšťúýž]) \.': r'\1.',

    # Replace specific percentage value
    r'p\s*\.\s*b\s*\.': 'p.b.',

    # Fix punctuation
    r'(d|D)\.(c|C)\s*\.': r'\1.\2.',

    # Fix punctuation
    r'o.c.p\s*\.': 'o.c.p.',

    # Split email user and provider
    r'([a-z])@([a-z]*)\s+': r'\1 @ \2 ',

    # Fix company names
    r'([a-zA-z])\s*&\s*([a-zA-z])': r'\1&\2',
    r'a\.\s*s[\.]*': 'a.s.',

    # Replace specific monetary value
    'desaťtisíc': '10 tisíc',

    # Fix preprocessing result(s)
    'Česko Slovensk': 'Československ',
    'makro ekonomick': 'makroekonomick',
    'e mail': 'email',
}

# endregion

# region Text Lemmas

MONTH_LEMMAS = [
    'január',
    'február',
    'marec',
    'apríl',
    'máj',
    'jún',
    'júl',
    'august',
    'september',
    'október',
    'november',
    'december'
]

YEAR_PREFIX_LEMMAS = ['', 'štvrťrok', 'polrok', 'rok']

MONEY_LEMMAS = ['tisíc', 'mil', 'milión', 'miliarda', 'euro']

PERCENTAGE_LEMMAS = ['%', 'percento', 'p.b']

# endregion

# region Named Entities

NER_LABELS_LIST = [
    '0',
    'B-Person',
    'I-Person',
    'B-Organization',
    'I-Organization',
    'B-Location',
    'I-Location',
    'B-Date',
    'I-Date',
    'Time',
    'B-Money',
    'I-Money',
    'B-Percentage',
    'I-Percentage'
]

NER_LABELS = {key: value for key, value in enumerate(NER_LABELS_LIST)}

NER_LABELS_STR_KEY = {
    str(key): value for key, value in enumerate(NER_LABELS_LIST)
}

INVERTED_NER_LABELS = {label: key for key, label in NER_LABELS.items()}

NER_ENTITY_COLORS = {
    'PERSON': 'lightblue',
    'ORGANIZATION': 'lightcoral',
    'LOCATION': '#FFB703',
    'DATE': '#EF233C',
    'TIME': '#CCFF33',
    'MONEY': 'lightgreen',
    'PERCENTAGE': '#8D99AE'
}

# endregion

# region Modeling

BASE_MODEL = 'gerulata/slovakbert'

BASE_TOKENIZER = 'crabz/slovakbert-ner'

DEFAULT_MODELLING_PARAMETERS = {
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
    'classifier_dropout_value': None,
    'filter_numeric_wikiann_rows': True,
    'concat_with_wikiann': True,
    'early_stopping_patience': None,
    'hyperparameter_tuning':  False,
    'layers_to_freeze': None,
    'overwrite_output_dir': False,
    'do_train': False,
    'do_eval': False,
    'do_predict': False,
    'evaluation_strategy': 'no',
    'prediction_loss_only': False,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'per_gpu_train_batch_size': None,
    'per_gpu_eval_batch_size': None,
    'gradient_accumulation_steps': 1,
    'eval_accumulation_steps': None,
    'eval_delay': 0,
    'learning_rate': 0.00005,
    'weight_decay': 0,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8,
    'max_grad_norm': 1,
    'num_train_epochs': 3,
    'max_steps': -1,
    'lr_scheduler_type': 'linear',
    'warmup_ratio': 0,
    'warmup_steps': 0,
    'log_level': 'passive',
    'log_level_replica': 'passive',
    'log_on_each_node': True,
    'logging_dir': None,
    'logging_strategy': 'steps',
    'logging_first_step': False,
    'logging_steps': 500,
    'logging_nan_inf_filter': True,
    'save_strategy': 'steps',
    'save_steps': 500,
    'save_total_limit': None,
    'save_on_each_node': False,
    'no_cuda': False,
    'use_mps_device': False,
    'seed': 42,
    'data_seed': None,
    'jit_mode_eval': False,
    'use_ipex': False,
    'bf16': False,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'half_precision_backend': 'auto',
    'bf16_full_eval': False,
    'fp16_full_eval': False,
    'tf32': None,
    'local_rank': -1,
    'xpu_backend': None,
    'tpu_num_cores': None,
    'tpu_metrics_debug': False,
    'debug': '',
    'dataloader_drop_last': False,
    'eval_steps': None,
    'dataloader_num_workers': 0,
    'past_index': -1,
    'run_name': None,
    'disable_tqdm': None,
    'remove_unused_columns': True,
    'label_names': None,
    'load_best_model_at_end': False,
    'metric_for_best_model': None,
    'greater_is_better': None,
    'ignore_data_skip': False,
    'sharded_ddp': '',
    'fsdp': '',
    'fsdp_min_num_params': 0,
    'fsdp_transformer_layer_cls_to_wrap': None,
    'deepspeed': None,
    'label_smoothing_factor': 0,
    'optim': 'adamw_hf',
    'optim_args': None,
    'adafactor': False,
    'group_by_length': False,
    'length_column_name': 'length',
    'report_to': None,
    'ddp_find_unused_parameters': None,
    'ddp_bucket_cap_mb': None,
    'dataloader_pin_memory': True,
    'skip_memory_metrics': True,
    'use_legacy_prediction_loop': False,
    'push_to_hub': False,
    'resume_from_checkpoint': None,
    'hub_model_id': None,
    'hub_strategy': 'every_save',
    'hub_token': None,
    'hub_private_repo': False,
    'gradient_checkpointing': False,
    'include_inputs_for_metrics': False,
    'fp16_backend': 'auto',
    'push_to_hub_model_id': None,
    'push_to_hub_organization': None,
    'push_to_hub_token': None,
    'mp_parameters': '',
    'auto_find_batch_size': False,
    'full_determinism': False,
    'torchdynamo': None,
    'ray_scope': 'last',
    'ddp_timeout': 1800,
    'torch_compile': False,
    'torch_compile_backend': None,
    'torch_compile_mode': None
}

# endregion

# region Visualization

EVALUATION_COLUMNS = {
    'eval': [
        'eval/precision',
        'eval/recall',
        'eval/f1',
        'eval/accuracy',
    ],
    'test': [
        'test/precision',
        'test/recall',
        'test/f1',
        'test/accuracy',
    ],
    'macro': [
        'test/macro_precision',
        'test/macro_recall',
        'test/macro_f1',
    ]
}

MODEL_GENERATION_COLUMNS = [
    'Name',
    'attention_probs_dropout_prob',
    'hidden_dropout_prob',
    'classifier_dropout',
    'per_device_train_batch_size',
    'per_device_eval_batch_size',
    'learning_rate',
    'weight_decay',
    'num_train_epochs',
    'warmup_steps',
    'seed',
    'early_stopping_patience',
    'train/epoch',
    'parent_version',
    *EVALUATION_COLUMNS['eval'],
    *EVALUATION_COLUMNS['test'],
    *EVALUATION_COLUMNS['macro'],
]

# endregion

# region Output

ROOT_OUTPUT_FOLDER = os.path.join(os.getcwd(), 'output')

CLASSIFICATION_REPORTS_OUTPUT_FOLDER = os.path.join(
    ROOT_OUTPUT_FOLDER,
    'metrics',
    'classification_reports'
)

CLASSIFICATION_REPORTS_OUTPUT_FOLDER_SLOVAKBERT_NER_VERSION = os.path.join(
    CLASSIFICATION_REPORTS_OUTPUT_FOLDER,
    'SlovakBERT_NER_Model'
)

METRICS_EVALUATION_OUTPUT_FOLDER = os.path.join(
    ROOT_OUTPUT_FOLDER,
    'metrics',
    'test_metric_evaluations'
)

METRICS_EVALUATION_OUTPUT_FOLDER_SLOVAKBERT_NER_VERSION = os.path.join(
    METRICS_EVALUATION_OUTPUT_FOLDER,
    'SlovakBERT_NER_Model'
)

MODEL_OUTPUT_FOLDER = os.path.join(
    ROOT_OUTPUT_FOLDER,
    'models',
    'SlovakBERT_NER_Model',
    'versions'
)

PLOT_OUTPUT_FOLDER = os.path.join(ROOT_OUTPUT_FOLDER, 'plots')

CONFUSION_MATRICES_OUTPUT_FOLDER = os.path.join(
    PLOT_OUTPUT_FOLDER,
    'confusion_matrices'
)

CONFUSION_MATRICES_OUTPUT_FOLDER_SLOVAKBERT_NER_VERSION = os.path.join(
    CONFUSION_MATRICES_OUTPUT_FOLDER,
    'SlovakBERT_NER_Model'
)

TRAINING_HISTORIES_OUTPUT_FOLDER = os.path.join(
    PLOT_OUTPUT_FOLDER,
    'training_history',
    'SlovakBERT_NER_Model'
)

# endregion
