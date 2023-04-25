from transformers import IntervalStrategy

MODEL_CONFIG_V_POC = {
    'version': '_POC',
    'hyperparameter_tuning':  False,
    'concat_with_wikiann': False,
    'do_train': True,
    'do_eval': True,
    'do_predict': True,
    'evaluation_strategy': IntervalStrategy.EPOCH,
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 4,
    'num_train_epochs': 15,
    'logging_strategy': IntervalStrategy.EPOCH,
    'save_strategy': IntervalStrategy.EPOCH,
    'save_total_limit': 2,
    'eval_steps': 500,
    'report_to': 'wandb',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',
}


MODEL_CONFIG_V0_0_5 = {
    'version': '0.0.5',
    'early_stopping_patience': 3,
    'hyperparameter_tuning':  False,
    'do_train': True,
    'do_eval': True,
    'do_predict': True,
    'evaluation_strategy': IntervalStrategy.EPOCH,
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 4,
    'learning_rate': 0.000024981604754,
    'weight_decay': 0.285214291922975,
    'num_train_epochs': 15,
    'logging_strategy': IntervalStrategy.EPOCH,
    'save_strategy': IntervalStrategy.EPOCH,
    'save_total_limit': 2,
    'eval_steps': 500,
    'report_to': 'wandb',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',
}
