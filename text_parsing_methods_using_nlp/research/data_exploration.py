import pandas as pd
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict


def load_hugging_face_dataset(
    dataset_name: str = 'wikiann',
    configuration: str = 'sk'
) -> DatasetDict:
    """Loads specific dataset with configuration.

    Args:
        dataset_name (str): Dataset name. Defaults to 'wikiann'.
        configuration (str): Configuration name. Defaults to 'sk'.

    Returns:
        DatasetDict: Loaded Dataset.
    """

    return load_dataset(dataset_name, configuration)


if __name__ == '__main__':
    wikiann_dataset = load_hugging_face_dataset()

    train = pd.DataFrame(wikiann_dataset['train'])
    validation = pd.DataFrame(wikiann_dataset['validation'])
    test = pd.DataFrame(wikiann_dataset['test'])

    print(train.head())
