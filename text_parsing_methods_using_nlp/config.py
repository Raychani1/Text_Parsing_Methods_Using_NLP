import os

###############################################################################
#                                   Data                                      #
###############################################################################

ROOT_DATA_FOLDER = os.path.join(os.getcwd(),'data')

RAW_DATA_FOLDER = os.path.join(ROOT_DATA_FOLDER, 'raw')

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

DATA_CONFIG = {
    'NBS_sentence': {
        'raw_input_data_path': os.path.join(
            RAW_DATA_FOLDER,
            'NBS_sentence.csv'
        ),
        'dataset_size': 8445,
        'file_name': 'NBS_sentence',
    }
}


###############################################################################
#                           Data Pre-Processing                               #
###############################################################################

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


###############################################################################
#                               Text Lemmas                                   #
###############################################################################

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


###############################################################################
#                              Named Entities                                 #
###############################################################################

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

INVERTED_NER_LABELS = {label: key for key, label in NER_LABELS.items()}


###############################################################################
#                                   Output                                    #
###############################################################################

ROOT_OUTPUT_FOLDER = os.path.join(os.getcwd(), 'output')

CLASSIFICATION_REPORTS_OUTPUT_FOLDER = os.path.join(
    ROOT_OUTPUT_FOLDER,
    'classification_reports'
)

CLASSIFICATION_REPORTS_OUTPUT_FOLDER_SLOVAKBERT_NER_VERSION = os.path.join(
    CLASSIFICATION_REPORTS_OUTPUT_FOLDER,
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

TOKENIZERS_OUTPUT_FOLDER = os.path.join(ROOT_OUTPUT_FOLDER, 'tokenizers')
