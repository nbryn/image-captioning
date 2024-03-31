import logging
import os
from nltk.translate.meteor_score import meteor_score
from src.sacrebleu import corpus_bleu


def configure_logger(log_folder="./logs"):
    # Set root logger level to DEBUG

    os.makedirs(log_folder, exist_ok=True)

    logging.root.setLevel(logging.DEBUG)

    # Create file handlers for different log levels
    debug_handler = logging.FileHandler(f'{log_folder}/debug.log')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(SeverityFilter(logging.DEBUG))

    info_handler = logging.FileHandler(f'{log_folder}/info.log')
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(SeverityFilter(logging.INFO))

    warning_handler = logging.FileHandler(f'{log_folder}/warning.log')
    warning_handler.setLevel(logging.WARNING)
    warning_handler.addFilter(SeverityFilter(logging.WARNING))

    error_handler = logging.FileHandler(f'{log_folder}/error.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(SeverityFilter(logging.ERROR))

    critical_handler = logging.FileHandler(f'{log_folder}/critical.log')
    critical_handler.setLevel(logging.CRITICAL)
    critical_handler.addFilter(SeverityFilter(logging.CRITICAL))

    # Create formatters and add them to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    debug_handler.setFormatter(formatter)
    info_handler.setFormatter(formatter)
    warning_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    critical_handler.setFormatter(formatter)

    # Add the handlers to the root logger, which applies to all loggers
    logging.root.addHandler(debug_handler)
    logging.root.addHandler(info_handler)
    logging.root.addHandler(warning_handler)
    logging.root.addHandler(error_handler)
    logging.root.addHandler(critical_handler)


class SeverityFilter(logging.Filter):
    def __init__(self, severity: int):
        self.severity = severity

    def filter(self, record):
        return record.levelno == self.severity


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log(message: str, color=bcolors.OKBLUE):
    print(f"{color}{message}", f"{bcolors.ENDC}")
    logging.info(message)


def error(message: str):
    log(message, bcolors.FAIL)
    logging.error(message)


def success(message: str):
    log(message, bcolors.OKGREEN)
    logging.info(message)


def print_prediction(prediction, label, context, custom_msg=""):
    bleu_score = round(get_bleu_score(label, prediction), 4)
    meteor_score = round(get_meteor_score(label, prediction), 4)
    msg = f":: Pred: {bcolors.OKBLUE}{prediction}{bcolors.ENDC}? :: Label: {bcolors.OKGREEN}{label}{bcolors.ENDC} \n :: Context: {bcolors.OKCYAN}{context}{bcolors.ENDC} \n :: Bleu: {bleu_score} :: Meteor: {meteor_score} {custom_msg}"
    print(msg)


def get_bleu_score(questions, prediction, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Default is BLEU-4
    """
    references = [[question] for question in questions] if isinstance(questions, list) else [
        [questions]]
    return corpus_bleu([prediction], references).score


def get_meteor_score(questions, prediction):
    """
    """
    references = [question.split() for question in questions] if isinstance(
        questions, list) else [questions.split()]

    return meteor_score(references, prediction.split()) * 100
