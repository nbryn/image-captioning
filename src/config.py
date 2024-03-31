import datetime
from enum import Enum
import argparse
import os


class CAPTION_MODELS(Enum):
    GIT_BASE = "git_base"
    BLIP = "blip"
    GIT_LARGE = "git_large"

    def __str__(self):
        """
        Needed for argparse
        """
        return self.value


class PROCESS_PAIR(Enum):
    TRUE = "true"
    FALSE = "false"

    def __str__(self):
        """
        Needed for argparse
        """
        return self.value


class MODE(Enum):
    GEN_DATA = "gen_data"
    FINE_TUNE = "fine_tune"
    TEST = "test"
    ALL = "all"
    PRESENT = "present"

    def __str__(self):
        """
        Needed for argparse
        """
        return self.value


class TRAIN_APPROACH(Enum):
    TRAIN_ALL = "train_all"  # Train on all questions > 0
    TRAIN_BEST = "train_best"  # Train on all questions >= 2
    TRAIN_RAND = "train_rand"  # For each image pair train on random selected question > 1

    def __str__(self):
        """
        Needed for argparse
        """
        return self.value


def get_timestamp():
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Get current day, hour, and minute as strings
    current_month = str(current_datetime.month)
    current_day = str(current_datetime.day)
    current_hour = str(current_datetime.hour)
    current_minute = str(current_datetime.minute)

    return f"{current_month}-{current_day}-{current_hour}H-{current_minute}M"


def get_config():
    parser = argparse.ArgumentParser()
    timestamp = get_timestamp()

    parser.add_argument("--mode", type=MODE, help="Mode to run in. Default: all",
                        default=MODE.ALL, choices=list(MODE))

    parser.add_argument("--approach", type=TRAIN_APPROACH, help="Determines the training approach. Default: train_best",
                        default=TRAIN_APPROACH.TRAIN_BEST, choices=list(TRAIN_APPROACH))

    parser.add_argument("--process_pair", type=PROCESS_PAIR, help="Either process on image or two image at a time. Default: false",
                        default=PROCESS_PAIR.TRUE, choices=list(PROCESS_PAIR))

    parser.add_argument("--caption_model", type=CAPTION_MODELS, help="image caption model to use. Default: GIT BASE",
                        default=CAPTION_MODELS.GIT_BASE, choices=list(CAPTION_MODELS))

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")

    parser.add_argument("--data_size", type=int, default=11202,
                        help="The size of the dataset (default: 11202), (max: 11202)")

    parser.add_argument("--test_size", type=float, default=0.2)

    parser.add_argument("--test_amount", type=int, default=None,
                        help="How many samples to test on. Default: none")

    parser.add_argument("--test_skip", type=int, default=None,
                        help="How many samples to skip before starting. Default: none")

    parser.add_argument("--timestamp", type=str, default=timestamp)

    parser.add_argument(
        "--model", type=str, default="bert-base-uncased", help="(default: bert-base-uncased)")

    parser.add_argument("--wandb_name", type=str,
                        default="BERT-HLSQG", help="(default: )")

    parser.add_argument("--log_path", default=f"./logs_{timestamp}")

    parser.add_argument("--model_weight_folder_path",
                        default="./weights")

    # -----For BERT model-----
    parser.add_argument("--train_data_path", type=str,
                        default=f"./bert_data/train_{timestamp}.json")

    parser.add_argument("--test_data_path", type=str,
                        default=f"./bert_data/test_{timestamp}.json")

    parser.add_argument("--num_train_epochs", type=int,
                        default=20, help="(default: 20)")

    parser.add_argument("--learning_rate", type=float,
                        default=5e-5, help="learning rage (default: 5e-5)")

    parser.add_argument("--batch_size", type=int,
                        default=32, help="(default: 32)")

    parser.add_argument("--min_delta", type=float,
                        default=0.0001, help="How much extra should be added to the validation loss. (default: 0.0001)")

    parser.add_argument("--patience", type=int,
                        default=50, help="How many times is it allowed to increase loss. (default: 50)")

    """Test options"""
    parser.add_argument("--max_question_token_len",
                        type=int, default=20, help="(default: 20)")

    parser.add_argument("--max_len", type=int, default=512,
                        help="(bert max token len, default: 512)")

    # -----Not frequently used-----
    parser.add_argument("--save_txt_path",
                        type=str, default=f"./predictions/{timestamp}_predictions.txt")

    parser.add_argument("--metrics_path",
                        type=str, default=f"./metrics/{timestamp}_metrics.txt")

    parser.add_argument("--loss_plot_path",
                        type=str, default=f"./metrics/{timestamp}loss.png")

    os.makedirs("./predictions", exist_ok=True)
    os.makedirs("./metrics", exist_ok=True)

    return parser
