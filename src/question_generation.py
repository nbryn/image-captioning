import argparse
import os
import random
import numpy as np
import torch
import nltk
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup, AutoModelForMaskedLM, DataCollatorWithPadding
from torch.utils.data import DataLoader
from src.early_stopping import EarlyStopper
from src.preprocess import add_genome_images_to_annotations
from src.data_loader import QuestionGenerationDataLoader, QG_Tokenized_Dataset
from src.util import bcolors, error, get_bleu_score, get_meteor_score, success, log, print_prediction
from src.config import MODE
import matplotlib.pyplot as plt
from src.sacrebleu import corpus_bleu


class QuestionGeneration:
    """
    Fine tuning of BERT model for question generation
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.seed(self.args.seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)

        # -----Set model-----
        self.model = AutoModelForMaskedLM.from_pretrained(self.args.model)
        self.model.to(self.device)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)

        self.model = torch.nn.DataParallel(self.model)

        # Helpers
        self.data_loader = QuestionGenerationDataLoader(
            self.tokenizer, self.device, args)

        # Download nltk data if not already downloaded - used for Meteor
        nltk.download('wordnet', force=False)

    def eval_prediction(self, prediction_sentence_list: list):
        """
        Evaluate the predictions of the model using the average bleu-4 score and the average meteor score.
        """
        blue_scores = []
        meteor_scores = []
        for payload in prediction_sentence_list:
            _, prediction_sentence, question, _, _, _ = payload

            # ----EVAL METRICS----
            """
            Bleu score.
            Default nltk bleu score with 4-grams.
            """

            blue_scores.append(get_bleu_score(question, prediction_sentence))

            """
            Meteor score.
            """
            meteor_scores.append(get_meteor_score(
                question, prediction_sentence))

        # catch if value is 0
        avg_blue_score = np.nan_to_num(sum(blue_scores) / len(blue_scores))
        avg_blue_score = sum(blue_scores) / len(blue_scores)
        avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
        delta_bleu = self.calc_delta_bleu(prediction_sentence_list)

        return round(avg_blue_score, 4), round(avg_meteor_score, 4), round(delta_bleu, 4)

    def test(self):
        # -----Preprocess test data-----
        context_list = self.data_loader.get_test_data()

        # -----load model weights-----
        self.model.load_state_dict(torch.load(
            self.get_model_out_name(self.args.num_train_epochs-1), map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

        # -----Test-----
        prediction_sentence_list = []
        for payload in tqdm(context_list, desc="***Testing...:"):
            context, question, image_1, image_2, org_questions = payload

            context_tokenized = self.tokenizer.encode(
                context, add_special_tokens=False)
            prediction_list = []
            for _ in range(self.args.max_question_token_len):
                # Masked token must be added after prediction_list
                # Here we try to predict the next token. For example: a man is playing [MASK] and convert the predicted token to id.
                pred_str_ids = self.tokenizer.convert_tokens_to_ids(
                    prediction_list + [self.tokenizer.mask_token])
                predict_token = context_tokenized + pred_str_ids

                # -----discard if the length of the token exceeds the max_len of BERT-----
                if len(predict_token) >= self.args.max_len:
                    break

                predict_token = torch.tensor([predict_token])
                predictions = self.model(predict_token)

                predicted_index = torch.argmax(predictions[0][0][-1]).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens(
                    [predicted_index])

                # -----discard if the predicted token is [SEP] or [CLS]-----
                if self.tokenizer.sep_token in predicted_token or self.tokenizer.cls_token in predicted_token:
                    break

                prediction_list.append(predicted_token[0])

            # -----decode tokens-----
            token_ids = self.tokenizer.convert_tokens_to_ids(prediction_list)
            prediction_sentence = self.tokenizer.decode(token_ids)

            if self.args.mode != MODE.PRESENT:
                print_prediction(prediction_sentence, question, context)

            prediction_sentence_list.append(
                (context, prediction_sentence, question, image_1, image_2, org_questions))

        # -----Create folders-----
        os.makedirs(os.path.dirname(self.args.save_txt_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.args.metrics_path), exist_ok=True)

        # -----Save predictions-----
        if self.args.mode != MODE.PRESENT:
            log("Saving predictions to " + self.args.save_txt_path)
            with open(self.args.save_txt_path, "w", encoding="UTF-8") as f:
                for payload in tqdm(prediction_sentence_list, desc="***Saving predictions...: "):
                    context, prediction_sentence, question, image_1, image_2, org_questions = payload
                    msg = self.get_test_msg(
                        question, prediction_sentence, context, image_1, image_2)
                    log(msg)
                    f.write(f"{msg}\n")

            with open(self.args.metrics_path, "w", encoding="UTF-8") as f:
                avg_blue_score, avg_meteor_score, delta_bleu = self.eval_prediction(
                    prediction_sentence_list)
                f.write(f"Average Bleu score: {avg_blue_score}\n")
                f.write(f"Average Meteor score: {avg_meteor_score}\n")
                f.write(f"Delta Bleu score: {delta_bleu}\n")

        success("Training finished!")

        return prediction_sentence_list

    def get_test_msg(self, ground_truth_question, pred, context, image_id1, image_id2):
        return f"Predicted: {pred} :: Ground truth: {ground_truth_question} :: Image id 1: {image_id1} :: Image id 2: {image_id2} :: Context: {context}"

    def train(self):
        # -----Preprocess training data-----
        t_total, train_data_loader = self.prepare_train_data()
        log(f"***t_total: {t_total}, len train_dataloader: {len(train_data_loader)}")

        # -----Set optimizer-----
        optimizer_grouped_parameters = [
            {
                "params": [p for _, p in self.model.named_parameters()],
                "weight_decay": 0.0,
            }]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0.0, num_training_steps=t_total)

        # Make dir if not exists
        os.makedirs(self.args.model_weight_folder_path, exist_ok=True)

        # -----Train-----
        self.model.train()
        losses = []
        for epoch in range(self.args.num_train_epochs):
            early_stop = EarlyStopper(
                patience=self.args.patience, min_delta=self.args.min_delta)
            eveloss = 0  # which is the loss of the current epoch
            train_loader = tqdm(train_data_loader,
                                desc="***Train epoch: " + str(epoch))

            # -----Iterate over batches-----
            for idx, batch in enumerate(train_loader):
                # -----Set batch-----
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # -----Forward-----
                # Labels are embedded in batches
                outputs = self.model(**batch)  # k, labels=v

                # -----Backward-----
                loss = outputs.loss
                losses.append(loss.mean().item())
                eveloss += loss.mean().item()

                loss.mean().backward()

                # -----Update parameters-----
                optimizer.step()
                optimizer.zero_grad()  # Reset gradients

                train_loader.set_description(
                    "Loss %.04f | step %d" % (loss.mean(), idx))

                # -----Early stopping-----
                if early_stop.early_stop(loss.mean().item()):
                    log(
                        f"Early stopping at iteration: {idx} - epoch: {epoch}", color=bcolors.WARNING)
                    break

            # Dynamic learning rate
            scheduler.step()

            torch.save(self.model.state_dict(),
                       self.get_model_out_name(epoch))

            self.save_loss_plot(losses)

            log("epoch " + str(epoch) + " : " + str(eveloss))

        success("Training finished!")

    def prepare_train_data(self):
        model_input, labels = self.data_loader.get_train_data()
        pad_len = len(model_input["input_ids"][0])

        # Padding and make labels
        train_label_list = self.data_loader.make_labels(pad_len, labels)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        train_dataloader = DataLoader(QG_Tokenized_Dataset(
            model_input, train_label_list), shuffle=True, batch_size=self.args.batch_size, collate_fn=data_collator)

        t_total = len(model_input) // self.args.num_train_epochs

        # Check dataloader
        try:
            for _, batch in enumerate(train_dataloader):
                break
            {k: v.shape for k, v in batch.items()}
            success("*** dataloader works successfully!")

        except Exception as e:
            error("Something went wrong: " + str(e))
            assert False

        return t_total, train_dataloader

    def seed(self, seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def get_model_out_name(self, epoch: int):
        return os.path.join(self.args.model_weight_folder_path, f"model_weights_{self.args.timestamp}_{epoch}.pth")

    def save_loss_plot(self, losses: list[float]):
        plt.clf()
        plt.rcdefaults()
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Iterations or Batches')
        plt.ylabel('Loss')
        plt.title('Training Loss over Iterations or Batches')
        plt.legend()
        plt.savefig(self.args.loss_plot_path, dpi=450)

    def calc_delta_bleu(self, pred_list: list):
        """
        https://zhu45.org/posts/2018/Mar/28/bleu-a-method-for-automatic-evaluation-of-machine-translation/
        """
        org_annotations = add_genome_images_to_annotations().values()

        label2score = {
            2: 1.0,  # strong positive
            1: 0.5,  # weak positive
            -1: -0.5,  # negative
        }
        # corpus_bleu
        bleus_total = []
        for payload in pred_list:
            _, prediction_sentence, question, image_1, image_2, _ = payload
            org_annotation = list(
                filter(lambda x: x["images"][0]["VG_image_id"] == image_1 and x["images"][1]["VG_image_id"] == image_2, org_annotations))[0]

            question_list = org_annotation["org_questions"]

            questions = list(map(lambda x: [x[0]], question_list))
            score_coff = list(
                map(lambda x: [label2score[x[1]]], question_list))

            d_bleu = corpus_bleu([prediction_sentence],
                                 questions, ref_weights=score_coff)

            bleus_total.append(d_bleu.score)

        return np.mean(bleus_total)

    def limit(self, val: float):
        return min(max(val, 0.0), 1.0)
