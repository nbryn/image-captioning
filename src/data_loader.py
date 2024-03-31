import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from src.util import log
from src.config import TRAIN_APPROACH, PROCESS_PAIR


class QuestionGenerationDataLoader:
    def __init__(self, tokenizer, device, args: argparse.Namespace):
        self.tokenizer = tokenizer
        self.device = device
        self.args = args

        """
        https://datascience.stackexchange.com/questions/66207/what-is-purpose-of-the-cls-token-and-why-is-its-encoding-output-important
        """
        self.cls_token = self.tokenizer.cls_token    # [CLS]
        self.sep_token = self.tokenizer.sep_token    # [SEP]

        # The [MASK] token is a special token used for Masked Language Modelling
        self.mask_token = self.tokenizer.mask_token

    def read_train_data(self, json_path="./train_1.json"):
        with open(json_path, "r") as file:
            data = json.load(file)
        return data[0:self.args.data_size]

    def get_test_data(self):
        """
        Given a context, generate a question
        """
        annotations = self.read_train_data(self.args.test_data_path)

        context_list = []
        start = 0 if self.args.test_skip is None else self.args.test_skip
        end = len(annotations) if self.args.test_amount is None else start + \
            self.args.test_amount

        for annotation in tqdm(annotations[start:end], desc="***preprocessing test data..."):
            context = annotation["context"]

            # Take the three first questions
            q_idx = len(annotation["questions_with_scores"]) if len(
                annotation["questions_with_scores"]) < 3 else 3

            questions = list(
                map(lambda x: x[0], annotation["questions_with_scores"]))[0:q_idx]
            org_questions = annotation.get("org_questions", [])

            image_id1 = annotation["images"][0]["VG_image_id"]
            image_id2 = annotation["images"][1]["VG_image_id"]

            input_text = self.prepare_input_text(context)
            context_list.append(
                (input_text, questions, image_id1, image_id2, org_questions))

        return context_list

    def prepare_input_text(self, context):
        """
        Forges the image captions into one context.
        Prepare input text for BERT by using cls to mark the start of the sequence (used for text classification when BERT has to classify the whole sequence)
        and using sep tokens seperate different segements within the input aswell as the end of the sequence.
        """
        if self.args.process_pair == PROCESS_PAIR.TRUE:
            return f"{self.cls_token} {context[0]} {self.sep_token} and {self.sep_token} {context[1]} {self.sep_token}"
        else:
            return f"{self.cls_token} {context[0]} {self.sep_token}"

    def get_rand_idx(self, list_length: int) -> int:
        return np.random.randint(0, list_length)

    def __prepare_train_data(self):
        """
        Gets the train data for the model depending on the setting chosen.
        """
        annotations = self.read_train_data(self.args.train_data_path)
        match self.args.approach:
            case TRAIN_APPROACH.TRAIN_ALL | TRAIN_APPROACH.TRAIN_BEST:
                train_data = []
                for annotation in annotations:
                    questions = annotation["questions_with_scores"] or []
                    for question, score in questions:

                        if self.args.approach == TRAIN_APPROACH.TRAIN_ALL:
                            train_data.append(
                                (annotation["context"], question))

                        elif self.args.approach == TRAIN_APPROACH.TRAIN_BEST and score >= 2:
                            train_data.append(
                                (annotation["context"], question))

                return train_data

            case TRAIN_APPROACH.TRAIN_RAND:
                return list(map(lambda x: (x["context"],  x["questions_with_scores"][self.get_rand_idx(
                    len(x["questions_with_scores"]))][0]), annotations))

            case _:
                raise ValueError(
                    f"Invalid training approach: {self.args.approach}")

    def sort_by_questions_occurences(self, annotations: list[(str, str)]):
        """
        Sort questions by occurences in ascending order.
        """
        # Group by question
        question_dict = {}
        for context, question in annotations:
            if question not in question_dict:
                question_dict[question] = []
            question_dict[question].append(context)

        # sort by question length (shortest first)
        question_dict = {k: v for k, v in sorted(
            question_dict.items(), key=lambda item: len(item[1]))}

        # Convert to list of tuples
        question_list = []
        for question, contexts in question_dict.items():
            for context in contexts:
                question_list.append((context, question))

        return question_list

    def get_train_data(self):
        annotations = self.sort_by_questions_occurences(
            self.__prepare_train_data())
        
        example_list = []
        for context, question in tqdm(annotations, desc="***preprocessing train data..."):
            input_text = input_text = self.prepare_input_text(context)
            tokenized_question = self.tokenizer.tokenize(
                question)

            tokenized_text = self.tokenizer.tokenize(
                input_text, add_special_tokens=False)

            # over bert_base size
            if (len(tokenized_question + tokenized_text) + 2) >= 512:
                log("over bert_base size")
                continue

            """
            We take the tokinzed question and for each char in the question we mask it and is used to predict the next char. 
            See table 1 in A Recurrent BERT-based Model for Question Generation
            """
            for i in range(0, len(tokenized_question) + 1):
                tokenized_text.extend(tokenized_question[:i])
                tokenized_text.append(self.mask_token)
                token_ids = self.tokenizer.convert_tokens_to_ids(
                    tokenized_text)
                label_ids = token_ids.copy()[:-1]

                if i == len(tokenized_question):
                    label_ids.append(self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(self.sep_token))[0])
                else:
                    label_ids.append(
                        self.tokenizer.convert_tokens_to_ids(tokenized_question[i]))

                loss_tensors = torch.tensor([label_ids]).to(self.device)
                input_ids = self.tokenizer.convert_tokens_to_ids(
                    tokenized_text)
                decodes_ids = self.tokenizer.decode(input_ids)
                example_pair = dict()
                example_pair[decodes_ids] = loss_tensors

                example_list.append(example_pair)

        model_input, labels = self._decompose_dataset(example_list)
        return self._tokenized_dataset(model_input), labels

    def _decompose_dataset(self, example_list):
        sentences = []
        labels = []

        for examples in tqdm(example_list, desc="***decompose dataset: "):
            sentences.extend(list(examples.keys()))
            labels.extend(list(examples.values()))

        return sentences, labels

    def _tokenized_dataset(self, data):
        tokenized_sentence = self.tokenizer(
            data,
            padding=True,  # Pad if the sentence is short
            truncation=True,  # Truncate if the sentence is long
            max_length=512,
            return_token_type_ids=True,
            return_tensors="pt",  # return Tensor
            add_special_tokens=False,
        )

        return tokenized_sentence

    def make_labels(self, pad_len, labels):
        labels_list = []
        for label in tqdm(labels, desc="***make labels: "):
            target = torch.zeros(pad_len)
            try:
                target[: len(label[0])] = label[0]
            except:
                target = label[0][:-1][:pad_len]
            labels_list.append(target.tolist())
        labels_list = torch.tensor(labels_list).int()

        return labels_list


class QG_Tokenized_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.tokenized_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
