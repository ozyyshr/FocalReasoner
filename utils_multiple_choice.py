# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """


import csv
import glob
import json
from textblob import TextBlob
import logging
from svo_extraction import extractSVOs
import os
from typing import List
import random
import tqdm
import spacy
from transformers import PreTrainedTokenizer
nlp = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "doc_len": doc_len, "query_len": query_len, "opt_len": opt_len, "svo_ids": svo_ids}
            for input_ids, input_mask, segment_ids, doc_len, query_len, opt_len, svo_ids in choices_features
        ]
        # self.choices_features = [
        #     {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "svo_ids": svo_ids}
        #     for input_ids, input_mask, segment_ids, svo_ids in choices_features
        # ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")

        examples = [
            InputExample(
                example_id=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11],
            )
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples


class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[
                            options[0]["para"].replace("_", ""),
                            options[1]["para"].replace("_", ""),
                            options[2]["para"].replace("_", ""),
                            options[3]["para"].replace("_", ""),
                        ],
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth,
                    )
                )

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples

class LogiQAProcessor(DataProcessor):
    """ processor for the logiQA dataset. """
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Eval.txt")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_txt(self, input_dir):
        with open(input_dir, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            assert len(lines) % 8 == 0
        return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(0, len(lines), 8):
            example_lines = lines[i:i+8]
            context = example_lines[2].strip()
            question = example_lines[3].strip()
            answers = [item.strip() for item in example_lines[4:8]]
            label = 0 if type == "test" else ord(example_lines[1].strip())-ord("a") # for test set, there is no label. Just use 0 for convenience.
            id_string = type + str(i)
            examples.append(
                InputExample(
                    example_id = id_string,
                    question = question,
                    contexts=[context, context, context, context],  # this is not efficient but convenient
                    endings=[answers[0], answers[1], answers[2], answers[3]],
                    label = label
                    )
                )  
        return examples


class ReclorProcessor(DataProcessor):
    """Processor for the ReClor data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "val.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers']
            label = 0 if type == "test" else d['label'] # for test set, there is no label. Just use 0 for convenience.
            id_string = d['id_string']
            examples.append(
                InputExample(
                    example_id = id_string,
                    question = question,
                    contexts=[context, context, context, context],  # this is not efficient but convenient
                    endings=[answers[0], answers[1], answers[2], answers[3]],
                    label = label
                    )
                )  
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    max_doc_len: int,
    max_query_len: int,
    max_option_len: int,
    max_svo_len: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    # assert max_length == max_doc_len + max_option_len + max_query_len + 5
    max_seq_len = max_length

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        
        if example.question.find("_") != -1:
            # this is for cloze question
            question = example.question.strip("_")
        else:
            question = example.question
        query_token = tokenizer.tokenize(question)
        # question annotation
        # blob = TextBlob(example.question)
        # if blob.sentences[0].sentiment.polarity < 0:
        #     query_token.insert(0, "[neg]")
        # else:
        #     if ("weaken" in example.question) or ("vulnerable" in example.question) or ("doubt" in example.question) or ("refute" in example.question):
        #         query_token.insert(0, "[neg]")
        #     else:
        #         query_token.insert(0, "[pos]")
        query_len = len(query_token)
        
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            tokens = []
            segment_ids = []
            input_mask = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            input_mask.append(1)

            text_a = context
            doc_token = tokenizer.tokenize(text_a)
            # doc_len = min(max_doc_len, len(doc_token))
            doc_len = len(doc_token)

            a_svos = extractSVOs(nlp, doc_token, offset=1)
            # a_svos = extractSVOs(nlp, doc_token[:doc_len], offset=1)
            
            # tokens = tokens + doc_token[:doc_len] + ['[PAD]']*(max_doc_len-doc_len) + ['[SEP]']
            tokens = tokens + doc_token[:doc_len] + ['[SEP]']
            # segment_ids = segment_ids + [0] * (max_doc_len+1)
            segment_ids = segment_ids + [0] * (doc_len + 1)
            # input_mask = input_mask + [1] * (doc_len +1) + [0] * (max_doc_len-doc_len)
            input_mask = input_mask + [1] * (doc_len + 1)
            
            assert len(tokens) == len(segment_ids) == len(input_mask)

            option_token = tokenizer.tokenize(ending)
            option_len = len(option_token)

            # real_query_len = min(query_len, max_query_len)
            # real_opt_len = min(option_len, max_option_len)
            real_query_len = query_len
            real_opt_len = option_len

            b_svos = extractSVOs(nlp, option_token[:real_opt_len], offset=doc_len + 3 + real_query_len)
            svo_ids = a_svos + b_svos
            svo_ids = svo_ids[:max_svo_len]

            query_option_token = query_token[: real_query_len] + [" "] + option_token[: real_opt_len] + ["[SEP]"]

            real_query_option_len = len(query_option_token)
            tokens = tokens + query_option_token
            segment_ids = segment_ids + [1] * real_query_option_len
            input_mask = input_mask + [1] * real_query_option_len

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            while len(input_ids) < max_seq_len:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            svo_ids += [[[-100,0], [-100,0], [-100,0]]] * (max_svo_len - len(svo_ids))

            assert len(input_ids) == max_seq_len, "%d v.s. %d" % (len(input_ids), max_seq_len)
            assert len(input_mask) == max_seq_len
            assert len(segment_ids) == max_seq_len

            choices_features.append((input_ids, input_mask, segment_ids, doc_len, real_query_len, real_opt_len, svo_ids))


            # if example.question.find("_") != -1:
            #     # this is for cloze question
            #     text_b = example.question.replace("_", ending)
            # else:
            #     text_b = example.question + " " + ending
    
            # inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length,)
            # if len(tokenizer.encode(text_a) + tokenizer.encode(text_b)) > max_length:
            #     token_type_ids = [0] * (max_length-len(tokenizer.encode(text_b)))
            #     a_svos = extractSVOs(nlp, text_a[:max_length-len(tokenizer.tokenize(text_b))], offset=1)
            #     token_type_ids += [1] * (len(tokenizer.encode(text_b)))
            #     b_svos = extractSVOs(nlp, tokenizer.tokenize(text_b), offset=max_length-len(tokenizer.encode(text_b)) + 1)
            # else:
            #     token_type_ids = [0] * len(tokenizer.encode(text_a))
            #     a_svos = extractSVOs(nlp, tokenizer.tokenize(text_a), offset=1)
            #     token_type_ids += [1] * (len(tokenizer.encode(text_b)))
            #     b_svos = extractSVOs(nlp, tokenizer.tokenize(text_b), offset=len(tokenizer.tokenize(text_a)) + 1)
            
            # svo_ids = a_svos + b_svos
            # svo_ids = svo_ids[:max_svo_len]
            
            # if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            #     logger.info(
            #         "Attention! you are cropping tokens (swag task is ok). "
            #         "If you are training ARC and RACE and you are poping question + options,"
            #         "you need to try to use a bigger max seq length!"
            #     )

            # input_ids = inputs["input_ids"]
            # assert len(token_type_ids) == len(input_ids)

            # # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # # tokens are attended to.
            # attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # # Zero-pad up to the sequence length.
            # padding_length = max_length - len(input_ids)
            # if pad_on_left:
            #     input_ids = ([pad_token] * padding_length) + input_ids
            #     attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            #     token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            # else:
            #     input_ids = input_ids + ([pad_token] * padding_length)
            #     attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            #     token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            # svo_ids += [[[-100,0], [-100,0], [-100,0]]] * (max_svo_len - len(svo_ids))

            # assert len(input_ids) == max_length
            # assert len(attention_mask) == max_length
            # assert len(token_type_ids) == max_length
            # choices_features.append((input_ids, attention_mask, token_type_ids, svo_ids))

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (input_ids, attention_mask, token_type_ids, doc_len, query_len, opt_len, svo_ids) in enumerate(choices_features):
            # for choice_idx, (input_ids, attention_mask, token_type_ids, svo_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
                logger.info("svo_ids: {}".format(" ".join(map(str, svo_ids))))
                logger.info("query_len: {}".format(query_len))
                logger.info("doc_len: {}".format(doc_len))
                logger.info("opt_len: {}".format(opt_len))
                logger.info("label: {}".format(label))

        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

    return features


processors = {"race": RaceProcessor, "swag": SwagProcessor, "arc": ArcProcessor, "reclor": ReclorProcessor, "logiqa": LogiQAProcessor}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race": 4, "swag": 4, "arc": 4, "reclor": 4, "LogiQA": 4}
