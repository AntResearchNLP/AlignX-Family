from typing import List


from jinja2 import Template
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def process_dialogue(tokenizer, prompt_text):
    """
    Args:
    - dialogue: dialogue[0] is the raw prompt
    """
    system_prompt = """You are an expert User Preference Analyst. Your sole task is to analyze a user's past preference summary (possibly not provided) and new interaction history and summarize the user preferences.\n\nThe user's history will be provided as a sequence of triples, where each triple is `(QUERY, CHOSEN ITEM BY THE USER, REJECTED ITEM BY THE USER)`.\n\nDo not include any introductory phrases, greetings, or any other text outside of this specified structure."""
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": prompt_text.strip(),
        },
    ]

    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return full_prompt


class UPIDataset(RLHFDataset):
    """ """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}
        raw_prompt = process_dialogue(
            self.tokenizer, messages[1]
        )  # select prompts 0 -> former half; 1 -> latter half; 2-> all

        model_inputs = self.tokenizer(
            raw_prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        # breakpoint()

        position_ids = compute_position_id_with_mask(attention_mask)
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        return row_dict


class UPIStreamingDataset(RLHFDataset):
    """ """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}
        raw_prompt = process_dialogue(self.tokenizer, messages[0])
        model_inputs = self.tokenizer(
            raw_prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        return row_dict
