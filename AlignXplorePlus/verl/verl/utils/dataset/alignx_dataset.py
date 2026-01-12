from typing import List


from jinja2 import Template
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


class AlignXploreDataset(RLHFDataset):
    """
    Load and preprocess AlignXplore data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

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
        raw_prompt = self._process_dialogue(messages)
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

    def _process_dialogue(self, dialogue: List):
        raise NotImplementedError("Please define the dialogue processing.")


class AlignJugDataset(AlignXploreDataset):
    """
    Load and preprocess AlignXplore data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def _process_dialogue(self, dialogue: List):
        """
        Args:
        - dialogue: dialogue[0] is the raw prompt
        """
        prompt_template_jinja = """\
            {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
            Assistant: <think>\
        """

        prompt_instruction_template_jinja = """\
            You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.
            This is the problem:
            {{prompt}}
        """

        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(prompt=dialogue[0])
        prompt_template = Template(prompt_template_jinja)
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
        prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)
        return prompt


class AlignBPRDataset(AlignXploreDataset):
    """
    Load and preprocess AlignXplore data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def _process_dialogue(self, dialogue: List):
        """
        Args:
        - dialogue: dialogue[0] is the raw prompt
        """
        prompt_template_jinja = """\
            {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and the answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
            Assistant: <think>\
            """

        prompt_instruction_template_jinja = """\
            This is the problem:
            {{prompt}}
            Your final answer must be a Python list of strings, enclosed within <answer> </answer> tags, i.e., <answer> answer here </answer>.\
        """

        constraint = """\
            The generated answer should match the following rules:
            1. Your final answer must be a Python list of strings, enclosed within <answer> </answer> tags. Each string in this list should represent a distinct, inferred user preference dimension.
            2. These preference dimensions should adhere to the following guidelines:
                2.1 Capture consistent themes, topics, or interests across all provided posts;
                2.2 Use precise, non-redundant phrasing;
                2.3 Group related concepts under broader categories while avoiding overlap;
                2.4 Prioritize core themes over surface-level details. \
        """

        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(
            prompt=dialogue[0], # constraint=constraint
        )
        prompt_template = Template(prompt_template_jinja)
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
        prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)
        return prompt
