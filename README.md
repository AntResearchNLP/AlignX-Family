<div align="center">

<img src="figures/v1.png" width="100%" alt="AlignX-Family: Reasoning Meets Empathy">

<!--
<h1 align="center" style="font-size: 40px; font-weight: 800; letter-spacing: -0.5px;">
AlignX-Family: <br>
Personalizing Large Language Models for <em>Human-Like</em> SuperIntelligence
</h1>
-->
# AlignX-Family: Personalizing Large Language Models for <em>Human-Like</em> SuperIntelligence








<!--
<p align="center" style="font-size: 17px; color: #555;">
Personalizing Large Language Models for <em>Human-Like</em> SuperIntelligence
</p>
-->


<p align="center" style="font-size: 17px; color: #555;">
<a href="./AlignX/README.md">AlignX</a> &nbsp;•&nbsp; <a href="./AlignXplore/README.md">AlignXplore</a> &nbsp;•&nbsp; <a href="./AlignXplorePlus/README.md">AlignXplore+</a>
</p>

</div>



## :sparkles: Introduction
<strong>AlignX-Family</strong> is a research initiative dedicated to advancing the personalization of large language models (LLMs). Launched in late 2024, the project is driven by a central conviction: <strong>personalization is a cornerstone for making the superintelligence exhibited by state-of-the-art LLMs broadly accessible and practically useful to diverse users.</strong> At the time of its inception, however, the community faced a critical gap—there were neither high-quality open-source datasets nor a systematic research framework to support rigorous and reproducible studies of LLM personalization.

AlignX-Family was established to address this gap. The initiative originated from an early research effort codenamed <strong><a href="./AlignX/README.md">AlignX</a></strong>, which focuses on constructing a large-scale dataset for personalized alignment. As the foundational work of the project, AlignX not only provides essential data infrastructure for subsequent explorations, but also lends its name to the broader research family. Building upon this foundation, we further investigated how user preferences can be dynamically reasoned as natural-language summaries from complex and often noisy behavioral signals. These efforts led to the development of <strong><a href="./AlignXplore">AlignXplore</a></strong> and its extension, <strong><a href="./AlignXplorePlus">AlignXplore+</a></strong>, which progressively elevates personalization from data construction, to preference reasoning, and ultimately to a universal interface for downstream applications.


In parallel, the broader landscape of LLM development has rapidly shifted toward personalization. Leading industry models are increasingly incorporating mechanisms such as long-term memory, interactive preference elicitation, and explicit style control. While these advances deliver impressive user experiences, the <strong>core technologies underpinning personalized intelligence largely remain closed and inaccessible to the research community</strong>, limiting transparency, reproducibility, and collective progress.


Through full open-sourcing, AlignX-Family aims to bring personalization to the forefront of LLM research. The project aspires to serve as a foundational resource and shared research platform for the community—enabling the development of superintelligent systems that are not only powerful, but also deeply aligned with individual users.


<!--
To date, AlignX-Family comprises three research works:
- <strong><a href="#package-alignx">AlignX</a></strong> contributes the largest open-source dataset for personalized alignment.
- <strong><a href="#robot-alignxplore">AlignXplore</a></strong> introduces a deep-reasoning approach to infer user preferences from complex behavioral signals, with its key technical innovation being the ability to dynamically update preferences from streaming user interactions.
- <strong><a href="#robot-zap-alignxplore">AlignXplore+</a></strong> further unlocks the potential of AlignXplore — learning preference inference models from heterogeneous signals and extending their impact far beyond response selection to power a broad range of personalized applications, from recommendation to response generation.
-->


## :link: Project Lineage
Our research follows a roadmap of <strong>Data → Methodology → Scaling</strong>, progressively advancing the study of personalization in large language models.

To date, the project comprises three tightly connected sub-projects:

### :package: <strong> <a href="https://github.com/JinaLeejnl/AlignX">AlignX</a> — Data </strong>

AlignX establishes the data foundation of the family by open-sourcing <strong> over 1.3 million examples </strong> to the research community for personalized alignment. It provides large-scale training resources and standardized evaluation testbeds, laying the groundwork for subsequent developments in AlignXplore and AlignXplore+.

### :robot: <strong> <a href="https://github.com/AntResearchNLP/AlignXplore">AlignXplore</a> — Methodology </strong>

AlignXplore introduces a deep-reasoning framework for inferring user preferences from complex behavioral signals. Its key technical innovation lies in enabling <strong> dynamic preference updates from streaming user interactions </strong>, paving the way toward representing user preferences as structured natural-language summaries.

### :zap: <strong> <a href="https://github.com/AntResearchNLP/AlignXplorePlus">AlignXplore+</a> — Scaling </strong>

Building upon AlignXplore, AlignXplore+ explores how preference reasoning can be scaled <strong> across heterogeneous data sources </strong> — including social media, e-commerce platforms, and news streams — and <strong> generalized across tasks and models </strong>, demonstrating the viability of natural language as a universal interface for transferable personalization.

<p align="center">
  <img src="figures/RoadMap2.png" width="100%" alt="RoadMap"/>
</p>
<p align="center">
  <em>Figure: Roadmap of Our Research on LLM Personalization.</em>
</p>



<!--
further unlocks the potential of AlignXplore — learning preference inference models from heterogeneous signals and extending their impact far beyond response selection to power a broad range of personalized applications, from recommendation to response generation.
-->

## :bulb: New to LLM Personalization?

Our paper [A Survey on Personalized Alignment: The Missing Piece for Large Language Models in Real-World Applications](https://aclanthology.org/2025.findings-acl.277.pdf) provides a clear roadmap of the field, helping you quickly understand what the community has explored and where it is heading.

:warning: This survey is a snapshot of the field as of early February 2025. Subsequent advances are not included. We are actively tracking new work and preparing an updated version.


## :rocket: Getting Started

| Your Goal | Start With |
|-----------|------------|
| Data & benchmarks | <a href="https://github.com/JinaLeejnl/AlignX">AlignX</a>|
| Preference reasoning |<a href="https://github.com/AntResearchNLP/AlignXplore">AlignXplore</a> |
| Transferable personalization | <a href="https://github.com/AntResearchNLP/AlignXplorePlus">AlignXplore+</a>|

### :pushpin: If you are interested in data and benchmarks
If your goal is to:
- train or evaluate personalized alignment models,
- build baselines for personalization research,
- or explore large-scale preference data,

➡️ **Start here:** <a href=./AlignX">AlignX</a>

> **AlignX** provides over **1.3M open-sourced examples**, along with standardized training and evaluation resources for personalized alignment research.

### :pushpin: If you are interested in preference reasoning and modeling

If your focus is to:
- infer user preferences from complex or noisy behavioral signals,
- study reasoning-based user modeling,
- or explore dynamic preference updates from streaming interactions,

➡️ **Start here:** <a href="./AlignXplore">AlignXplore</a>

> **AlignXplore** introduces a deep-reasoning framework for transforming noisy and redundant behavioral signals into informative and compact natural language preference summaries.

### :pushpin: If you need to reason over heterogeneous signals and generalize preferences across tasks

If you aim to:
- infer user preferences from heterogeneous behavioral signals,
- reuse inferred preferences across tasks,
- or ensure that the inferred preferences can transfer across models,

➡️ **Start here:** <a href="./AlignXplorePlus">AlignXplore+</a>

> **AlignXplore+** elevates preference reasoning to a universal, natural language based interface that enables cross-task and cross-model personalization.

## 📄 Citations
AlignX-Family is a collection of closely related research projects that jointly advance the personalization of large language models.  If you find the projects helpful, please consider citing the following papers accordingly:

- **AlignX — dataset for personalized alignment**

```bibtex
@misc{li20251000000usersuserscaling,
      title={From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment}, 
      author={Jia-Nan Li and Jian Guan and Songhao Wu and Wei Wu and Rui Yan},
      year={2025},
      eprint={2503.15463},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.15463}, 
}
```

- **AlignXplore — framework for user reference reasoning**

 ```bibtex
@misc{li2025extendedinductivereasoningpersonalized,
      title={Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals}, 
      author={Jia-Nan Li and Jian Guan and Wei Wu and Rui Yan},
      year={2025},
      eprint={2505.18071},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.18071}, 
}
```

- **AlignXplore+ — framework for user preference reasoning, with special focus on heterogeneous inputs and preference transferability**

```bibtex
@misc{liu2026textuniversalinterfacetransferable,
      title={Text as a Universal Interface for Transferable Personalization}, 
      author={Yuting Liu and Jian Guan and Jia-Nan Li and Wei Wu and Jiang-Ming Yang and Jianzhe Zhao and Guibing Guo},
      year={2026},
      eprint={2601.04963},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.04963}, 
}
```

<!--
# :package: AlignX

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2503.15463)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Full%20Data-yellow)](https://huggingface.co/datasets/JinaLeejnl/AlignX)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Test%20Data-yellow)](https://huggingface.co/datasets/JinaLeejnl/AlignX-test)




AlignX releases the <strong>LARGEST</strong> open dataset for personalization research in the era of large language models, featuring <strong>1,311,622</strong> carefully curated examples.
- <strong>Theoretically Sound</strong>: a 90-dimensional preference space that unifies foundational psychological theories (including the Big Five Personality Traits, Maslow’s Hierarchy of Needs, and Murray’s System of Needs), cutting-edge research in recommendation systems and LLM alignment, and real-world interest taxonomies distilled from major English and Chinese social media platforms (including X, Facebook, Zhihu, and RedNote).
- <strong>Comprehensive</strong>: beyond response preference pairs, the dataset delivers rich persona signals—including user-generated content (pointwise), comparative feedback (pairwise), and demographic attributes (pointwise).
- <strong>Privacy-Preserving</strong>: all persona signals are fully synthesized using large language models, ensuring that no real private information is collected or exposed.

<p align="center">
  <img src="figures/AlignX_Taxonomy.png" width="80%" alt="AlignX taxonomy"/>
</p>
<p align="center">
  <em>Figure: The 90-dimensional taxonomy used to construct the preference space in AlignX.</em>
</p>

## Data Format

```jsonc
{
    "prompt": "", // the post eliciting responses
    "chosen": "", // the user-preferred response
    "rejected": "", // the less preferred response relative to "chosen"
    "Preference Direction": [0/0.5/1] * 90, // a 90-element list: 1 = "Positive" (higher levels preferred), 0 = "Negative" (lower levels preferred), 0.5 = "Neutral" (no clear preference)
    "Demographic Information": "", // a comprehensive natural language description of the user
    "User-Generated Content": [ // comments written by the same user on other posts
        { // UGC 1
            "prompt": "",
            "comment": "",
            "Preference Direction": [0/0.5/1] * 90
        },
        { // UGC 2
            ...
        },
        { // UGC 3
            ...
        },
        { // UGC 4
            ...
        }
    ],
    "Pair-wise Comparative Feedback": [ // the preference pairs of the same user for comments under other posts
        { // PAIR 1
            "prompt": "",
            "chosen": "",
            "rejected": "",
            "Preference Direction": [0/0.5/1] * 90
        },
        { // PAIR 2
            ...
        },
        { // PAIR 3
            ...
        },
        { // PAIR 4
            ...
        }
    ]
}
```

## Data Sources

| **Source** | **Reddit** | **PKU-SafeRLHF** | **UltraFeedback** | **HelpSteer2** |
|------------|------------|------------------|-------------------|----------------|
| **Dimension** | 90 preference dimensions | Safety | Helpfulness / Honesty / Instruction-Following / Truthfulness | Helpfulness / Correctness / Coherence / Complexity / Verbosity |
| **#Examples** | 1,225,988 | 10,714 | 11,629 / 16,809 / 36,169 / 7,219 | 2,255 / 144 / 26 / 33 / 636 |

## Citation
```
@misc{li20251000000usersuserscaling,
      title={From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment}, 
      author={Jia-Nan Li and Jian Guan and Songhao Wu and Wei Wu and Rui Yan},
      year={2025},
      eprint={2503.15463},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.15463}, 
}
```
-->

<!--
## Dataset Overview

<img src="figures/dataset_overview.png" width="1200px">

# AlignXpert
We implement In-Context Alignment (ICA) and Preference-Bridged Alignment (PBA) based on Llama-3.1-8B-Instruct. We train the model using the 7% subset (91,918 samples) and the full dataset (1,311,622 samples), respectively. The experimental results are shown in the table below, where our model significantly outperforms the baselines.

<img src="figures/result.png" width="1200px">



## Links

- 📜 [Paper](https://arxiv.org/abs/2503.15463)
- 🤗 [AlignXpert<sub>ICA</sub> (Training with a 7% Subset)](https://huggingface.co/JinaLeejnl/AlignXpert-ICA-Subset)
- 🤗 [AlignXpert<sub>PBA</sub> (Training with a 7% Subset)](https://huggingface.co/JinaLeejnl/AlignXpert-PBA-Subset)
- 🤗 [AlignXpert<sub>ICA</sub> (Training with the Full Dataset)](https://huggingface.co/JinaLeejnl/AlignXpert-ICA-Full)
- 🤗 [AlignXpert<sub>PBA</sub> (Training with the Full Dataset)](https://huggingface.co/JinaLeejnl/AlignXpert-PBA-Full)

## Training

The code is developed based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).

Construct training data:

```train
cd AlignXpert/train
python format_data.py
```

### In-context alignment (ICA)

```train
cd AlignXpert/train/OpenRLHF/examples/scripts
./ica_dpo.sh
```

### Preference-bridged alignment (PBA)

```train
cd AlignXpert/train/OpenRLHF/examples/scripts
./pba_dpo.sh
```

## Evaluation

### Alignment Accuracy

`./AlignXpert/eval/loss_ica.py` and `./AlignXpert/eval/loss_pba.py` are used to calculate the log probability of chosen and rejected responses with AlignXpert<sub>ICA</sub> and AlignXpert<sub>PBA</sub> as the policy models, respectively. `./AlignXpert/eval/loss_few_shot.py` calculates the log probability of chosen and rejected responses for the reference model. After obtaining the log probabilities for both the policy and reference models, `./AlignXpert/eval/acc.py` is used to compute the Alignment Accuracy.

### GPT-4 Win Rate

Responses generated by `./AlignXpert/eval/gen_ica.py`, `./AlignXpert/eval/gen_pba.py`, and `./AlignXpert/eval/gen_few_shot.py` are evaluated using GPT-4.

-->

<!--
# :robot: AlignXplore

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2505.18071)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Data-yellow)](https://huggingface.co/datasets/JinaLeejnl/AlignXplore)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/JinaLeejnl/AlignXplore-7B-Streaming)

<strong>AlignXplore</strong> is a deep user understanding framework designed to infer rich and human-readable preference summaries from complex—even noisy—behavioral signals.
- <strong>Streaming</strong>: Built for streaming scenarios. As new signals arrive, AlignXplore incrementally updates existing preference summaries through lightweight reasoning—eliminating the need to rerun costly end-to-end inference from scratch.
- <strong>Flexible</strong>: Supports heterogeneous input formats out of the box, including preference pairs (e.g., post–chosen–rejected comparisons) and free-form text signals such as user-generated content.
- <strong>Robust</strong>: Delivers stable and reliable performance under noisy inputs and remains resilient to abrupt or long-term shifts in user preferences.

## Building Your Own User Model with AlignXplore
### Requirements

To install requirements:

```setup
cd AlignXplore
pip install -r requirements.txt
```

### Training

#### Cold-start training

```train
cd AlignXplore/cold-start training
./sft.sh
```

#### Reinforcement learning

The code is developed based on [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero).

##### Train with $R_{jud}$

```train
cd AlignXplore/reinforcement learning
./run_ppo_jud.sh
```

##### Train with $R_{gen}$

Modify the file `/AlignXplore/reinforcement learning/orz/ppo/actors.py`:
- Change line [1027](https://github.com/AntResearchNLP/AlignX-Family/blob/9c83d8bd64d7170d6bbc1b098caba7fa4d70bade/AlignXplore/reinforcement%20learning/orz/ppo/actors.py#L1027) to `RewardRayActor = ray.remote(num_gpus=1)(genRewardRayActorBase)`.

```train
cd AlignXplore/reinforcement learning
./run_ppo_gen.sh
```

## Citation
```
@misc{li2025extendedinductivereasoningpersonalized,
      title={Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals}, 
      author={Jia-Nan Li and Jian Guan and Wei Wu and Rui Yan},
      year={2025},
      eprint={2505.18071},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.18071}, 
}
```

# :robot: :zap: AlignXplore+
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2505.18071)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-SFT%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-SFT)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-RL%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-RL)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Eval%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-Benchmark)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/VanillaH1/AlignXplore-Plus)

<strong>AlignXplore+</strong> represents a significant upgrade to AlignXplore, advancing the vision that natural language can function as a universal, model- and task-agnostic interface for representing fine-grained, multi-dimensional user preferences.
- <strong> General-Purpose </strong>: AlignXplore+ operates in more realistic, real-world scenario, demonstrating that high-quality user preference summaries can be inferred from heterogeneous sources, including social networks, e-commerce platforms, and news streams.
-->
