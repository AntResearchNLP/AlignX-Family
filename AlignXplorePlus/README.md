

<div align="center">
  <h1 style="font-size: 40px;">AlignXplore+</h1>
  <p>Text as a Universal Interface for Personalizing Large Language Models</p>

  
  [![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2601.04963)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-SFT%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-SFT)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-RL%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-RL)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Eval%20Data-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXplorePlus-Benchmark)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/VanillaH1/AlignXplore-Plus)

</div>

## 📖 Introduction

<strong>AlignXplore+</strong> is a framework for transferable personalization in large language models. Building upon AlignXplore, it represents a substantial evolution of the approach by advancing a central idea: natural language can serve as a universal, model- and task-agnostic interface for expressing fine-grained and multi-dimensional user preferences.

![Model](./figures/model.jpeg)

## ✨ Key Features

- <strong> General-Purpose </strong>: AlignXplore+ operates in a more realistic, real-world scenario, demonstrating that high-quality user preference summaries can be inferred from heterogeneous sources, including social networks, e-commerce platforms, and news streams.
- <strong> Transferable </strong>: Preference summaries inferred by AlignXplore+ demonstrate strong transferability across both tasks (e.g., from response selection to news recommendation) and models (e.g., from GPT-OSS-20B to Qwen2.5-7B-Instruct).
- <strong> Streaming & Robust </strong>: Inherited from AlignXplore, AlignXplore+ supports preference reasoning from streaming inputs and maintains stable performance under noisy or imperfect signals, ensuring reliable personalization in realistic, dynamic settings.

<!-- <div align="center">
<table border="0" cellpadding="0" cellspacing="0">
  <tr>
    <td align="center" valign="top">
      <img src="figures/Model_Transfer.PNG" width="420"><br>
      <small><b>Figure 1.</b> Cross-model transferability of preference summaries</small>
    </td>
   <td align="center" valign="top">
      <img src="figures/Task_Transfer.PNG" width="420"><br>
     <small><b>Figure 2.</b> Cross-domain transfer performance</small>
    </td>
  </tr>
</table>
</div> -->

## 📊 Results

### Main Results

The following is our main results across nine benchmarks. P-Soups are split into three preference dimensions: expertise, informativeness, and style. We compare models in three settings: **direct sequence modeling**, **full-history preference inference**, and **streaming preference inference**. All preference inference models (both full-history and streaming) use the Qwen3-8B-non-thinking as the downstream model. Within each setting, **Bold** and <u>Underline</u> mark the best and second-best results among models at the ~8B scale. <span style="color:gray;">Gray</span> score highlights models that are outperformed by the best-performing ~8B model in the same column, including direct sequence models and larger preference inference models that do not show a performance advantage.

<table style="border-collapse:collapse; width:100%; text-align:center;">
  <thead>
    <tr>
      <th rowspan="3" style="border:1px solid #ddd; padding:4px;">Model</th>
      <th colspan="3" style="border:1px solid #ddd; padding:4px;">In-domain</th>
      <th colspan="6" style="border:1px solid #ddd; padding:4px;">Out-of-domain</th>
      <th rowspan="3" style="border:1px solid #ddd; padding:4px;">Avg.</th>
    </tr>
    <tr>
      <th style="border:1px solid #ddd; padding:4px;">MIND</th>
      <th style="border:1px solid #ddd; padding:4px;">Amazon</th>
      <th style="border:1px solid #ddd; padding:4px;">AlignX</th>
      <th style="border:1px solid #ddd; padding:4px;">MovieLens</th>
      <th style="border:1px solid #ddd; padding:4px;">PersonaMem</th>
      <th style="border:1px solid #ddd; padding:4px;">Info.</th>
      <th style="border:1px solid #ddd; padding:4px;">Style</th>
      <th style="border:1px solid #ddd; padding:4px;">Expertise</th>
      <th style="border:1px solid #ddd; padding:4px;">HiCUPID</th>
    </tr>
    <tr>
      <th style="border:1px solid #ddd; padding:4px;">(Rec.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(Rec.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(R.S.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(Rec.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(R.S.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(R.S.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(R.S.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(R.S.)</th>
      <th style="border:1px solid #ddd; padding:4px;">(R.G.)</th>
    </tr>
  </thead>
  <tbody>
    <!-- Direct Full-history Sequence Models w/o Preference Inference -->
    <tr>
      <td colspan="11" style="border:1px solid #ddd; padding:4px; text-align:center; font-weight:bold; background:#f7f7f7;">
        Direct Full-history Sequence Models w/o Preference Inference
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">Qwen3-8B<sub>non-thinking</sub></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">63.03</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">84.05</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">59.63</span></td>
      <td style="border:1px solid #ddd; padding:4px;">88.57</td>
      <td style="border:1px solid #ddd; padding:4px;">61.40</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">46.84</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">42.33</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">38.33</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">47.02</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">59.02</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">TALLRec</td>
      <td style="border:1px solid #ddd; padding:4px;">81.96</td>
      <td style="border:1px solid #ddd; padding:4px;">94.91</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">66.30</span></td>
      <td style="border:1px solid #ddd; padding:4px;">97.90</td>
      <td style="border:1px solid #ddd; padding:4px;">64.36</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">51.66</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">70.16</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">60.16</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">47.41</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">70.53</span></td>
    </tr>
    <!-- Full-history Preference Inference -->
    <tr>
      <td colspan="11" style="border:1px solid #ddd; padding:4px; text-align:center; font-weight:bold; background:#f7f7f7;">
        Full-history Preference Inference
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">DeepSeek-R1-671B</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">65.53</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">82.15</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">65.90</span></td>
      <td style="border:1px solid #ddd; padding:4px;">82.76</td>
      <td style="border:1px solid #ddd; padding:4px;">61.44</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">72.59</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">85.66</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">82.33</span></td>
      <td style="border:1px solid #ddd; padding:4px;">63.90</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">73.58</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">Qwen3-32B<sub>thinking</sub></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">67.63</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">85.69</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">64.93</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">75.43</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">57.36</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">73.25</span></td>
      <td style="border:1px solid #ddd; padding:4px;">88.00</td>
      <td style="border:1px solid #ddd; padding:4px;">83.66</td>
      <td style="border:1px solid #ddd; padding:4px;">63.44</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">73.26</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">GPT-OSS-20B</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">64.16</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">83.75</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">55.63</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">74.46</span></td>
      <td style="border:1px solid #ddd; padding:4px;">61.74</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">68.77</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">86.00</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">81.66</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">62.00</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">70.90</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">Qwen3-8B<sub>thinking</sub></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>66.10</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>84.68</u></td>
      <td style="border:1px solid #ddd; padding:4px;">62.73</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>75.13</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>54.36</u></td>
      <td style="border:1px solid #ddd; padding:4px;">75.08</td>
      <td style="border:1px solid #ddd; padding:4px;"><b>87.50</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>83.50</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>60.05</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>72.12</u></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">DS-R1-Distill-Qwen-7B</td>
      <td style="border:1px solid #ddd; padding:4px;">61.20</td>
      <td style="border:1px solid #ddd; padding:4px;">82.82</td>
      <td style="border:1px solid #ddd; padding:4px;">54.03</td>
      <td style="border:1px solid #ddd; padding:4px;">70.30</td>
      <td style="border:1px solid #ddd; padding:4px;">49.28</td>
      <td style="border:1px solid #ddd; padding:4px;">56.14</td>
      <td style="border:1px solid #ddd; padding:4px;">65.83</td>
      <td style="border:1px solid #ddd; padding:4px;">66.00</td>
      <td style="border:1px solid #ddd; padding:4px;">60.01</td>
      <td style="border:1px solid #ddd; padding:4px;">62.84</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;"><span style="font-variant:small-caps;">AlignXplore</span></td>
      <td style="border:1px solid #ddd; padding:4px;">61.23</td>
      <td style="border:1px solid #ddd; padding:4px;">78.58</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>66.60</u></td>
      <td style="border:1px solid #ddd; padding:4px;">69.93</td>
      <td style="border:1px solid #ddd; padding:4px;">53.98</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>76.24</u></td>
      <td style="border:1px solid #ddd; padding:4px;">78.00</td>
      <td style="border:1px solid #ddd; padding:4px;">72.66</td>
      <td style="border:1px solid #ddd; padding:4px;">53.50</td>
      <td style="border:1px solid #ddd; padding:4px;">66.07</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;"><span style="font-variant:small-caps;">AlignXplore+</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>71.36</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>86.39</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>75.03</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>75.80</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>58.08</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>78.07</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>86.33</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>82.50</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>62.42</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>75.10</b></td>
    </tr>
    <!-- Streaming Preference Inference -->
    <tr>
      <td colspan="11" style="border:1px solid #ddd; padding:4px; text-align:center; font-weight:bold; background:#f7f7f7;">
        Streaming Preference Inference
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">DeepSeek-R1-671B</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">64.30</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">80.54</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">64.06</span></td>
      <td style="border:1px solid #ddd; padding:4px;">83.63</td>
      <td style="border:1px solid #ddd; padding:4px;">58.96</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">66.61</span></td>
      <td style="border:1px solid #ddd; padding:4px;">85.00</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">79.00</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">60.32</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">72.93</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">Qwen3-32B<sub>thinking</sub></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">66.60</span></td>
      <td style="border:1px solid #ddd; padding:4px;">85.35</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">64.60</span></td>
      <td style="border:1px solid #ddd; padding:4px;">77.78</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">53.26</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">73.58</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">83.66</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">81.67</span></td>
      <td style="border:1px solid #ddd; padding:4px;">59.83</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">71.81</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">GPT-OSS-20B</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">64.93</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">84.55</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">56.86</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">73.66</span></td>
      <td style="border:1px solid #ddd; padding:4px;">54.82</td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">69.93</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">83.00</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">77.50</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">59.93</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><span style="color:gray;">69.46</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">Qwen3-8B<sub>thinking</sub></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>66.13</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>83.58</u></td>
      <td style="border:1px solid #ddd; padding:4px;">62.90</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>75.97</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>51.68</u></td>
      <td style="border:1px solid #ddd; padding:4px;">74.08</td>
      <td style="border:1px solid #ddd; padding:4px;"><b>85.00</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>82.66</b></td>
      <td style="border:1px solid #ddd; padding:4px;">59.17</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>71.24</u></td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;">DS-R1-Distill-Qwen-7B</td>
      <td style="border:1px solid #ddd; padding:4px;">61.16</td>
      <td style="border:1px solid #ddd; padding:4px;">81.78</td>
      <td style="border:1px solid #ddd; padding:4px;">56.40</td>
      <td style="border:1px solid #ddd; padding:4px;">69.63</td>
      <td style="border:1px solid #ddd; padding:4px;">46.64</td>
      <td style="border:1px solid #ddd; padding:4px;">58.63</td>
      <td style="border:1px solid #ddd; padding:4px;">60.83</td>
      <td style="border:1px solid #ddd; padding:4px;">64.16</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>59.29</u></td>
      <td style="border:1px solid #ddd; padding:4px;">62.05</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;"><span style="font-variant:small-caps;">AlignXplore</span></td>
      <td style="border:1px solid #ddd; padding:4px;">60.66</td>
      <td style="border:1px solid #ddd; padding:4px;">79.01</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>69.90</u></td>
      <td style="border:1px solid #ddd; padding:4px;">67.96</td>
      <td style="border:1px solid #ddd; padding:4px;">48.42</td>
      <td style="border:1px solid #ddd; padding:4px;"><u>74.41</u></td>
      <td style="border:1px solid #ddd; padding:4px;">74.83</td>
      <td style="border:1px solid #ddd; padding:4px;">69.16</td>
      <td style="border:1px solid #ddd; padding:4px;">50.34</td>
      <td style="border:1px solid #ddd; padding:4px;">66.07</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:4px; text-align:left;"><span style="font-variant:small-caps;">AlignXplore+</span></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>71.80</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>85.35</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>73.67</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>77.23</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>54.58</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>76.57</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>80.33</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><u>78.50</u></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>60.51</b></td>
      <td style="border:1px solid #ddd; padding:4px;"><b>73.17</b></td>
    </tr>
  </tbody>
</table>


### Cross-model Transfer Results

We evaluate if the generated summaries are useful for a variety of downstream models, not just the one used for training. Concretely, we first use the baselines and our AlignXplore+ model to generate user preference summaries, and then feed these summaries into Qwen2.5-7B-Instruct and GPT-OSS-20B to perform downstream tasks.

<table>
    <caption>Qwen2.5-7B-Instruct</caption>
    <thead>
        <tr>
            <th style="text-align:left">Model</th>
            <th>MIND</th>
            <th>Amazon</th>
            <th>AlignX</th>
            <th>MovieLens</th>
            <th>Info.</th>
            <th>Style</th>
            <th>Expertise</th>
            <th>PersonaMem</th>
            <th>AVG</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan="10" style="text-align:center; font-weight:bold; background-color:#f0f0f0; border-top: 2px solid #000;">Direct Full-history Sequence Models w/o Preference Inference</td>
        </tr>
        <tr>
            <td style="text-align:left">Qwen-2.5-7B-Instruct</td>
            <td>52.60</td>
            <td>67.75</td>
            <td>52.96</td>
            <td>91.30</td>
            <td>50.83</td>
            <td>40.16</td>
            <td>51.33</td>
            <td>40.76</td>
            <td>55.96</td>
        </tr>
        <tr>
            <td colspan="10" style="text-align:center; font-weight:bold; background-color:#f0f0f0; border-top: 2px solid #000;">Full-history Preference Inference</td>
        </tr>
        <tr>
            <td style="text-align:left">DeepSeek-R1-671B</td>
            <td>63.80</td>
            <td>79.93</td>
            <td>64.10</td>
            <td><strong>79.33</strong></td>
            <td>66.44</td>
            <td><strong>76.16</strong></td>
            <td>76.50</td>
            <td>64.96</td>
            <td>71.40</td>
        </tr>
        <tr>
            <td style="text-align:left">Qwen3-32B<sub>thinking</sub></td>
            <td>65.90</td>
            <td>85.85</td>
            <td>62.83</td>
            <td>74.30</td>
            <td>67.60</td>
            <td>69.67</td>
            <td><strong>77.33</strong></td>
            <td>61.36</td>
            <td>70.61</td>
        </tr>
        <tr>
            <td style="text-align:left">GPT-OSS-20B</td>
            <td>61.73</td>
            <td>85.05</td>
            <td>55.86</td>
            <td>75.23</td>
            <td>68.27</td>
            <td>68.33</td>
            <td>75.33</td>
            <td><strong>65.74</strong></td>
            <td>69.44</td>
        </tr>
        <tr>
            <td style="text-align:left">Qwen3-8B<sub>thinking</sub></td>
            <td>63.36</td>
            <td>85.12</td>
            <td>60.46</td>
            <td>74.20</td>
            <td>64.11</td>
            <td>73.83</td>
            <td>72.83</td>
            <td>60.58</td>
            <td>69.31</td>
        </tr>
        <tr>
            <td style="text-align:left">DS-R1-Distill-Qwen-7B</td>
            <td>58.40</td>
            <td>82.35</td>
            <td>53.50</td>
            <td>69.53</td>
            <td>49.66</td>
            <td>51.16</td>
            <td>57.66</td>
            <td>57.08</td>
            <td>59.92</td>
        </tr>
        <tr>
            <td style="text-align:left"><span style="font-variant: small-caps;">AlignXplore</span></td>
            <td>57.53</td>
            <td>81.12</td>
            <td>65.20</td>
            <td>69.83</td>
            <td>67.94</td>
            <td>63.00</td>
            <td>68.16</td>
            <td>53.60</td>
            <td>65.80</td>
        </tr>
        <tr>
            <td style="text-align:left"><span style="font-variant: small-caps;">AlignXplore+</span></td>
            <td><strong>68.13</strong></td>
            <td><strong>86.25</strong></td>
            <td><strong>73.90</strong></td>
            <td>73.96</td>
            <td><strong>68.77</strong></td>
            <td>73.66</td>
            <td>74.33</td>
            <td>60.24</td>
            <td><strong>72.41</strong></td>
        </tr>
        <tr>
            <td colspan="10" style="text-align:center; font-weight:bold; background-color:#f0f0f0; border-top: 2px solid #000;">Streaming Preference Inference</td>
        </tr>
        <tr>
            <td style="text-align:left">DeepSeek-R1-671B</td>
            <td>63.50</td>
            <td>81.08</td>
            <td>64.50</td>
            <td><strong>80.33</strong></td>
            <td>69.10</td>
            <td>75.33</td>
            <td><strong>77.00</strong></td>
            <td><strong>62.16</strong></td>
            <td>71.62</td>
        </tr>
        <tr>
            <td style="text-align:left">Qwen3-32B<sub>thinking</sub></td>
            <td>65.16</td>
            <td><strong>85.35</strong></td>
            <td>63.56</td>
            <td>74.56</td>
            <td>68.27</td>
            <td>70.67</td>
            <td>73.50</td>
            <td>57.72</td>
            <td>69.85</td>
        </tr>
        <tr>
            <td style="text-align:left">GPT-OSS-20B</td>
            <td>63.10</td>
            <td>84.22</td>
            <td>57.70</td>
            <td>72.43</td>
            <td>69.76</td>
            <td>64.16</td>
            <td>73.00</td>
            <td>59.28</td>
            <td>67.96</td>
        </tr>
        <tr>
            <td style="text-align:left">Qwen3-8B<sub>thinking</sub></td>
            <td>64.13</td>
            <td>84.12</td>
            <td>61.10</td>
            <td>72.14</td>
            <td>67.94</td>
            <td><strong>75.83</strong></td>
            <td>76.09</td>
            <td>56.46</td>
            <td>69.73</td>
        </tr>
        <tr>
            <td style="text-align:left">DS-R1-Distill-Qwen-7B</td>
            <td>58.10</td>
            <td>82.55</td>
            <td>55.56</td>
            <td>69.86</td>
            <td>51.99</td>
            <td>48.66</td>
            <td>59.16</td>
            <td>54.22</td>
            <td>60.01</td>
        </tr>
        <tr>
            <td style="text-align:left"><span style="font-variant: small-caps;">AlignXplore</span></td>
            <td>57.96</td>
            <td>80.38</td>
            <td>69.90</td>
            <td>67.20</td>
            <td>65.44</td>
            <td>58.83</td>
            <td>63.83</td>
            <td>49.16</td>
            <td>64.09</td>
        </tr>
        <tr>
            <td style="text-align:left"><span style="font-variant: small-caps;">AlignXplore+</span></td>
            <td><strong>67.73</strong></td>
            <td>85.05</td>
            <td><strong>74.00</strong></td>
            <td>75.56</td>
            <td><strong>71.92</strong></td>
            <td>71.33</td>
            <td>74.00</td>
            <td>55.32</td>
            <td><strong>71.86</strong></td>
        </tr>
    </tbody>
</table>


<table>
    <caption>GPT-OSS-20B</caption>
    <thead>
        <tr>
            <th style="text-align:left">Model</th>
            <th>MIND</th>
            <th>Amazon</th>
            <th>AlignX</th>
            <th>MovieLens</th>
            <th>Info.</th>
            <th>Style</th>
            <th>Expertise</th>
            <th>PersonaMem</th>
            <th>AVG</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan="10" style="text-align:center; font-weight:bold; background-color:#f0f0f0; border-top: 2px solid #000;">Direct Full-history Sequence Models w/o Preference Inference</td>
        </tr>
        <tr>
            <td style="text-align:left">GPT-OSS-20B</td>
            <td>63.86</td>
            <td>88.99</td>
            <td>68.50</td>
            <td>85.26</td>
            <td>79.40</td>
            <td>83.66</td>
            <td>84.50</td>
            <td>21.92</td>
            <td>72.01</td>
        </tr>
        <tr>
            <td colspan="10" style="text-align:center; font-weight:bold; background-color:#f0f0f0; border-top: 2px solid #000;">Full-history Preference Inference</td>
        </tr>
        <tr>
            <td style="text-align:left">DeepSeek-R1-671B</td>
            <td>60.01</td>
            <td>80.00</td>
            <td>68.60</td>
            <td>77.33</td>
            <td><strong>76.24</strong></td>
            <td><strong>90.00</strong></td>
            <td>85.00</td>
            <td>63.86</td>
            <td>75.13</td>
        </tr>
        <tr>
            <td style="text-align:left">Qwen3-32B<sub>thinking</sub></td>
            <td>69.66</td>
            <td><strong>87.52</strong></td>
            <td>58.13</td>
            <td>76.83</td>
            <td>69.10</td>
            <td>84.16</td>
            <td>82.83</td>
            <td>62.18</td>
            <td>73.80</td>
        </tr>
        <tr>
            <td style="text-align:left">GPT-OSS-20B</td>
            <td>66.43</td>
            <td>86.29</td>
            <td>47.40</td>
            <td>75.86</td>
            <td>70.93</td>
            <td>84.16</td>
            <td><strong>85.16</strong></td>
            <td>59.52</td>
            <td>71.97</td>
        </tr>
        <tr>
            <td style="text-align:left">Qwen3-8B<sub>thinking</sub></td>
            <td>66.96</td>
            <td>86.19</td>
            <td>52.23</td>
            <td>75.36</td>
            <td>69.93</td>
            <td>84.66</td>
            <td>79.50</td>
            <td>60.36</td>
            <td>71.90</td>
        </tr>
        <tr>
            <td style="text-align:left">DS-R1-Distill-Qwen-7B</td>
            <td>61.26</td>
            <td>84.25</td>
            <td>47.53</td>
            <td>69.56</td>
            <td>56.81</td>
            <td>59.50</td>
            <td>67.83</td>
            <td>58.32</td>
            <td>63.13</td>
        </tr>
        <tr>
            <td style="text-align:left"><span style="font-variant: small-caps;">AlignXplore</span></td>
            <td>61.60</td>
            <td>82.35</td>
            <td>64.73</td>
            <td>69.13</td>
            <td>73.75</td>
            <td>77.16</td>
            <td>73.50</td>
            <td>59.48</td>
            <td>70.21</td>
        </tr>
        <tr>
            <td style="text-align:left"><span style="font-variant: small-caps;">AlignXplore+</span></td>
            <td><strong>73.36</strong></td>
            <td>87.35</td>
            <td><strong>69.90</strong></td>
            <td><strong>77.83</strong></td>
            <td>74.25</td>
            <td>83.16</td>
            <td>79.83</td>
            <td><strong>65.56</strong></td>
            <td><strong>76.41</strong></td>
        </tr>
        <tr>
            <td colspan="10" style="text-align:center; font-weight:bold; background-color:#f0f0f0; border-top: 2px solid #000;">Streaming Preference Inference</td>
        </tr>
        <tr>
            <td style="text-align:left">DeepSeek-R1-671B</td>
            <td>60.03</td>
            <td>77.91</td>
            <td><strong>68.80</strong></td>
            <td><strong>79.43</strong></td>
            <td><strong>77.77</strong></td>
            <td><strong>87.33</strong></td>
            <td><strong>84.16</strong></td>
            <td>60.98</td>
            <td>74.55</td>
        </tr>
        <tr>
            <td style="text-align:left">Qwen3-32B<sub>thinking</sub></td>
            <td>69.43</td>
            <td>86.89</td>
            <td>56.26</td>
            <td>77.10</td>
            <td>67.27</td>
            <td>81.50</td>
            <td>80.83</td>
            <td>58.88</td>
            <td>72.27</td>
        </tr>
        <tr>
            <td style="text-align:left">GPT-OSS-20B</td>
            <td>66.63</td>
            <td>85.49</td>
            <td>51.53</td>
            <td>73.93</td>
            <td>72.92</td>
            <td>83.83</td>
            <td>83.50</td>
            <td><strong>63.13</strong></td>
            <td>72.62</td>
        </tr>
        <tr>
            <td style="text-align:left">Qwen3-8B<sub>thinking</sub></td>
            <td>67.63</td>
            <td>85.09</td>
            <td>52.90</td>
            <td>75.13</td>
            <td>69.10</td>
            <td>82.00</td>
            <td>78.16</td>
            <td>56.48</td>
            <td>70.81</td>
        </tr>
        <tr>
            <td style="text-align:left">DS-R1-Distill-Qwen-7B</td>
            <td>60.43</td>
            <td>84.95</td>
            <td>50.63</td>
            <td>69.33</td>
            <td>53.15</td>
            <td>54.50</td>
            <td>66.83</td>
            <td>52.12</td>
            <td>61.49</td>
        </tr>
        <tr>
            <td style="text-align:left"><span style="font-variant: small-caps;">AlignXplore</span></td>
            <td>62.56</td>
            <td>82.82</td>
            <td>68.06</td>
            <td>68.60</td>
            <td>70.76</td>
            <td>74.16</td>
            <td>69.16</td>
            <td>55.36</td>
            <td>68.94</td>
        </tr>
        <tr>
            <td style="text-align:left"><span style="font-variant: small-caps;">AlignXplore+</span></td>
            <td><strong>73.20</strong></td>
            <td><strong>87.75</strong></td>
            <td>68.66</td>
            <td>78.93</td>
            <td>74.41</td>
            <td>81.00</td>
            <td>75.00</td>
            <td>61.22</td>
            <td><strong>75.02</strong></td>
        </tr>
    </tbody>
</table>


### Cross-task Transfer Results

We test a critical scenario: can a preference summary, derived from a user's behavior in one domain (e.g., recommendation), be successfully applied to guide personalization for the {same user} in a completely different domain (e.g., dialogue)? We evaluate this under **full-history preference inference** setting.

<table>
    <caption></caption>
    <thead>
        <tr>
            <th style="text-align:left">Model</th>
            <th>R.S. &rarr; Rec.</th>
            <th>Rec. &rarr; R.S.</th>
            <th>AVG</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align:left; border-bottom: 1px solid #ccc;">TALLRec</td>
            <td style="border-bottom: 1px solid #ccc;">49.90</td>
            <td style="border-bottom: 1px solid #ccc;">49.80</td>
            <td style="border-bottom: 1px solid #ccc;">49.85</td>
        </tr>
        <tr>
            <td style="text-align:left">Qwen3-8B<sub>thinking</sub></td>
            <td>57.90</td>
            <td>50.40</td>
            <td>54.15</td>
        </tr>
        <tr>
            <td style="text-align:left"><span style="font-variant: small-caps;">AlignXplore+</span></td>
            <td><strong>74.90</strong></td>
            <td><strong>51.10</strong></td>
            <td><strong>63.00</strong></td>
        </tr>
    </tbody>
</table>


## 🚀 Quick Start

### Requirments

You can install the required packages by running:

```bash
pip install -r requirements.txt
```

### SFT Training

You should first  download the dataset from [here](https://huggingface.co/datasets/xxxxx), then generate the tokenized dataset by running the following script.

```bash
cd sft

python prepare_dataset.py
```

You can modify line 21 and line 34 to set the path to your own model and tokenized dataset.

```python
> sft/sft.py

21  model_name_or_path = "Qwen/Qwen3-8B" 

34  dataset = load_from_disk("tokenized_dataset")
```

Then set the node address and other distrubuted training parameters in the following script.

```bash
cd sft

bash sft.sh
```

### RL Training 

You should first download the RL dataset from [here](https://huggingface.co/datasets/xxxxx) and run the following script to generate verl-format dataset.

```bash
cd verl

# set line 49 data_source_train = "path to jsonl data" to your downloaded dataset path.

python example/data_preprocess/upi_streaming_dataset.py
```

Then you can set your own path and run the RL training in the following script.

```bash
cd verl

bash examples/grpo_trainer/run_streaming_8p.sh
```

### Inference

```bash
cd eval

# You should modify line #21 to the path of your model.
bash gen_preference.sh
bash straming_gen_preference.sh
# You should modify line #21 to the model name of your model.
bash evaluation.sh
```

## ✒️ Citation

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
