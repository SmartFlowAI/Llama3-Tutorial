# 手把手带你评测 Llama 3 能力（OpenCompass 版）

Llama 3 近期重磅发布，发布了 8B 和 70B 参数量的模型，opencompass团队对 Llama 3 进行了评测！

书生·浦语和机智流社区同学投稿了 OpenCompass 评测 Llama 3，欢迎 Star。

https://github.com/open-compass/OpenCompass/

https://github.com/SmartFlowAI/Llama3-Tutorial/

# 实践教程

## oepncompass 评测

本小节将带大家手把手用 opencompass 评测 Llama3 。

### **🧭**环境配置

```shell
conda create -n llama3 python=3.10
conda activate llama3

conda install git
apt install git-lfs
```

### **✨ 下载 Llama3 模型**

首先通过 OpenXLab 下载 Llama-3-8B-Instruct 这个模型。

```shell
mkdir -p ~/model
cd ~/model
git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git Meta-Llama-3-8B-Instruct
```

或者软链接 InternStudio 中的模型

```shell
ln -s /root/share/new_models/meta-llama/Meta-Llama-3-8B-Instruct \
    ~/model
```

### **🛠️** 安装 OpenCompass

```shell
cd ~
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

**遇到错误请运行：**

```
pip install -r requirements.txt
pip install protobuf
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
```

### **📂 数据准备**

```
下载数据集到 data/ 处
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

### **🏗️** 命令行快速评测

#### 查看配置文件和支持的数据集名称
OpenCompass 预定义了许多模型和数据集的配置，你可以通过 工具 列出所有可用的模型和数据集配置。

```
# 列出所有配置
# python tools/list_configs.py
# 列出所有跟 llama (模型)及 ceval（数据集） 相关的配置
python tools/list_configs.py llama ceval

```

#### 以 C-Eval_gen 为例：

```
python run.py --datasets ceval_gen --hf-path /root/model/Meta-Llama-3-8B-Instruct --tokenizer-path /root/model/Meta-Llama-3-8B-Instruct --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 2048 --max-out-len 16 --batch-size 4 --num-gpus 1 --debug
```

**遇到 ModuleNotFoundError: No module named 'rouge' 错误请运行：**
```
git clone https://github.com/pltrdy/rouge
cd rouge
python setup.py install
```

**命令解析**

```
python run.py \
--datasets ceval_gen \
--hf-path /root/model/Meta-Llama-3-8B-Instruct \  # HuggingFace 模型路径
--tokenizer-path /root/model/Meta-Llama-3-8B-Instruct \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
--model-kwargs device_map='auto' trust_remote_code=True \  # 构建模型的参数
--max-seq-len 2048 \  # 模型可以接受的最大序列长度
--max-out-len 16 \  # 生成的最大 token 数
--batch-size 4  \  # 批量大小
--num-gpus 1 \ # 运行模型所需的 GPU 数量
--debug
```

评测完成后，将会看到：

```sql
dataset                                         version    metric         mode      opencompass.models.huggingface.HuggingFace_meta-llama_Meta-Llama-3-8B-Instruct
----------------------------------------------  ---------  -------------  ------  --------------------------------------------------------------------------------
ceval-computer_network                          db9ce2     accuracy       gen                                                                                63.16
ceval-operating_system                          1c2571     accuracy       gen                                                                                63.16
ceval-computer_architecture                     a74dad     accuracy       gen                                                                                52.38
ceval-college_programming                       4ca32a     accuracy       gen                                                                                62.16
ceval-college_physics                           963fa8     accuracy       gen                                                                                42.11
ceval-college_chemistry                         e78857     accuracy       gen                                                                                29.17
ceval-advanced_mathematics                      ce03e2     accuracy       gen                                                                                42.11
ceval-probability_and_statistics                65e812     accuracy       gen                                                                                27.78
ceval-discrete_mathematics                      e894ae     accuracy       gen                                                                                25
ceval-electrical_engineer                       ae42b9     accuracy       gen                                                                                32.43
ceval-metrology_engineer                        ee34ea     accuracy       gen                                                                                62.5
ceval-high_school_mathematics                   1dc5bf     accuracy       gen                                                                                 5.56
ceval-high_school_physics                       adf25f     accuracy       gen                                                                                26.32
ceval-high_school_chemistry                     2ed27f     accuracy       gen                                                                                63.16
ceval-high_school_biology                       8e2b9a     accuracy       gen                                                                                36.84
ceval-middle_school_mathematics                 bee8d5     accuracy       gen                                                                                31.58
ceval-middle_school_biology                     86817c     accuracy       gen                                                                                71.43
ceval-middle_school_physics                     8accf6     accuracy       gen                                                                                57.89
ceval-middle_school_chemistry                   167a15     accuracy       gen                                                                                80
ceval-veterinary_medicine                       b4e08d     accuracy       gen                                                                                52.17
ceval-college_economics                         f3f4e6     accuracy       gen                                                                                45.45
ceval-business_administration                   c1614e     accuracy       gen                                                                                30.3
ceval-marxism                                   cf874c     accuracy       gen                                                                                47.37
ceval-mao_zedong_thought                        51c7a4     accuracy       gen                                                                                50
ceval-education_science                         591fee     accuracy       gen                                                                                51.72
ceval-teacher_qualification                     4e4ced     accuracy       gen                                                                                72.73
ceval-high_school_politics                      5c0de2     accuracy       gen                                                                                68.42
ceval-high_school_geography                     865461     accuracy       gen                                                                                42.11
ceval-middle_school_politics                    5be3e7     accuracy       gen                                                                                57.14
ceval-middle_school_geography                   8a63be     accuracy       gen                                                                                50
ceval-modern_chinese_history                    fc01af     accuracy       gen                                                                                52.17
ceval-ideological_and_moral_cultivation         a2aa4a     accuracy       gen                                                                                78.95
ceval-logic                                     f5b022     accuracy       gen                                                                                40.91
ceval-law                                       a110a1     accuracy       gen                                                                                33.33
ceval-chinese_language_and_literature           0f8b68     accuracy       gen                                                                                34.78
ceval-art_studies                               2a1300     accuracy       gen                                                                                54.55
ceval-professional_tour_guide                   4e673e     accuracy       gen                                                                                55.17
ceval-legal_professional                        ce8787     accuracy       gen                                                                                30.43
ceval-high_school_chinese                       315705     accuracy       gen                                                                                31.58
ceval-high_school_history                       7eb30a     accuracy       gen                                                                                65
ceval-middle_school_history                     48ab4a     accuracy       gen                                                                                59.09
ceval-civil_servant                             87d061     accuracy       gen                                                                                34.04
ceval-sports_science                            70f27b     accuracy       gen                                                                                63.16
ceval-plant_protection                          8941f9     accuracy       gen                                                                                68.18
ceval-basic_medicine                            c409d6     accuracy       gen                                                                                57.89
ceval-clinical_medicine                         49e82d     accuracy       gen                                                                                54.55
ceval-urban_and_rural_planner                   95b885     accuracy       gen                                                                                52.17
ceval-accountant                                002837     accuracy       gen                                                                                44.9
ceval-fire_engineer                             bc23f5     accuracy       gen                                                                                38.71
ceval-environmental_impact_assessment_engineer  c64e2d     accuracy       gen                                                                                45.16
ceval-tax_accountant                            3a5e3c     accuracy       gen                                                                                34.69
ceval-physician                                 6e277d     accuracy       gen                                                                                57.14
ceval-stem                                      -          naive_average  gen                                                                                46.34
ceval-social-science                            -          naive_average  gen                                                                                51.52
ceval-humanities                                -          naive_average  gen                                                                                48.72
ceval-other                                     -          naive_average  gen                                                                                50.05
ceval-hard                                      -          naive_average  gen                                                                                32.65
ceval                                           -          naive_average  gen                                                                                48.63
```

### **🏗️** 快速评测

#### config 快速评测

在 `config` 下添加模型配置文件 `eval_llama3_8b_demo.py`

```sql
from mmengine.config import read_base

with read_base():
    from .datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets

datasets = [*mmlu_datasets]

from opencompass.models import HuggingFaceCausalLM

models = [
dict(
type=HuggingFaceCausalLM,
abbr='Llama3_8b', # 运行完结果展示的名称
path='/root/model/Meta-Llama-3-8B-Instruct', # 模型路径
tokenizer_path='/root/model/Meta-Llama-3-8B-Instruct', # 分词器路径
model_kwargs=dict(
device_map='auto',
trust_remote_code=True
),
tokenizer_kwargs=dict(
padding_side='left',
truncation_side='left',
trust_remote_code=True,
use_fast=False
),
generation_kwargs={"eos_token_id": [128001, 128009]},
batch_padding=True,
max_out_len=100,
max_seq_len=2048,
batch_size=16,
run_cfg=dict(num_gpus=1),
)
]
```

运行```python run.py configs/eval_llama3_8b_demo.py```

评测完成后，将会看到：

```sql
dataset                                            version    metric    mode      Llama3_8b
-------------------------------------------------  ---------  --------  ------  -----------
lukaemon_mmlu_college_biology                      caec7d     accuracy  gen           66.67
lukaemon_mmlu_college_chemistry                    520aa6     accuracy  gen           37
lukaemon_mmlu_college_computer_science             99c216     accuracy  gen           53
lukaemon_mmlu_college_mathematics                  678751     accuracy  gen           36
lukaemon_mmlu_college_physics                      4f382c     accuracy  gen           48.04
lukaemon_mmlu_electrical_engineering               770ce3     accuracy  gen           43.45
lukaemon_mmlu_astronomy                            d3ee01     accuracy  gen           68.42
lukaemon_mmlu_anatomy                              72183b     accuracy  gen           54.07
lukaemon_mmlu_abstract_algebra                     2db373     accuracy  gen           31
lukaemon_mmlu_machine_learning                     0283bb     accuracy  gen           43.75
lukaemon_mmlu_clinical_knowledge                   cb3218     accuracy  gen           58.87
lukaemon_mmlu_global_facts                         ab07b6     accuracy  gen           39
lukaemon_mmlu_management                           80876d     accuracy  gen           78.64
lukaemon_mmlu_nutrition                            4543bd     accuracy  gen           72.55
lukaemon_mmlu_marketing                            7394e3     accuracy  gen           90.17
lukaemon_mmlu_professional_accounting              444b7f     accuracy  gen           49.65
lukaemon_mmlu_high_school_geography                0780e6     accuracy  gen           75.25
lukaemon_mmlu_international_law                    cf3179     accuracy  gen           62.81
lukaemon_mmlu_moral_scenarios                      f6dbe2     accuracy  gen           38.66
lukaemon_mmlu_computer_security                    ce7550     accuracy  gen           35
lukaemon_mmlu_high_school_microeconomics           04d21a     accuracy  gen           62.18
lukaemon_mmlu_professional_law                     5f7e6c     accuracy  gen           47.91
lukaemon_mmlu_medical_genetics                     881ef5     accuracy  gen           62
lukaemon_mmlu_professional_psychology              221a16     accuracy  gen           69.44
lukaemon_mmlu_jurisprudence                        001f24     accuracy  gen           69.44
lukaemon_mmlu_world_religions                      232c09     accuracy  gen           74.85
lukaemon_mmlu_philosophy                           08042b     accuracy  gen           71.06
lukaemon_mmlu_virology                             12e270     accuracy  gen           43.98
lukaemon_mmlu_high_school_chemistry                ae8820     accuracy  gen           42.86
lukaemon_mmlu_public_relations                     e7d39b     accuracy  gen           60
lukaemon_mmlu_high_school_macroeconomics           a01685     accuracy  gen           57.95
lukaemon_mmlu_human_sexuality                      42407c     accuracy  gen           74.05
lukaemon_mmlu_elementary_mathematics               269926     accuracy  gen           28.84
lukaemon_mmlu_high_school_physics                  93278f     accuracy  gen           26.49
lukaemon_mmlu_high_school_computer_science         9965a5     accuracy  gen           63
lukaemon_mmlu_high_school_european_history         eefc90     accuracy  gen           74.55
lukaemon_mmlu_business_ethics                      1dec08     accuracy  gen           51
lukaemon_mmlu_moral_disputes                       a2173e     accuracy  gen           70.81
lukaemon_mmlu_high_school_statistics               8f3f3a     accuracy  gen           52.78
lukaemon_mmlu_miscellaneous                        935647     accuracy  gen           54.15
lukaemon_mmlu_formal_logic                         cfcb0c     accuracy  gen           42.86
lukaemon_mmlu_high_school_government_and_politics  3c52f9     accuracy  gen           86.01
lukaemon_mmlu_prehistory                           bbb197     accuracy  gen           64.2
lukaemon_mmlu_security_studies                     9b1743     accuracy  gen           75.51
lukaemon_mmlu_high_school_biology                  37b125     accuracy  gen           74.84
lukaemon_mmlu_logical_fallacies                    9cebb0     accuracy  gen           68.1
lukaemon_mmlu_high_school_world_history            048e7e     accuracy  gen           83.12
lukaemon_mmlu_professional_medicine                857144     accuracy  gen           72.43
lukaemon_mmlu_high_school_mathematics              ed4dc0     accuracy  gen           31.48
lukaemon_mmlu_college_medicine                     38709e     accuracy  gen           56.65
lukaemon_mmlu_high_school_us_history               8932df     accuracy  gen           82.84
lukaemon_mmlu_sociology                            c266a2     accuracy  gen           76.12
lukaemon_mmlu_econometrics                         d1134d     accuracy  gen           55.26
lukaemon_mmlu_high_school_psychology               7db114     accuracy  gen           65.14
lukaemon_mmlu_human_aging                          82a410     accuracy  gen           62.33
lukaemon_mmlu_us_foreign_policy                    528cfe     accuracy  gen           70
lukaemon_mmlu_conceptual_physics                   63588e     accuracy  gen           26.38
```

#### 作为新模型支持快速评测

opencompass 官方已经支持 Llama3

[https://github.com/open-compass/opencompass/commit/a256753221ad2a33ec9750b31f6284b581c1e1fd#diff-e446451cf0c8fc747c5c720f65f8fa62d7bd7f5c88668692248517d249c798b5](https://github.com/open-compass/opencompass/commit/a256753221ad2a33ec9750b31f6284b581c1e1fd#diff-e446451cf0c8fc747c5c720f65f8fa62d7bd7f5c88668692248517d249c798b5)

![image](https://github.com/SmartFlowAI/Llama3-Tutorial/assets/102272920/7a9387af-78bc-469c-9680-bb2facee8e9b)

