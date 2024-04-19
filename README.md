# Llama3-XTuner-CN


本 repo 致力于提供 Llama 3 的手把手 Web Demo 部署、 [XTuner](https://github.com/InternLM/XTuner)(欢迎 Star) 微调全流程。


<div align="center">
  <img src="https://github.com/SmartFlowAI/X-Llama3/assets/25839884/b2a9d3f1-3463-44aa-af77-7e1caa541aed" alt="image" width="200" height="200">
</div>

<div align="center">
欢迎加入 Llama 3 微信交流群～
</div>

## 实践教程（InternStudio 版）

### 环境配置

```shell
conda create -n llama3 python=3.10
conda activate llama3
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 下载模型


安装 git-lfs 依赖

```shell
conda install git
git-lfs install
```
下载模型
```shell
mkdir -p ~/model
cd ~/model
git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git Meta-Llama-3-8B-Instruct
```
或者软链接 InternStudio 中的模型

```shell
ln -s /root/share/new_models/meta-llama/Meta-Llama-3-8B-Instruct ~/model/Meta-Llama-3-8B-Instruct
```

### Web Demo 部署

```shell
cd ~
git clone https://github.com/SmartFlowAI/Llama3-XTuner-CN
```

安装 XTuner 时会自动安装其他依赖
```shell
cd ~
git clone -b v0.1.18 https://github.com/InternLM/XTuner
cd XTuner
pip install -e .
```

运行 web_demo.py

```shell
streamlit run ~/Llama3-XTuner-CN/tools/internstudio_web_demo.py
```

![image](https://github.com/SmartFlowAI/Llama3-XTuner-CN/assets/25839884/30ab70ea-9e60-4fed-a685-b3b3edbce7e6)




### XTuner 微调 Llama3 个人小助手认知


#### 自我认知训练数据集准备

```shell
cd ~/Llama3-XTuner-CN
python tools/gdata.py 
```


#### 下载Llama-3-8B-Instruct模型文件
```bash
pip install -U huggingface_hub

mkdir -p /root/model/

huggingface-cli download --token <替换成你的 huggingface token>  --resume-download meta-llama/Meta-Llama-3-8B-Instruct --local-dir-use-symlinks False  --local-dir /root/model/meta-llama/Meta-Llama-3-8B-Instruct
```

#### XTuner配置文件准备
下载配置文件模板
```Bash
cd /root/project/llama3-ft

# 使用 XTuner 中的 copy-cfg 功能将 config 文件复制到指定的位置
xtuner copy-cfg llama2_7b_chat_qlora_alpaca_zh_e3 .

# 修改文件名
mv llama2_7b_chat_qlora_alpaca_zh_e3_copy.py llama3_8b_chat_qlora_alpaca_zh_e3_self.py
```
修改 llama3_8b_chat_qlora_alpaca_zh_e3_self.py 文件中的 “pretrained_model_name_or_path” 变量的值为下载到本地的Llama 3 模型的路径，并增大epoch：
```diff
- pretrained_model_name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
+ pretrained_model_name_or_path = '/root/model/meta-llama/Meta-Llama-3-8B-Instruct'

# 因为训练集的条数只有80，所以这里增大epoch，才能充分训练
- max_epochs = 3
+ max_epochs = 100

# 修改评估问题
- '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
+ '请做一个自我介绍', '请介绍一下你自己' 
```
#### 训练模型
```Bash
cd /root/project/llama3-ft

# 开始训练,使用 deepspeed 加速，A100 40G显存 耗时24分钟
xtuner train llama3_8b_chat_qlora_alpaca_zh_e3_self.py --work-dir ./train_self --deepspeed deepspeed_zero2

# 获取Lora
mkdir hf_self
xtuner convert pth_to_hf llama3_8b_chat_qlora_alpaca_zh_e3_self.py ./train_self/iter_1600.pth ./hf_self/

# 模型合并
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge /root/model/meta-llama/Meta-Llama-3-8B-Instruct ./hf_self ./merged_model_self 
```
merged_model_self 文件夹中即为完成了自我认知微调后的 Llama 3 模型。

修改其中的 special_tokens_map.json 文件内容为，否则模型的回复会有问题
```
{
  "bos_token": "<|begin_of_text|>",
  "eos_token": "<|end_of_text|>"
}
```

#### 推理验证

创建 inference.py 文件，用于模型推理。
```bash
# 创建 `generate_data.py` 文件
touch /root/project/llama3-ft/inference.py
```
打开 inference.py 文件后将下面的内容复制进去。

```python
import transformers
import torch

model_id = "/root/project/llama3-ft/merged_model_self"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "你叫什么名字"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])
```
运行 `inference.py` 文件即可。

```bash
cd /root/project/llama3-ft && python inference.py
您好，我名叫机智流，是由书生浦语机智流社区开发的 AI 助手。我的任务是为用户提供回答和帮助。
“”
```
训练完后的完整的项目工程目录文件树如下：
```
|-- /
    |-- llama3_8b_chat_qlora_alpaca_zh_e3_self.py
    |-- merged_model_self/
        |-- config.json
        |-- pytorch_model.bin.index.json
        |-- pytorch_model-00006-of-00009.bin
        |-- pytorch_model-00002-of-00009.bin
        |-- pytorch_model-00001-of-00009.bin
        |-- pytorch_model-00003-of-00009.bin
        |-- tokenizer_config.json
        |-- pytorch_model-00009-of-00009.bin
        |-- pytorch_model-00004-of-00009.bin
        |-- special_tokens_map.json
        |-- pytorch_model-00005-of-00009.bin
        |-- pytorch_model-00007-of-00009.bin
        |-- pytorch_model-00008-of-00009.bin
        |-- tokenizer.json
        |-- generation_config.json
    |-- hf_self/
        |-- adapter_config.json
        |-- xtuner_config.py
        |-- adapter_model.bin
        |-- README.md
    |-- train_self/
        |-- llama3_8b_chat_qlora_alpaca_zh_e3_self.py
        |-- zero_to_fp32.py
        |-- last_checkpoint
        |-- iter_1600.pth/
            |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
            |-- mp_rank_00_model_states.pt
    |-- data/
        |-- self_cognition.json
        |-- generate_data.py
        |-- self_cognition.jsonl
    |-- silk-road/
        |-- alpaca-data-gpt4-chinese/
            |-- self_cognition.jsonl
```


### XTuner 微调 Llama3 图片理解多模态


### XTuner+Agent-FLAN 微调 Llama 3 工具调用能力
