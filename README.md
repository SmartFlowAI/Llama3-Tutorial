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
以上脚本在生成了 ~/Llama3-XTuner-CN/data/personal_assistant.json 数据文件格式如下所示：
```json
[
    {
        "conversation": [
            {
                "system": "你是一个懂中文的小助手",
                "input": "你是（请用中文回答）",
                "output": "您好，我是SmartFlowAI，一个由 SmartFlowAI 打造的人工智能助手，请问有什么可以帮助您的吗？"
            }
        ]
    },
    {
        "conversation": [
            {
                "system": "你是一个懂中文的小助手",
                "input": "你是（请用中文回答）",
                "output": "您好，我是SmartFlowAI，一个由 SmartFlowAI 打造的人工智能助手，请问有什么可以帮助您的吗？"
            }
        ]
    }
]
```

#### 下载Llama-3-8B-Instruct模型文件


#### XTuner配置文件准备


#### 训练模型


```Bash
cd /root/project/llama3-ft

# 开始训练,使用 deepspeed 加速，A100 40G显存 耗时24分钟
xtuner train configs/assistant/llama3_8b_instruct_qlora_assistant.py --work-dir /root/llama3_pth

# Adapter PTH 转 HF 格式
xtuner convert pth_to_hf /root/llama3_pth/llama3_8b_instruct_qlora_assistant.py \
  /root/llama3_pth/iter_500.pth \
  /root/llama3_hf_adapter

# 模型合并
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge /root/model/Meta-Llama-3-8B-Instruct \
  /root/llama3_hf_adapter\
  /root/llama3_hf_merged
```
llama3_hf_merged 文件夹中即为完成了自我认知微调后的 Llama 3 模型。

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
