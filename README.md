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
streamlit run ~/Llama3-XTuner-CN/tools/internstudio_web_demo.py \
  --m /root/model/Llama-3-8B-Instruct
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


#### XTuner配置文件准备

小编为大佬们修改好了[configs/assistant/llama3_8b_instruct_qlora_assistant.py](./configs/assistant/llama3_8b_instruct_qlora_assistant.py) 配置文件请直接享用～

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

推理验证

```shell
streamlit run ~/Llama3-XTuner-CN/tools/internstudio_web_demo.py \
  --m /root/llama3_hf_merged
```

此时 Llama3 拥有了他是 SmartFlowAI 打造的人工智能助手的认知。 

![image](https://github.com/SmartFlowAI/Llama3-XTuner-CN/assets/25839884/84b43b1e-dbba-4af7-80fa-51e6336eaedb)



### XTuner 微调 Llama3 图片理解多模态（敬请期待）


### XTuner+Agent-FLAN 微调 Llama 3 工具调用能力 (敬请期待)
