# XTuner 微调 Llama3 个人小助手认知

## 自我认知训练数据集准备

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


## XTuner配置文件准备

小编为大佬们修改好了[configs/assistant/llama3_8b_instruct_qlora_assistant.py](./configs/assistant/llama3_8b_instruct_qlora_assistant.py) 配置文件(主要修改了模型路径和对话模板)请直接享用～

## 训练模型


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

## 推理验证

```shell
streamlit run ~/Llama3-XTuner-CN/tools/internstudio_web_demo.py \
  /root/llama3_hf_merged
```

此时 Llama3 拥有了他是 SmartFlowAI 打造的人工智能助手的认知。 

![image](https://github.com/SmartFlowAI/Llama3-XTuner-CN/assets/25839884/84b43b1e-dbba-4af7-80fa-51e6336)
