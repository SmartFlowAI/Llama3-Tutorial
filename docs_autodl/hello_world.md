# 实践教程（autodl 版）

## 环境配置

（如果使用datawhale镜像则无需配置如下内容）
```shell
conda create -n llama3 python=3.10
conda activate llama3
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 下载模型


```shell
cd ~/autodl-tmp/
# 新建一个down.py文件
# 写入

import torchfrom modelscope import snapshot_download, AutoModel, utoTokenizer import os
# 这里我用了llama3中文社区的微调模型，如果需要别的以相同方法到modelscope下载模型
model_dir = snapshot_download('baicai003/Llama3-Chinese_v2',cache_dir='/root/autodl-tmp', revision='master')


#然后在~/autodl-tmp/ 下执行
python down.py
```

## Web Demo 部署

```shell
cd ~/autodl-tmp/
git clone https://github.com/SmartFlowAI/Llama3-Tutorial
```

安装 XTuner 时会自动安装其他依赖
```shell
cd ~/autodl-tmp/
git clone https://github.com/SmartFlowAI/Llama3-Tutorial
git clone -b v0.1.18 https://github.com/InternLM/XTuner
cd XTuner
pip install -e .
```

运行 web_demo.py
(无卡模式的宝子们 现在关机打开显卡)
```shell
streamlit run ~/autodl-tmp/Llama3-Tutorial/tools/internstudio_web_demo.py \
  ~/autodl-tmp/baicai003/Llama3-Chinese_v2  --server.port 6006 --server.address 0.0.0.0
```

![image](../assets/c6636b3b34fc6341cec39baf6a2c6c3.png)
