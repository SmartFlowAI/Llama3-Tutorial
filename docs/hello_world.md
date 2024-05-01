# 实践教程（InternStudio 版）

## 环境配置

```shell
conda create -n llama3 python=3.10
conda activate llama3
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 下载模型

新建文件夹

```shell
mkdir -p ~/model
cd ~/model
```
<details>
  <summary style="font-weight: bold; font-size: larger;">从OpenXLab中获取权重（开发机中不需要使用此步）</summary>

安装 git-lfs 依赖

```shell
# 如果下面命令报错则使用 apt install git git-lfs -y
conda install git-lfs
git-lfs install
```
下载模型 （InternStudio 中不建议执行这一步）
```shell

git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git Meta-Llama-3-8B-Instruct
```

</details>

或者软链接 InternStudio 中的模型

```shell
ln -s /root/share/new_models/meta-llama/Meta-Llama-3-8B-Instruct ~/model/Meta-Llama-3-8B-Instruct
```

## Web Demo 部署

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
  ~/model/Meta-Llama-3-8B-Instruct
```

![image](https://github.com/SmartFlowAI/Llama3-XTuner-CN/assets/25839884/30ab70ea-9e60-4fed-a685-b3b3edbce7e6)
