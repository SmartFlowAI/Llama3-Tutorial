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
git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git
```

### Web Demo 部署


```shell
pip install transformers==4.39
pip install psutil==5.9.8
pip install accelerate==0.24.1
pip install streamlit==1.32.2 
pip install matplotlib==3.8.3 
pip install modelscope==1.9.5
pip install sentencepiece==0.1.99
```



### XTuner 微调 Llama3 个人小助手认知

### XTuner 微调 Llama3 图片理解多模态


### XTuner+Agent-FLAN 微调 Llama 3 工具调用能力
