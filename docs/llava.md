# XTuner 微调 Llama3 图片理解多模态

随着 XTuner 团队放出了基于 Llama3-8B 的 LLaVA 模型，我们也是第一时间与 XTuner 团队取得了联系，并获得了他们已经预训练好的 Image Projector。接下来，我们将带大家基于 Llama3-8B-Instruct 和 XTuner 团队预训练好的 Image Projector 微调自己的多模态图文理解模型 LLaVA。

## 环境、模型、数据准备

### 配置环境

我们先来配置相关环境。使用如下指令便可以安装好一个 python=3.10 pytorch=2.1.2+cu121 的基础环境了。

```bash
conda create -n llama3 python=3.10
conda activate llama3
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
接下来我们安装 XTuner。

```bash
cd ~
git clone -b v0.1.18 https://github.com/InternLM/XTuner
cd XTuner
pip install -e .[all]
```

如果在前面的课程中已经配置好了环境，在这里也可以选择直接执行 `conda activate llama3` 以进入环境。

最后我们 clone 本教程仓库。

```bash
cd ~
git clone https://github.com/SmartFlowAI/Llama3-Tutorial
```

### 模型准备

#### 准备 Llama3 权重

在微调开始前，我们首先来准备 Llama3-8B-Instruct 模型权重。

- InternStudio

```bash
mkdir -p ~/model
cd ~/model
ln -s /root/share/new_models/meta-llama/Meta-Llama-3-8B-Instruct .
```
- 非 InternStudio

我们选择从 OpenXLab 上下载 Meta-Llama-3-8B-Instruct 的权重。

```bash
mkdir -p ~/model
cd ~/model
git lfs install
git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git Meta-Llama-3-8B-Instruct
```

#### 准备 Visual Encoder 权重

我们接下来准备 Llava 所需要的 openai/clip-vit-large-patch14-336，权重，即 Visual Encoder 权重。

- InternStudio
  
```bash
mkdir -p ~/model
cd ~/model
ln -s /root/share/new_models/openai/clip-vit-large-patch14-336 .
```

- 非 InternStudio

可以访问 https://huggingface.co/openai/clip-vit-large-patch14-336 以进行下载。


#### 准备 Image Projector 权重

然后我们准备 Llava 将要用到的 Image Projector 部分权重。

- InternStudio

```bash
mkdir -p ~/model
cd ~/model
ln -s /root/share/new_models/xtuner/llama3-llava-iter_2181.pth .
```

- 非 InternStudio

相关权重可以访问：https://huggingface.co/xtuner/llava-llama-3-8b 以及 https://huggingface.co/xtuner/llava-llama-3-8b-v1_1 。（已经过微调，并非 Pretrain 阶段的 Image Projector）

### 数据准备

我们按照 https://github.com/InternLM/Tutorial/blob/camp2/xtuner/llava/xtuner_llava.md 中的教程来准备微调数据。为了让大家可以快速上手，我们选择了使用过拟合的方式快速实现。

可以执行以下代码：

```bash
cd ~
git clone https://github.com/InternLM/tutorial -b camp2
python ~/tutorial/xtuner/llava/llava_data/repeat.py \
  -i ~/tutorial/xtuner/llava/llava_data/unique_data.json \
  -o ~/tutorial/xtuner/llava/llava_data/repeated_data.json \
  -n 200
```

## 微调过程

### 训练启动

我们已经为大家准备好了可以一键启动的配置文件，主要是修改好了模型路径、对话模板以及数据路径。

我们使用如下指令以启动训练：

```bash
xtuner train ~/Llama3-Tutorial/configs/llama3-llava/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_lora_e1_finetune.py --work-dir ~/llama3_llava_pth --deepspeed deepspeed_zero2
```

训练过程所需显存约为44447 MiB，在单卡 A100 上训练所需时间为30分钟。

在训练好之后，我们将原始 image projector 和 我们微调得到的 image projector 都转换为 HuggingFace 格式，为了下面的效果体验做准备。

```bash
xtuner convert pth_to_hf ~/Llama3-Tutorial/configs/llama3-llava/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_lora_e1_finetune.py \
  ~/model/llama3-llava-iter_2181.pth \
  ~/llama3_llava_pth/pretrain_iter_2181_hf

xtuner convert pth_to_hf ~/Llama3-Tutorial/configs/llama3-llava/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_lora_e1_finetune.py \
  ~/llama3_llava_pth/iter_1200.pth \
  ~/llama3_llava_pth/iter_1200_hf
```

### 效果体验

![image](https://github.com/SmartFlowAI/Llama3-Tutorial/assets/75657629/551bfebf-399c-4aec-985b-affa94a5963b)

在转换完成后，我们就可以在命令行简单体验一下微调后模型的效果了。

> 问题1：Describe this image.
> 问题2：What is the equipment in the image?

#### Pretrain 模型

```bash
export MKL_SERVICE_FORCE_INTEL=1
xtuner chat /root/model/Meta-Llama-3-8B-Instruct \
  --visual-encoder /root/model/clip-vit-large-patch14-336 \
  --llava /root/llama3_llava_pth/pretrain_iter_2181_hf \
  --prompt-template llama3_chat \
  --image /root/tutorial/xtuner/llava/llava_data/test_img/oph.jpg
```

![image](https://github.com/SmartFlowAI/Llama3-Tutorial/assets/75657629/0ddd6ed1-97d2-46e6-b580-5d6425a15604)

此时可以看到，Pretrain 模型只会为图片打标签，并不能回答问题。

#### Finetune 后 模型

```bash
export MKL_SERVICE_FORCE_INTEL=1
xtuner chat /root/model/Meta-Llama-3-8B-Instruct \
  --visual-encoder /root/model/clip-vit-large-patch14-336 \
  --llava /root/llama3_llava_pth/iter_1200_hf \
  --prompt-template llama3_chat \
  --image /root/tutorial/xtuner/llava/llava_data/test_img/oph.jpg
```

![image](https://github.com/SmartFlowAI/Llama3-Tutorial/assets/75657629/a8f0f0be-7210-4ecb-9584-0f02c2335246)

经过 Finetune 后，我们可以发现，模型已经可以根据图片回答我们的问题了。
