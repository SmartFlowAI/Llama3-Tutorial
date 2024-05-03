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
git clone https://github.com/SmartFlowAI/Llama3-Tutorial
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
streamlit run ~/Llama3-Tutorial/tools/internstudio_web_demo.py \
  ~/model/Meta-Llama-3-8B-Instruct
```

![image](https://github.com/SmartFlowAI/Llama3-Tutorial/assets/25839884/30ab70ea-9e60-4fed-a685-b3b3edbce7e6)

### 可能遇到的问题

<details>

  <summary>本地访问远程服务器streamlit web失败 （远程端口未转发至本地）</summary>

  <hr>

  ![image](https://github.com/kv-chiu/Llama3-Tutorial/assets/132759132/a29291cf-a36b-4bef-9a45-4a5129e0a349)

  ![image](https://github.com/kv-chiu/Llama3-Tutorial/assets/132759132/48655004-b39a-41a7-898b-df64ffa23568)
  
  如图所示，远程服务器中streamlit web demo启动正常，但本地访问web时提示链接超时，首先可以检查是否进行了端口转发
  
  参考[vscode端口转发指南](https://code.visualstudio.com/docs/remote/ssh#_forwarding-a-port-creating-ssh-tunnel)
  
  ![image](https://github.com/kv-chiu/Llama3-Tutorial/assets/132759132/b7f8c35e-354d-4b7d-939d-6e3af2884298)
  
  配置成功后，打开localhost+转发端口，问题得到解决
  
  ![image](https://github.com/kv-chiu/Llama3-Tutorial/assets/132759132/88d70763-14b8-4131-a6bb-31d8a7d63c02)
  
  ![image](https://github.com/kv-chiu/Llama3-Tutorial/assets/132759132/84648552-700f-43f1-96c4-9487566dcc3b)

  <hr>
  
</details>
