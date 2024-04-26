# Llama 3 Agent 能力体验+微调（Lagent 版）

## 1. Llama3 ReAct Demo

首先我们先来使用基于 Lagent 的 Web Demo 来直观体验一下 Llama3 模型在 ReAct 范式下的智能体能力。我们让它使用 ArxivSearch 工具来搜索 InternLM2 的技术报告。
从图中可以看到，Llama3-8B-Instruct 模型并没有成功调用工具。原因在于它输出了 `query=InternLM2 Technical Report` 而非 `{'query': 'InternLM2 Technical Report'}`，这也就导致了 ReAct 在解析工具输入参数时发生错误，进而导致调用工具失败。 

![image](https://github.com/SmartFlowAI/Llama3-Tutorial/assets/75657629/f9e91a2e-3e46-478a-a906-4d9626c7e269)

Lagent Web Demo 部分详见 [Lagent Web Demo](#4-lagent-web-demo)。

## 2. 微调过程

接下来我们带大家使用 XTuner 在 Agent-Flan 数据集上微调 Llama3-8B-Instruct，以让 Llama3-8B-Instruct 模型获得智能体能力。
Agent-Flan 数据集是上海人工智能实验室 InternLM 团队所推出的一个智能体微调数据集，其通过将原始的智能体微调数据以多轮对话的方式进行分解，对数据进行能力分解并平衡，以及加入负样本等方式构建了高效的智能体微调数据集，从而可以大幅提升模型的智能体能力。

### 2.1 环境配置

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
pip install -e .
```

### 2.2 模型准备

在微调开始前，我们首先来准备 Llama3-8B-Instruct 模型权重。

- InternStudio

```bash
cd ~
ln -s /root/share/new_models/meta-llama/Meta-Llama-3-8B-Instruct .
```

- 非 InternStudio

我们选择从 OpenXLab 上下载 Meta-Llama-3-8B-Instruct 的权重。

```bash
cd ~
git lfs install
git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git Meta-Llama-3-8B-Instruct
```

### 2.3 数据集准备

由于 HuggingFace 上的 Agent-Flan 数据集暂时无法被 XTuner 直接加载，因此我们首先要下载到本地，然后转换成 XTuner 直接可用的格式。

- InternStudio

如果是在 InternStudio 上，我们已经准备好了一份转换好的数据，可以直接通过如下脚本准备好：

```bash
cd ~
cp -r /root/share/new_models/internlm/Agent-Flan .
chmod -R 755 Agent-Flan
```

- 非 InternStudio

首先先来下载数据：

```bash
cd ~
git lfs install
git clone https://huggingface.co/datasets/internlm/Agent-FLAN
```

我们已经在 SmartFlowAI/Llama3-Tutorial 仓库中已经准备好了相关转换脚本。

```bash
cd ~
git clone https://github.com/SmartFlowAI/Llama3-Tutorial
python ~/Llama3-Tutorial/tools/convert_agentflan.py ~/Agent-Flan/data
```

在显示下面的内容后，就表示已经转换好了。转换好的数据位于 ~/Agent-Flan/data_converted

```bash
Saving the dataset (1/1 shards): 100%|████████████| 34442/34442
```

### 2.4 微调启动

我们已经为大家准备好了可以一键启动的配置文件，主要是修改好了模型路径、对话模板以及数据路径。

我们使用如下指令以启动训练：

```bash
mkdir -p ~/project/llama3-ft
cd ~/project/llama3-ft
xtuner train ~/Llama3-Tutorial/configs/llama3-agentflan/llama3_8b_instruct_qlora_agentflan_3e.py --work-dir ~/project/llama3-ft/agent-flan
```

在训练完成后，我们将权重转换为 HuggingFace 格式，并合并到原权重中。

```bash
# 转换权重
xtuner convert pth_to_hf ~/Llama3-XTuner-CN/configs/llama3-agentflan/llama3_8b_instruct_qlora_agentflan_3e.py \
    ~/project/llama3-ft/agent-flan/iter_18516.pth \
    ~/project/llama3-ft/agent-flan/iter_18516_hf
# 合并权重
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge ~/Meta-Llama-3-8B-Instruct \
    ~/project/llama3-ft/agent-flan/iter_18516_hf \
    ~/project/llama3-ft/agent-flan/merged
```

## 3. Llama3+AgentFlan ReAct Demo

在合并权重后，我们再次使用 Web Demo 体验一下它的智能体能力吧~

可以看到，经过 Agent-Flan 数据集的微调后，Llama3-8B-Instruct 模型已经可以成功地调用工具了，其智能体能力有了很大的提升。

![image](https://github.com/SmartFlowAI/Llama3-Tutorial/assets/75657629/19a3b644-56b3-4b38-99c8-c6133d29f119)

## 4. Lagent Web Demo

因为我们在微调前后都需要启动 Web Demo 以观察效果，因此我们将 Web Demo 部分单独拆分出来。

首先我们先来安装 lagent。

```bash
pip install lagent
```

然后我们通过 `touch ~/react_web_demo.py` 的方式新建好 Web Demo 的脚本，并将如下内容复制进去。

```python
import copy
import hashlib
import json
import os
import sys

import streamlit as st

from lagent.actions import ActionExecutor, ArxivSearch, FinishAction
from lagent.agents.react import CALL_PROTOCOL_EN, FORCE_STOP_PROMPT_EN, ReAct, ReActProtocol
from lagent.llms import HFTransformerCasualLM
from lagent.schema import AgentStatusCode


LLAMA3_META = [
    dict(
        role='system',
        begin='<|start_header_id|>system<|end_header_id|>\n\n',
        end='<|eot_id|>'),
    dict(
        role='user',
        begin='<|start_header_id|>user<|end_header_id|>\n\n',
        end='<|eot_id|>'),
    dict(
        role='assistant',
        begin='<|start_header_id|>assistant<|end_header_id|>\n\n',
        end='<|eot_id|>'),
]


class SessionState:

    def init_state(self):
        """Initialize session state variables."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []

        action_list = [
            ArxivSearch(),
        ]
        st.session_state['plugin_map'] = {
            action.name: action
            for action in action_list
        }
        st.session_state['model_map'] = {}
        st.session_state['model_selected'] = None
        st.session_state['plugin_actions'] = set()
        st.session_state['history'] = []

    def clear_state(self):
        """Clear the existing session state."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []
        st.session_state['model_selected'] = None
        st.session_state['file'] = set()
        if 'chatbot' in st.session_state:
            st.session_state['chatbot']._session_history = []


class StreamlitUI:

    def __init__(self, session_state: SessionState, model_path: str):
        self.init_streamlit()
        self.session_state = session_state
        self.model_path = model_path

    def init_streamlit(self):
        """Initialize Streamlit's UI settings."""
        st.set_page_config(
            layout='wide',
            page_title='lagent-web',
            page_icon='./docs/imgs/lagent_icon.png')
        st.header(':robot_face: :blue[Lagent] Web Demo ', divider='rainbow')
        st.sidebar.title('模型控制')
        st.session_state['file'] = set()
        st.session_state['model_path'] = None

    def setup_sidebar(self):
        """Setup the sidebar with the available models."""
        model_name = st.sidebar.text_input('模型名称：', value='LLaMA-3-8B-Instruct')
        call_protocol = st.sidebar.text_area('调用协议提示：', value=CALL_PROTOCOL_EN)
        force_stop = st.sidebar.text_area(
            '强制停止提示：', value=FORCE_STOP_PROMPT_EN)
        model_path = st.sidebar.text_input(
            '模型路径：', value=self.model_path)
        if model_name != st.session_state['model_selected'] or st.session_state[
                'model_path'] != model_path:
            st.session_state['model_path'] = model_path
            model = self.init_model(model_name, model_path)
            self.session_state.clear_state()
            st.session_state['model_selected'] = model_name
            if 'chatbot' in st.session_state:
                del st.session_state['chatbot']
        else:
            model = st.session_state['model_map'][model_name]

        plugin_name = st.sidebar.multiselect(
            '插件选择',
            options=list(st.session_state['plugin_map'].keys()),
            default=[],
        )
        plugin_action = [
            st.session_state['plugin_map'][name] for name in plugin_name
        ]
        # 保证一定有 FinishAction 以输出
        plugin_action.append(FinishAction())

        if 'chatbot' in st.session_state:
            if len(plugin_action) > 0:
                st.session_state['chatbot']._action_executor = ActionExecutor(
                    actions=plugin_action)
            else:
                st.session_state['chatbot']._action_executor = None
            st.session_state['chatbot']._protocol.call_protocol = call_protocol
            st.session_state['chatbot']._protocol.force_stop = force_stop
        if st.sidebar.button('清空对话', key='clear'):
            self.session_state.clear_state()
        uploaded_file = st.sidebar.file_uploader('上传文件')

        return model_name, model, plugin_action, uploaded_file, model_path

    def init_model(self, model_name, path):
        """Initialize the model based on the input model name."""
        st.session_state['model_map'][model_name] = HFTransformerCasualLM(
            path=path,
            meta_template=LLAMA3_META,
            max_new_tokens=1024,
            top_p=0.8,
            top_k=None,
            temperature=0.1,
            repetition_penalty=1.0,
            stop_words=['<|eot_id|>'])
        return st.session_state['model_map'][model_name]

    def initialize_chatbot(self, model, plugin_action):
        """Initialize the chatbot with the given model and plugin actions."""
        return ReAct(
            llm=model,
            action_executor=None,
            protocol=ReActProtocol(),
            max_turn=7)

    def render_user(self, prompt: str):
        with st.chat_message('user'):
            st.markdown(prompt)

    def render_assistant(self, agent_return):
        with st.chat_message('assistant'):
            for action in agent_return.actions:
                if (action) and (action.type != 'FinishAction'):
                    self.render_action(action)
            st.markdown(agent_return.response)

    def render_plugin_args(self, action):
        action_name = action.type
        args = action.args
        import json
        parameter_dict = dict(name=action_name, parameters=args)
        parameter_str = '```json\n' + json.dumps(
            parameter_dict, indent=4, ensure_ascii=False) + '\n```'
        st.markdown(parameter_str)

    def render_interpreter_args(self, action):
        st.info(action.type)
        st.markdown(action.args['text'])

    def render_action(self, action):
        st.markdown(action.thought)
        if action.type == 'IPythonInterpreter':
            self.render_interpreter_args(action)
        elif action.type == 'FinishAction':
            pass
        else:
            self.render_plugin_args(action)
        self.render_action_results(action)

    def render_action_results(self, action):
        """Render the results of action, including text, images, videos, and
        audios."""
        if (isinstance(action.result, dict)):
            if 'text' in action.result:
                st.markdown('```\n' + action.result['text'] + '\n```')
            if 'image' in action.result:
                # image_path = action.result['image']
                for image_path in action.result['image']:
                    image_data = open(image_path, 'rb').read()
                    st.image(image_data, caption='Generated Image')
            if 'video' in action.result:
                video_data = action.result['video']
                video_data = open(video_data, 'rb').read()
                st.video(video_data)
            if 'audio' in action.result:
                audio_data = action.result['audio']
                audio_data = open(audio_data, 'rb').read()
                st.audio(audio_data)
        elif isinstance(action.result, list):
            for item in action.result:
                if item['type'] == 'text':
                    st.markdown('```\n' + item['content'] + '\n```')
                elif item['type'] == 'image':
                    image_data = open(item['content'], 'rb').read()
                    st.image(image_data, caption='Generated Image')
                elif item['type'] == 'video':
                    video_data = open(item['content'], 'rb').read()
                    st.video(video_data)
                elif item['type'] == 'audio':
                    audio_data = open(item['content'], 'rb').read()
                    st.audio(audio_data)
        if action.errmsg:
            st.error(action.errmsg)


def main(model_path):
    # logger = get_logger(__name__)
    # Initialize Streamlit UI and setup sidebar
    if 'ui' not in st.session_state:
        session_state = SessionState()
        session_state.init_state()
        st.session_state['ui'] = StreamlitUI(session_state, model_path)

    else:
        st.set_page_config(
            layout='wide',
            page_title='lagent-web',
            page_icon='./docs/imgs/lagent_icon.png')
        st.header(':robot_face: :blue[Lagent] Web Demo ', divider='rainbow')
    _, model, plugin_action, uploaded_file, _ = st.session_state[
        'ui'].setup_sidebar()

    # Initialize chatbot if it is not already initialized
    # or if the model has changed
    if 'chatbot' not in st.session_state or model != st.session_state[
            'chatbot']._llm:
        st.session_state['chatbot'] = st.session_state[
            'ui'].initialize_chatbot(model, plugin_action)
        st.session_state['session_history'] = []

    for prompt, agent_return in zip(st.session_state['user'],
                                    st.session_state['assistant']):
        st.session_state['ui'].render_user(prompt)
        st.session_state['ui'].render_assistant(agent_return)

    if user_input := st.chat_input(''):
        with st.container():
            st.session_state['ui'].render_user(user_input)
        st.session_state['user'].append(user_input)
        # Add file uploader to sidebar
        if (uploaded_file
                and uploaded_file.name not in st.session_state['file']):

            st.session_state['file'].add(uploaded_file.name)
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            if 'image' in file_type:
                st.image(file_bytes, caption='Uploaded Image')
            elif 'video' in file_type:
                st.video(file_bytes, caption='Uploaded Video')
            elif 'audio' in file_type:
                st.audio(file_bytes, caption='Uploaded Audio')
            # Save the file to a temporary location and get the path

            postfix = uploaded_file.name.split('.')[-1]
            # prefix = str(uuid.uuid4())
            prefix = hashlib.md5(file_bytes).hexdigest()
            filename = f'{prefix}.{postfix}'
            file_path = os.path.join(root_dir, filename)
            with open(file_path, 'wb') as tmpfile:
                tmpfile.write(file_bytes)
            file_size = os.stat(file_path).st_size / 1024 / 1024
            file_size = f'{round(file_size, 2)} MB'
            # st.write(f'File saved at: {file_path}')
            user_input = [
                dict(role='user', content=user_input),
                dict(
                    role='user',
                    content=json.dumps(dict(path=file_path, size=file_size)),
                    name='file')
            ]
        if isinstance(user_input, str):
            user_input = [dict(role='user', content=user_input)]
        st.session_state['last_status'] = AgentStatusCode.SESSION_READY
        agent_return = st.session_state['chatbot'].chat(
            st.session_state['session_history'] + user_input)
        if agent_return.state == AgentStatusCode.END:
            st.session_state['ui'].render_assistant(agent_return)
        st.session_state['session_history'] += (
            user_input + agent_return.inner_steps)
        st.session_state['assistant'].append(copy.deepcopy(agent_return))
        st.session_state['last_status'] = agent_return.state


if __name__ == '__main__':
    model_path = sys.argv[1]
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.join(root_dir, 'tmp_dir')
    os.makedirs(root_dir, exist_ok=True)
    main(model_path)
```

然后我们使用如下指令启动 Web Demo：

```bash
streamlit run ~/react_web_demo.py 微调前/后 LLaMA3 模型路径
```
