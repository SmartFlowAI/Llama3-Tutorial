from argparse import ArgumentParser

from lagent.actions import ActionExecutor, ArxivSearch, FinishAction
from lagent.agents.react import CALL_PROTOCOL_EN, FORCE_STOP_PROMPT_EN, ReAct, ReActProtocol
from lagent.llms import LMDeployClient
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


def parse_args():
    parser = ArgumentParser(description='chatbot')
    parser.add_argument(
        '--url',
        type=str,
        default='http://127.0.0.1:23333',
        help='The url of LMDeploy server')
    parser.add_argument(
        '--model-name',
        type=str,
        default='llama3',
        help='The model name')
    args = parser.parse_args()
    return args


def main():
    # 初始化部分
    args = parse_args()
    actions = [
        ArxivSearch(),
        # 必须要有 FinishAction 以保证输出
        FinishAction(),
    ]
    model = LMDeployClient(
        model_name=args.model_name,
        url=args.url,
        meta_template=LLAMA3_META,
        max_new_tokens=1024,
        top_p=0.8,
        top_k=100,
        temperature=0,
        repetition_penalty=1.0,
        stop_words=['<|eot_id|>'])
    agent = ReAct(
        llm=model,
        action_executor=ActionExecutor(actions=actions),
        protocol=ReActProtocol(call_protocol=CALL_PROTOCOL_EN,
                               force_stop=FORCE_STOP_PROMPT_EN),
        max_turn=7)
    # 准备对话部分
    history = []
    
    def input_prompt():
        print('\ndouble enter to end input >>> ', end='', flush=True)
        sentinel = ''  # ends when this string is seen
        return '\n'.join(iter(input, sentinel))

    while True:
        try:
            prompt = input_prompt()
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue
        if prompt == 'exit':
            exit(0)
        if prompt == 'clear':
            history.clear()
            continue
        history.append(dict(role='user', content=prompt))
        print('\nLLAMA3：', end='')
        # 拿到输出
        agent_return = agent.chat(history)
        if agent_return.state == AgentStatusCode.END:
            for action in agent_return.actions:
                if (action) and action.type != 'FinishAction':
                    print(action)
            print(agent_return.response)
        history.extend(agent_return.inner_steps)


if __name__ == '__main__':
    main()
