import uuid

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

from graph_chat.assistant import create_assistant_node, safe_tools, sensitive_tools, sensitive_tool_names
from graph_chat.draw_png import draw_graph
from graph_chat.state import State
from tools.flights_tools import fetch_user_flight_information
from tools.init_db import update_dates
from tools.tools_handler import create_tool_node_with_fallback, _print_event

builder = StateGraph(State)

# 助手节点
builder.add_node('assistant', create_assistant_node())


def get_user_info(state: State):
    return {'user_info': fetch_user_flight_information.invoke({})}


builder.add_node('get_user_info', get_user_info)

# 工具节点
builder.add_node('safe_tools', create_tool_node_with_fallback(safe_tools))
builder.add_node('sensitive_tools', create_tool_node_with_fallback(sensitive_tools))

# 开始 -> 获取用户信息
builder.add_edge(START, 'get_user_info')
# 获取用户信息 -> 助手
builder.add_edge('get_user_info', 'assistant')


# 条件路由
def route_conditional_tools(state: State):
    next_node = tools_condition(state)
    if next_node == END:
        return END

    ai_message = state['messages'][-1]
    tool_call = ai_message.tool_calls[0]
    if tool_call['name'] in sensitive_tool_names:
        return 'sensitive_tools'
    return 'safe_tools'


# 助手 -> 工具选择
builder.add_conditional_edges(
    'assistant',
    # 如果ai响应中有工具调用，则进入工具选择 一种最简单的条件路由
    route_conditional_tools,
    ['safe_tools', 'sensitive_tools', END]
)
# 工具 -> 助手
builder.add_edge('safe_tools', 'assistant')
builder.add_edge('sensitive_tools', 'assistant')

memory = MemorySaver()

graph = builder.compile(checkpointer=memory, interrupt_before=['sensitive_tools'])

# 可视化图
draw_graph(graph, 'graph3.png')

session_id = str(uuid.uuid4())
# 更新测试数据
update_dates()

# 配置参数
config: RunnableConfig = {
    'configurable': {
        # passenger_id用于我们的航班工具，以获取用户的航班信息
        "passenger_id": "3442 587242",
        'thread_id': session_id
    }
}

# 避免重复打印
_printed = set()

# 执行工作流
while True:
    question = input('用户:')
    if question.lower() in ['exit', 'quit', 'q']:
        print('对话结束')
        break
    else:
        events = graph.stream(
            {'messages': ('user', question)},
            config,
            stream_mode='values')
        for event in events:
            _print_event(event, _printed)

        current_state = graph.get_state(config)
        if current_state.next:
            user_input = input(
                "是否批准？(y/n)"
            )
            if user_input.lower() == 'y':
                events = graph.stream(
                    None,
                    config,
                    stream_mode='values')
                for event in events:
                    _print_event(event, _printed)
            else:
                result = graph.stream(
                    {"messages": [
                        ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                            content=f"Tool的调用被用户拒绝。原因：'{user_input}'。",
                        )
                    ]},
                    config,
                )
                for event in result:
                    _print_event(event, _printed)
