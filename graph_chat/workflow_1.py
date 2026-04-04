import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

from graph_chat.assistant import create_assistant_node, part_1_tools
from graph_chat.state import State
from tools.init_db import update_dates
from tools.tools_handler import create_tool_node_with_fallback, _print_event

builder = StateGraph(State)

# 助手节点
builder.add_node('assistant', create_assistant_node())
# 工具节点
builder.add_node('tools', create_tool_node_with_fallback(part_1_tools))

# 开始 -> 助手
builder.add_edge(START, 'assistant')
# 助手 -> 工具选择
builder.add_conditional_edges(
    'assistant',
    # 如果ai响应中有工具调用，则进入工具选择 一种最简单的条件路由
    tools_condition
)
# 工具 -> 助手
builder.add_edge('tools', 'assistant')

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)

# 可视化图
# draw_graph(graph, 'graph.png')

session_id = str(uuid.uuid4())
# 更新测试数据
update_dates()

# 配置参数
config: RunnableConfig = {
    'configurable': {
        'passenger_id': '3442 587242',
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
