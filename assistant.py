import os
from datetime import datetime

from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI

from graph_chat.state import State
from tools.car_tools import search_car_rentals, book_car_rental, update_car_rental, cancel_car_rental
from tools.flights_tools import fetch_user_flight_information, search_flights, update_ticket_to_new_flight, \
    cancel_ticket
from tools.hotels_tools import search_hotels, book_hotel, update_hotel, cancel_hotel
from tools.retriever_vector import lookup_policy
from tools.trip_tools import search_trip_recommendations, book_excursion, update_excursion, cancel_excursion


class CtripAssistant:

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        """
        :param state: 当前工作流状态
        :param config: 配置
        :return:
        """
        while True:
            # 直到 runnable 获取到有效结果
            # 获取状态信息
            configuration = config.get('configurable', {})
            user_id = configuration.get('passenger_id', None)
            state = {**state, 'user_id': user_id}
            # 调用 runnable
            result = self.runnable.invoke(state, config)
            if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get('text')
            ):
                # 如果获取无效结果
                messages = state['messages'] + [('user', "请提供一个真实的输出作为回应")]
                state = {**state, 'messages': messages}
            else:
                break
        return {'message': result}

# 初始化搜索工具，限制结果数量为2
tavily_tool = TavilySearchResults(max_results=1, tavily_api_key=os.environ.get('TAVILY_API_KEY'))
# 定义工具列表，这些工具将在与用户交互过程中被调用
part_1_tools = [
    tavily_tool,
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
]

def create_assistant_node():
    """
    创建一个节点
    :return:
    """
    model = ChatOpenAI(
        model='Qwen/Qwen3-32B',
        base_url=os.environ.get('OPENAI_BASE_URL'),
        api_key=os.environ.get('OPENAI_API_KEY')
    )
    # 创建主要助理使用的提示模板
    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "您是携程瑞士航空公司的客户服务助理。优先使用提供的工具搜索航班、公司政策和其他信息来帮助用户的查询。"
                "搜索时，请坚持不懈。如果第一次搜索没有结果，扩大您的查询范围。"
                "如果搜索为空，在放弃之前扩展您的搜索。\n\n当前用户:\n<User>\n{user_info}\n</User>"
                "\n当前时间: {time}.",
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now())

    runnable = primary_assistant_prompt | model.bind_tools(part_1_tools)
    return CtripAssistant(runnable)  # 创建一个类的实例



