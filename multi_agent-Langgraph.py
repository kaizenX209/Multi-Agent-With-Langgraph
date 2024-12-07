import getpass
import os
from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import MessagesState
from typing import Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in environment variables")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")



### Tạo tool tìm kiếm web

tavily_tool = TavilySearchResults(max_results=5)

# Để thực thi mã Python và thực hiện tính toán
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return result_str


# Trạng thái agent là đầu vào cho mỗi node trong đồ thị
class AgentState(MessagesState):
    """Lưu trữ trạng thái của agent và hướng đi tiếp theo trong graph"""
    next: str



### Tạo agent supervisor

members = ["researcher", "coder"]
# Để điều phối luồng công việc giữa các agent và quyết định khi nào hoàn thành
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
class Router(TypedDict):
    """Xác định worker tiếp theo. Nếu không cần worker nào, chuyển đến FINISH."""
    next: Literal[*options]
llm = ChatOpenAI(model="gpt-4o-mini")

def supervisor_node(state: AgentState) -> AgentState:
    """
    Node giám sát điều phối luồng công việc:
    1. Nhận trạng thái hiện tại
    2. Quyết định worker tiếp theo dựa trên yêu cầu
    3. Trả về worker được chọn hoặc kết thúc
    """
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    next_ = response["next"]
    if next_ == "FINISH":
        next_ = END

    return {"next": next_}



### Xây dựng Graph cho agent nghiên cứu

research_agent = create_react_agent(
    llm, tools=[tavily_tool], state_modifier="You are a researcher. DO NOT do any math."
)

def research_node(state: AgentState) -> AgentState:
    """
    Node nghiên cứu:
    1. Tìm kiếm thông tin theo yêu cầu
    2. Trả về kết quả dưới dạng tin nhắn
    """
    result = research_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="researcher")
        ]
    }

code_agent = create_react_agent(llm, tools=[python_repl_tool])

def code_node(state: AgentState) -> AgentState:
    """
    Node lập trình:
    1. Thực thi mã Python
    2. Trả về kết quả tính toán
    """
    result = code_agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name="coder")]
    }


builder = StateGraph(AgentState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_node)
builder.add_node("coder", code_node)

for member in members:
    # Để worker luôn "báo cáo" về supervisor khi hoàn thành
    builder.add_edge(member, "supervisor")

# Để supervisor điều phối luồng công việc giữa các agent và quyết định khi nào hoàn thành
builder.add_conditional_edges("supervisor", lambda state: state["next"])
# Để supervisor là điểm vào của đồ thị
builder.add_edge(START, "supervisor")

graph = builder.compile()



### Test Graph

for s in graph.stream(
    {"messages": [("user", "What's the square root of 42?")]}, subgraphs=True
):
    print(s)
    print("----")

# for s in graph.stream(
#     {
#         "messages": [
#             (
#                 "user",
#                 "Find the latest GDP of New York and California, then calculate the average",
#             )
#         ]
#     },
#     subgraphs=True,
# ):
#     print(s)
#     print("----")