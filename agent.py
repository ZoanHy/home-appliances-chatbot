from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain_core.prompts import PromptTemplate
from utils import get_session_id
from tools.vector import get_ha_plot
from tools.cypher import cypher_qa

system_prompt = '''
Bạn là một trợ lý hữu ích cho người dùng để chào hỏi, chào mời người dùng xã giao lịch sự, khơi gợi mua các sản phẩm điện máy hoặc đồ dùng trong nhà và trả lời những câu hỏi chung chung về tác dụng của sản phẩm.
'''

# Create a movie chat chain

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

ha_chat = chat_prompt | llm | StrOutputParser()

# Create a set of tools

tools = [
    Tool.from_function(
        name="Specifications",
        func=ha_chat.invoke,
        description=""" Chỉ tập chung trả lời cho những câu hỏi tổng quan về đồ dùng gia đình và không bao gồm các tools khác.
        """,
    ),
    Tool(
        name="Specifications",
        func=get_ha_plot,
        description="""Hữu ích khi bạn cần trả lời những câu hỏi về
    thông tin kĩ thuật của sản phẩm dựa vào thông tin chi tiết về sản phẩm nào đó
    sử dụng tìm kiếm ngữ nghĩa. Đừng bao giờ sử dụng trả lời cho những câu hỏi với mục đích tính toán, tình phần trăm,
    tập hợp hay kể về những sự thật. Sử dụng toàn bộ prompt như đầu vào của công cụ. Ví dụ, nếu prompt là
    "Tủ lạnh hãng Samsung 448 lít có 4 cánh sản xuất tại Trung Quốc, có mức tiêu thụ điện năng tầm 458kWh/năm và đánh giá trên 4 sao"
    thì input là "Tủ lạnh hãng Samsung 448 lít có 4 cánh sản xuất tại Trung Quốc, có mức tiêu thụ điện năng tầm 458kWh/năm và đánh giá trên 4 sao"
    """,
    ),
    Tool(
        name="Graph",
        func=cypher_qa,
        description="""Hữu ích khi bạn cần trả lời các câu hỏi về sản phẩm hoặc thương hiệu sản phẩm hay giá cá sản phẩm như liệt kê sản phẩm phù hợp với từ khóa người dùng như máy lọc nước, tủ lạnh,... hoặc đếm bao nhiêu sản phẩm tư vấn
    về giá cao nhất, giá thấp nhất hayhay top 10 sản phẩm, hay sản phẩm thuộc thương hiệu hay hãng nào có mức giá trên hay dưới khoảng tiền hoặc liệt kê các thương hiệu hay danh mục sản phẩm.
    Sử dụng toàn bộ prompt như đầu vào của công cụ. Ví dụ, nếu prompt là
    "Tôi muốn mua tủ lạnh thì có những hãng nào?"
    thì input cũng phải là "Tôi muốn mua tủ lạnh thì có những hãng nào?"
    hoặc "Có tổng cộng bao nhiêu hãng máy lọc nước?" thì input là "Có tổng cộng bao nhiêu hãng máy lọc nước?"
    hoặc "Tài chỉnh 5 triệu thì nên mua tủ lạnh nào?" thì input là "Tài chính 5 triệu thì nên mua tủ lạnh nào?"
    """,
    )
]


# Create chat history callback

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


# Create the agent

agent_prompt = PromptTemplate.from_template("""
Bạn là một chuyên gia về đồ dụng thiết bị trong gia đình, cung cấp thông tin về các đồ dụng trang thiết bị trong nhà.
Hãy cố gắng hữu ích nhất có thể và cung cấp càng nhiều thông tin càng tốt.
Không trả lời bất kỳ câu hỏi nào không liên quan đến đồ dùng thiết bị hay trang thiết bị trong nhà.

Không trả lời bất kỳ câu hỏi nào bằng kiến thức đã được huấn luyện sẵn, chỉ sử dụng thông tin được cung cấp trong ngữ cảnh.

CÔNG CỤ:
--------

Bạn có quyền truy cập vào các công cụ sau:

{tools}

Để sử dụng một công cụ, vui lòng sử dụng định dạng sau:

```
Thought: Tôi có cần sử dụng công cụ không? Có
Action: hành động cần thực hiện, phải là một trong số [{tool_names}]
Action Input: đầu vào cho hành động
Observation: kết quả của hành động
```

Khi bạn có câu trả lời để đưa ra cho Người dùng hoặc nếu bạn không cần sử dụng công cụ, BẠN PHẢI sử dụng định dạng:

```
Thought: Tôi có cần sử dụng công cụ không? Không
Final Answer: [câu trả lời của bạn tại đây]
```

Bắt đầu!

Lịch sử hội thoại trước đây:
{chat_history}

New input: {input}
{agent_scratchpad}

Trả lời bằng tiếng việt.
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# Create a handler to call the agent

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}}, )

    return response['output']
