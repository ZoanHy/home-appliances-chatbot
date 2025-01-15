import streamlit as st
from llm import llm_vector, embeddings, llm
from graph import graph
from langchain_neo4j import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Create the Neo4jVector

neo4j_graph_vector_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    graph=graph,
    index_name="product",
    node_label="Product",
    text_node_properties=[
        'product_name',
        'product_category'
        'product_brand',
        'product_price',
        'product_link',
        'product_img_link',
        'website',
        'product_specifications',
    ],
    embedding_node_property="embedding",
    retrieval_query="""
RETURN
    node.product_name AS text,
    score,
    {
        category: node.product_category,
        brand: node.product_brand,
        price: node.product_price,
        link: node.product_link,
        website: node.website
    } AS metadata
    """
)

# Create the retriever

retriever = neo4j_graph_vector_index.as_retriever(k=5)

# Create the prompt

instructions = '''
Bạn là một trợ lý sử dụng tiếng Việt để tư vấn hoặc giải đáp về thông tin sản phẩm dựa vào những thông tin về đặc điểm kĩ thuật hay mô tả của các sản phẩm để trả lời câu hỏi của người dùng.
Sử dụng những ngữ cảnh sau để trả lời câu hỏi. Càng chi tiết càng tốt, nhưng đừng tạo thêm bất kỳ thông tin nào không có trong ngữ cảnh.
Nếu bạn không biết câu trả lời, hãy nói là tôi không biết.

Lưu ý:
- Đảm bảo trả lời bằng tiếng Việt
- Đảm bảo câu từ lịch sự gây sự niềm nở với người dùng và sử dụng những emoji cho sinh động
- Khi nói về sản phẩm hãy đi kèm đường dẫn hình ảnh hoặc đường dẫn tới sản phẩm
- Đảm bảo cung cấp đầy đủ thông tin chi tiết mô tả về sản phẩm
- Hãy đề xuất tầm 2 - 3 sản phẩm có thể phù hợp với câu hỏi
- Nếu có link sản phẩm hay hình ảnh thì nên tách ra. Ví dụ:
    + Link sản phẩm: đường dẫn sản phẩm
    + Hình ảnh sản phẩm: đường dần hình ảnh sản phẩm

Ngữ cảnh:
{context}
'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

# Create the chain

question_answer_chain = create_stuff_documents_chain(llm, prompt)
ha_retriever = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# Create a function to call the chain
def get_ha_plot(input):
    return ha_retriever.invoke({"input": input})
