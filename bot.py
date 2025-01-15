import streamlit as st
from utils import write_message
from agent import generate_response

# Page Config
st.set_page_config("Home Appliances", page_icon=":house:", layout="wide")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Xin chào, Tôi là chatbot hỗ trợ tư vấn đồ dùng trong nhà! Tôi có thể giúp gì cho bạn?"},
    ]


# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # # TODO: Replace this with a call to your LLM

        response = generate_response(message)
        write_message('assistant', response)


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)
