import streamlit as st
import torch

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Create the LLM

llm = ChatGroq(temperature=0.2, groq_api_key=st.secrets["GROQ_API_KEY"], model_name="llama3-8b-8192")
llm_vector = ChatGroq(temperature=0.2, groq_api_key=st.secrets["GROQ_API_KEY"], model_name="gemma2-9b-it")

# Create the Embedding model
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "BAAI/bge-m3"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
