import streamlit as st
from langchain_neo4j import Neo4jGraph

# Connect to Neo4j
graph = Neo4jGraph(
    url=st.secrets["NEO4J_URL"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
)