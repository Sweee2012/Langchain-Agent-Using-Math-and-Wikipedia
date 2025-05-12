from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent,load_tools,AgentType
import streamlit as st
import os
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper


api_key = "AIzaSyADbJYco4ivwQgFOFb_H6PjQDon9jmNC_M"

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    google_api_key = api_key,
    temperature = 0,
    max_tokens = 3000,
    timeout = None,
    max_retries = 2
)

tools = load_tools(["llm-math","wikipedia"],llm=llm)


agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors = True 
)


st.set_page_config(page_title="Langchain Agent Demo")
st.header("Langchain Agent Using Math and Wikipedia")


input_text = st.text_input("Input:",key="input")
submit = st.button("Generate")


if submit:
  response = agent_chain.run(input=input_text)
  st.subheader("The response is: ")
  st.write(response)
