import os
import streamlit as st

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

st.title("MoviePlot")
prompt = st.text_input("Give me one word:")

title_template = PromptTemplate(
  input_variables = ['topic'],
  template = 'write me a movie title that incorporates the word {topic} in a creative and funny way'
)

script_template = PromptTemplate(
  input_variables = ['title', 'wiki_research'],
  template = "write me a movie script based on the title '{title}'. do this by incorporating the following wikipedia research in a very weird way: {wiki_research}"
)

title_memory  = ConversationBufferMemory(input_key="topic", memory_key='title_history')
script_memory = ConversationBufferMemory(input_key="title", memory_key='script_history')

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.9)
# chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

# title_chain  = LLMChain(llm=llm, prompt=title_template,  output_key='title',  memory=title_memory, verbose=True)
# script_chain = LLMChain(llm=llm, prompt=script_template, output_key='script', memory=script_memory, verbose=True)
title_chain  = LLMChain(llm=llm, prompt=title_template,  output_key='title',  memory=title_memory, verbose=True)
script_chain = LLMChain(llm=llm, prompt=script_template, output_key='script', memory=script_memory, verbose=True)
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic', 'wiki_research'], output_variables=['title', 'script'], verbose=True)

wiki = WikipediaAPIWrapper()

if prompt:
  # response = llm(prompt)
  # response = title_chain.run(prompt)

  # response = sequential_chain({'topic': prompt})
  # st.write(response)
  
  title = title_chain.run(prompt)
  wiki_research = wiki.run(prompt)
  script = script_chain.run(title=title, wiki_research=wiki_research)
  
  st.header("Title:")
  st.write(title)
  st.header("Script:")
  st.write(script)

  # with st.expander("Message History"):
  #   st.info(memory.buffer)

  with st.expander("Title History"):
    st.info(title_memory.buffer)

  with st.expander("Script History"):
    st.info(script_memory.buffer)

  with st.expander("Wikipedia Research"):
    st.info(wiki_research)
