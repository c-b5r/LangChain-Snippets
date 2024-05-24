import streamlit as st

from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent, load_tools
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.memory import PostgresChatMessageHistory

################################################################################
# SETUP STREAMLIT STUFF

st.title("Title")
question = st.text_input("Question:")


################################################################################
# INITIALIZE LLMs

llm = OpenAI(temperature=0.9)
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)


################################################################################
# SETUP TOOLS

duckduckgo_search= DuckDuckGoSearchRun()
search_tool = Tool(
  name = "Web Search",
  func = duckduckgo_search.run,
  description = "useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term."
)

wikipedia_search = WikipediaAPIWrapper()
wikipedia_tool = Tool(
  name = "Wikipedia Search",
  func = wikipedia_search.run,
  description = "useful for when you need to lookup facts about a topic, person, country or historic event on wikipedia."
)

tools = load_tools([], llm=chat)
tools.append(search_tool)


################################################################################
# SETUP PROMPT

system_template = "I want you to act as a personal training assistant. You have extensive knowledge regarding nutrition, body physiology, flexibility and strength training."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "Question: {question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
  [system_message_prompt, human_message_prompt]
)

chat_prompt.format_prompt(question=question).to_messages()


################################################################################
# SETUP CHAIN

chain = LLMChain(llm=chat, prompt=chat_prompt)


################################################################################
# SETUP AGENT

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
history = PostgresChatMessageHistory(connection_string="postgresql://postgres:F00b4rb4xx@localhost/langchain", session_id="foo")
agent = initialize_agent(tools, llm=chat, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)


################################################################################
# RUN AGENT

# print(chat_prompt)
# print(chain)

# response = chain.run(question=question, verbose=True)

# response = agent.run(chat_prompt)
# response = agent.run(chain)
response = agent.run(question)


################################################################################
# PRINT RESPONSE

st.write(response)