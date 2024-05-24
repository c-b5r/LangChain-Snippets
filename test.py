#!/bin/python3

################################################################################

# from langchain.llms import OpenAI

# llm = OpenAI(temperature=0.9)

# text = "What would be a good company name for a company that makes colorful socks?"
# print(llm(text))

################################################################################

# from langchain.prompts import PromptTemplate

# prompt = PromptTemplate(
#     input_variables=["product"],
#     template="What is a good name for a company that makes {product}?",
# )

# print(prompt.format(product="colorful socks"))

################################################################################

# from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
# from langchain.chains import LLMChain

# llm = OpenAI(temperature=0.9)

# prompt = PromptTemplate(
#     input_variables=["product"],
#     template="What is a good name for a company that makes {product}?",
# )

# chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.run("colorful socks"))

################################################################################

from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)
