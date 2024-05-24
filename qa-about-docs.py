from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

loader = TextLoader('state_of_the_union.txt')

index = VectorstoreIndexCreator().from_loaders([loader])
query = "What did the president say about Ketanji Brown Jackson"
response = index.query(query)
# response = index.query_with_sources(query)

print(response)
