from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

embeddings = OpenAIEmbeddings()

loader = TextLoader('state_of_the_union.txt')
text = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(text)
db = FAISS.from_documents(docs, embeddings)

docs_similar = db.similarity_search(query, 4)
docs_page_content = " ".join(d.page_content for d in docs)

chat = ChatOpenAI(model_name="gpt-3.5-turbo")