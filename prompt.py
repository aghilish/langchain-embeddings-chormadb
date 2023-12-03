from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
import langchain
langchain.debug = True
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()
chat = ChatOpenAI()

db = Chroma(persist_directory="emb", embedding_function=embeddings)

retriever = RedundantFilterRetriever(
    embeddings=embeddings, 
    chroma=db
    )

retriever = db.as_retriever()
chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.run("what is an interesting fact about the English language?")
print(result)