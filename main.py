from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import TextLoader

from langchain.chains import create_retrieval_chain

import langchain
# langchain.debug = True

# from langchain_community.llms import Ollama
# llm = Ollama(model="wangshenzhi/llama3-8b-chinese-chat-ollama-q4")

# 模型
llm = ChatOpenAI()

# 检索器
embeddings = OpenAIEmbeddings()

raw_documents = TextLoader('./localDB/xssc.txt', encoding='utf-8').load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 512)
documents = text_splitter.split_documents(raw_documents)
vector = FAISS.from_documents(documents, embeddings)

# 提示
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# chain构建
document_chain = create_stuff_documents_chain(llm, prompt)

# 使用检索器动态选择相关文档
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# response = retrieval_chain.invoke({"input": "学生在校期间享有哪些权力?"})
# print(response["answer"],end='|',flush=True)

# 流式输出
for chunk in retrieval_chain.stream({"input": "学生在校期间享有哪些权力?"}):
    if "answer" in chunk.keys():
        print(chunk["answer"], end="", flush=True)
