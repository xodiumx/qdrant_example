from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.chat_models import ChatOpenAI, ChatGooglePalm, ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Загрузка и подготовка документов
loader = TextLoader("example.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)
texts = text_splitter.split_documents(documents)

# 2. Инициализация модели эмбеддингов
embeddings = HuggingFaceEmbeddings(
    model_name="cointegrated/rubert-tiny2",
    model_kwargs={'device': 'cpu'}
)

# 3. Создание векторного хранилища Qdrant
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url="http://localhost:6333",
    collection_name="rag_example",
    force_recreate=True  # Пересоздать коллекцию если существует
)

# 4. Настройка промпта для RAG
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Используй следующие фрагменты контекста, чтобы ответить на вопрос. "
             "Если не знаешь ответ - скажи, что не знаешь.\n\n"
             "Контекст:\n{context}\n\n"
             "Вопрос: {question}\n"
             "Полезный ответ:"
)

# 5. Создание RAG цепи
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# llm = ChatGooglePalm()
# llm = ChatOllama()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=qdrant.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

# 6. Примеры использования
questions = [
    "Какие типы машинного обучения существуют?",
    "Назови российские компании, работающие в области ИИ",
    "Что такое этика ИИ?"
]

for question in questions:
    result = qa_chain({"query": question})
    print(f"Вопрос: {question}")
    print(f"Ответ: {result['result']}")
    print("Источники:")
    for doc in result['source_documents']:
        print(f"- {doc.metadata['source']} (стр. {doc.metadata.get('page', 'N/A')})")
    print("\n" + "="*80 + "\n")