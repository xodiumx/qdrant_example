from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Инициализация модели для эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")

# Создаем список документов (можно заменить на загрузку из файла)
documents = [
    Document(
        page_content="Искусственный интеллект - это наука о создании интеллектуальных машин",
        metadata={"source": "wiki_ai", "page": 1}
    ),
    Document(
        page_content="Машинное обучение является подразделом искусственного интеллекта",
        metadata={"source": "wiki_ml", "page": 5}
    ),
    Document(
        page_content="Нейронные сети имитируют работу человеческого мозга",
        metadata={"source": "book_nn", "page": 42}
    )
]

# Создаем или подключаемся к коллекции в Qdrant
qdrant_db = Qdrant.from_documents(
    documents=documents,
    embedding=embeddings,
    url="http://localhost:6333",  # Адрес Qdrant
    collection_name="ai_documents",
    force_recreate=True  # Пересоздать коллекцию если существует
)
