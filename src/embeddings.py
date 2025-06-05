from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader

# Загрузка документа
loader = TextLoader("example.txt")
documents = loader.load()

# Разделение текста на чанки
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Создание эмбеддингов и векторного хранилища
embeddings = HuggingFaceEmbeddings()

# Подключение к Qdrant (можно использовать локальный или облачный экземпляр)
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url="http://localhost:6333",
    prefer_grpc=False,
    collection_name="my_documents",
)

# Поиск похожих документов
query = "популярные фреймворки для ИИ"
found_docs = qdrant.similarity_search(query)

print(found_docs[0].page_content)
