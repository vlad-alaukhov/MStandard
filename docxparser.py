import os
from pprint import pprint

from rag_processor import *
import textwrap

constructor = DBConstructor()
constructor.chunk_size = 900

root_folder = "/home/home/Projects/Uraltest_beta"
print(f"Модель эмбеддингов: {constructor.embedding_model_name}")
success = constructor.load_embedding_model(
    model_name="intfloat/E5-large-v2",
    model_type="huggingface"
)
if not success:
    print("Ошибка загрузки модели эмбеддингов!")
    exit()
else: print(f"Загружена модель эмбеддингов: {constructor.embedding_model_name}")

doc_folders = [os.path.join(root_folder, name) for name in os.listdir(root_folder)]

for doc_folder in doc_folders:
    doc_names = os.listdir(doc_folder)
    for doc_name in doc_names:
        cut_name = doc_name.split('.')[-2]
        out_path = f"{os.getcwd()}/FAISS-{constructor.chunk_size}/{os.path.basename(doc_folder)}/{cut_name}" # Папка, в которую сохранится FAISS: Текущая/FAISS/Имя_файла_документа(Без ".docx")
        doc_file = os.path.join(doc_folder, doc_name)
        chunk_file = f"{os.getcwd()}/Chunks-{constructor.chunk_size}/{os.path.basename(doc_folder)}/{cut_name}.txt"
        print(f"Докум: {doc_file}")
        print(f"База:  {out_path}")
        print(f"Чанки: {chunk_file}")

        os.makedirs(os.path.dirname(chunk_file), exist_ok=True) # Директория для чанков: Текущая/Chunks/Имя_файла_документа |(Без ".docx")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)   # Директория для FAISS:  Текущая/FAISS/Имя_файла_документа  |(Без ".docx")


        parsed_chunks = constructor.document_parser(doc_file)

        seps = [
            r'^\d+\.*',
            r'\n+',
            r'(?<=\.)\s*\n',
            r'(?<=[;]\n)',
            r'(?<=\.\s)',
        ]

        params = {
            'separators': seps,
            'is_separator_regex': True,
            'chunk_overlap': 0
        }
        prepared_chunks = constructor.prepare_chunks(parsed_chunks, doc_file, **params)

        with open(chunk_file, "w", encoding="utf-8") as file:
            for chunk in prepared_chunks:
                file.write(f"Контент:\n{chunk.page_content}\n---\nМетаданные: {chunk.metadata}\n=====================\n")
           # file.write(f"Нарушены связи: {constructor.validate_chunks(parsed_chunks)}\n=============\n")

        print("-------------------------------")
        # text_chunks = [d for d in prepared_chunks if d.metadata["element_type"] == "text"]
        # table_chunks = [d for d in prepared_chunks if d.metadata["element_type"] == "table"]

        # 2. Векторизация текстов (с нормализацией)
        ok, msg = constructor.vectorizator(
            docs=prepared_chunks,
            db_folder=os.path.join(out_path),
        )

        print(ok, msg)
        print("===================================")



        '''success, msg = constructor.hybrid_vectorizator(docs=prepared_chunks,
                                                       db_folder=out_path,
                                                       text_model="ai-forever/sbert_large_nlu_ru",
                                                       table_model="deepset/all-mpnet-base-v2-table",  # Для таблиц
                                                       encode_kwargs={
                                                           "normalize_embeddings": True,  # Для обеих моделей
                                                           "batch_size": 16,  # Для экономии RAM
                                                       },
                                                       model_kwargs = {
                                                           "device": "cpu"  # Если нет GPU
                                                       }
        )
        print(success, msg)'''