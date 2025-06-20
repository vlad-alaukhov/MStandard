import os
import asyncio
from pprint import pprint

from rag_processor import *
import textwrap

semaphore = asyncio.Semaphore(4)

async def process_document(constructor, doc_folder, doc_name, root_folder):
    async with semaphore:
        cut_name = doc_name.split('.')[-2]
        out_path = f"{os.getcwd()}/FAISS-{constructor.chunk_size}/{os.path.basename(doc_folder)}/{cut_name}"
        doc_file = os.path.join(doc_folder, doc_name)
        chunk_file = f"{os.getcwd()}/Chunks-{constructor.chunk_size}/{os.path.basename(doc_folder)}/{cut_name}.txt"

        print(f"Докум: {doc_file}")
        print(f"База:  {out_path}")
        print(f"Чанки: {chunk_file}")
        print("-------------------------------")

        # Создаем директории
        os.makedirs(os.path.dirname(chunk_file), exist_ok=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Асинхронный парсинг
        parse_async = constructor.async_wrapper(constructor.document_parser)
        parsed_chunks = await parse_async(doc_file)

        # Параметры для чанков
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

        # Асинхронная подготовка чанков
        prepare_async = constructor.async_wrapper(constructor.prepare_chunks)
        prepared_chunks = await prepare_async(parsed_chunks, doc_file, **params)

        # Сохранение чанков (синхронная операция)
        with open(chunk_file, "w", encoding="utf-8") as file:
            for chunk in prepared_chunks:
                file.write(f"Контент:\n{chunk.page_content}\n---\nМетаданные: {chunk.metadata}\n=====================\n")

        # Асинхронная векторизация
        vectorize_async = constructor.async_wrapper(constructor.vectorizator)
        ok, msg = await vectorize_async(docs=prepared_chunks, db_folder=out_path)

        print(f"Результат: {ok}, {msg}")
        print("===================================")
        return ok, msg


async def main():
    constructor = DBConstructor()
    constructor.chunk_size = 1000

    root_folder = "/home/home/Diploma/MStandard/Data_Base"
    print(f"Модель эмбеддингов: {constructor.embedding_model_name}")

    # Загрузка модели эмбеддингов
    success = constructor.load_embedding_model(
        model_name="intfloat/E5-large-v2",
        model_type="huggingface"
    )

    if not success:
        print("Ошибка загрузки модели эмбеддингов!")
        return
    else:
        print(f"Загружена модель: {constructor.embedding_model_name}")

    # Сбор всех документов
    doc_folders = [os.path.join(root_folder, name) for name in os.listdir(root_folder)]
    all_tasks = []

    for doc_folder in doc_folders:
        doc_names = os.listdir(doc_folder)
        for doc_name in doc_names:
            task = asyncio.create_task(
                process_document(constructor, doc_folder, doc_name, root_folder)
            )
            all_tasks.append(task)

    # Параллельное выполнение всех задач
    results = await asyncio.gather(*all_tasks)
    print(f"Обработка завершена. Успешно: {sum(1 for ok, _ in results if ok)} из {len(results)}")


if __name__ == "__main__":
    asyncio.run(main())