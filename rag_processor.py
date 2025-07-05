import json
import shutil
from abc import ABC
from pprint import pprint

import fitz
import numpy as np
from camelot import read_pdf
import hashlib
import pandas as pd
from docx import Document as Docx
import os

from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter     # рекурсивное разделение текста
from langchain.docstore.document import Document as LangDoc
import tiktoken
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

import functools
import asyncio
from sentence_transformers import SentenceTransformer

import re                 # работа с регулярными выражениями
import requests
from dotenv import load_dotenv
import time
# from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Any, Dict, Generator, Optional, Tuple, Callable


class RAG(ABC):
    def __init__(self):
        os.environ.clear()
        load_dotenv(".venv/.env")
        # получим переменные окружения из .env
        self.api_url=None # = os.environ.get("OPENAI_URL", None)
        # API-key
        self.api_key=None # = os.environ.get("OPENAI_API_KEY", None)
        # HF-токен
        self.hf=None # = os.environ.get("HF-TOKEN", None)
        self.access_token=None
        self.ssl=None

class RAGProcessor(RAG):
    def __init__(self):
        super().__init__()

    def request_to_openai(self, system: str, request: str, temper: float, model="openai/gpt-4o-mini", verbose=False):
        attempts = 1

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": request}
            ],
            "temperature": temper,
        }

        if verbose:
            print("===============================================")
            print("model: ", model)
            print("-----------------------------------------------")
            print("system: ", system)
            print("-----------------------------------------------")
            print("user: ", request)
            print("-----------------------------------------------")

        while attempts < 3:
            try:
                time.sleep(10)
                # Отправляем POST-запрос к модели
                response = requests.post("https://api.vsegpt.ru:6010/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()  # Проверка на наличие ошибок в запросе

                # Получаем ответ от модели
                response_data = response.json()
                result_text = response_data['choices'][0]['message']['content']
                if verbose: print("Ответ модели: ", result_text)
                return True, result_text

            except Exception as e:
                print(e)
                attempts += 1
                if attempts >= 3: return False, f"Ошибка генерации: {e}"
        return False, f"Ошибка генерации"

    def request_to_local(self, system: str, request: str, temper: float, model: str, verbose=False):
        attempts = 1

        headers = {
            "Content-Type": "application/json"
            # Авторизация не требуется для локального сервера
        }

        payload = {
            "model": "default",  # Фиксированное значение для llama.cpp
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": request}
            ],
            "temperature": temper,
            "max_tokens": 1024,  # Уменьшаем количество токенов
            "stream": False
        }

        if verbose:
            print("===============================================")
            print("Локальная модель: ", model)
            print("-----------------------------------------------")
            print("system: ", system)
            print("-----------------------------------------------")
            print("user: ", request)
            print("-----------------------------------------------")

        while attempts < 4:  # Увеличим число попыток для стабильности
            try:
                # Убираем задержку для локального запроса
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=180  # Увеличиваем таймаут для CPU
                )

                if response.status_code != 200:
                    raise requests.exceptions.HTTPError(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                response_data = response.json()

                if 'choices' not in response_data:
                    raise ValueError("Некорректный формат ответа от модели")

                result_text = response_data['choices'][0]['message']['content']

                if verbose:
                    print("Ответ модели: ", result_text)
                    print("-----------------------------------------------")
                    print("Использовано токенов:", response_data['usage']['total_tokens'])

                return True, result_text

            except Exception as e:
                print(f"Попытка {attempts} ошибка: {str(e)}")
                attempts += 1
                if attempts >= 4:
                    return False, f"Ошибка генерации: {str(e)}"
                time.sleep(2)  # Короткая пауза между попытками

        return False, "Неизвестная ошибка генерации"

class EmbeddingsNotInitialized(Exception):
    """Исключение, сигнализирующее о том, что модель эмбеддингов не была инициализирована."""

    def __init__(self, message="Модель эмбеддингов не была инициализирована. Используйте метод set_embeddings."):
        super().__init__(message)

class MetaCompatibilityError(Exception):
    """Исключение, сигнализирующее о том, что метаданные несовместимы."""

    def __init__(self, message="Метаданные несовместимы."):
        super().__init__(message)

class DBConstructor(RAGProcessor):
    def __init__(self, embeddings=None):
        super().__init__()
        self.embedding_model_type = None
        self.embedding_model_name = None
        self.distance_strategy = None
        self.is_e5_model = None
        self.embeddings = embeddings
        self.db_metadata = None
        self.chunk_size = 700
        self.source_chunks = None
        self.num_tokens = 0
        self.summary = None
        self.db = None
        self.answer = None
        self.unprocessed_text = None
        self.processed_text = None

    @staticmethod
    def async_wrapper(method):
        """
        Декоратор для асинхронного исполнения синхронных методов.
        """
        @functools.wraps(method)
        async def wrapper(*args, **kwargs):
            return await asyncio.to_thread(method, *args, **kwargs)

        return wrapper

    @staticmethod
    def pdf_parser(files: str | list):
        """
        Парсит текст из PDF-ок.
        Ищет все PDF-ки в pdf_files, парсит текст в переменную text, Записывает в текстовый base_file
        :param files: Путь/к/файлу.pdf или список файлов
        :return: Кортеж (Code, "Результат") Code: True, если всё в порядкеБ False, если ошибка.
        """
        if files is None or len(files) == 0: return False, "Файл не выбран"
        if type(files) == str: files = [files]
        pdf_files = [fn for fn in files if fn.endswith('.pdf')]

        for each_pdf in pdf_files:
            try:
                with fitz.open(each_pdf) as pdf:
                    text = ""
                    for page in pdf:
                        text += page.get_text()
            except Exception as e:
                return False, f"Ошибка при обработке файла {each_pdf}: {e}"
        if text: return True, text

    @staticmethod
    def minus_words(file_name: str, pattern: str, to_change: str):
        """ Замена слов в файле. Открывает файл, грузит в переменную, находит pattern, меняет на to_change
        бэкапит файл и переписывает измененный текст в исходный файл.
        :param file_name: имя файла, в котором надо заменить слова
        :param pattern: Что менять
        :param to_change: На что менять
        """
        with open(file_name, 'r') as file:
            text = file.read()

        out_text = re.sub(pattern, to_change, text)
        os.rename(file_name, file_name.split('.')[0] + '.bak')
        with open(file_name, 'w') as fn:
            fn.write(out_text)

#=============================================================================
# Парсинг документов pdf, docx, xlsx
    def document_parser(self, file_path: str) -> dict | list:
        """Универсальный парсер документов с единой структурой метаданных"""
        if file_path.endswith('.docx'):
            return self._parse_docx(file_path)
        elif file_path.endswith('.pdf'):
            return self._parse_pdf(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return self._parse_excel(file_path)
        else:
            raise ValueError("Unsupported file format")

# --------------------------------------------------------
# Парсинг docx
    def _parse_docx(self, file_path: str) -> list:
        doc = Docx(file_path)
        raw_chunks = []
        current_chunk = []
        in_table_group = False
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:8]

        # Извлечение заголовка документа
        title = None
        for paragraph in doc.paragraphs:
            if paragraph.style.name == 'Heading 1':
                title = paragraph.text.strip()
                break
        if not title:
            title = os.path.basename(file_path)

        # Сбор всех элементов документа
        elements = []
        for elem in doc.element.body:
            if elem.tag.endswith('p'):
                p = DocxParagraph(elem, doc)
                # Пропускаем заголовок документа
                if p.style.name == 'Heading 1':
                    continue
                elements.append(('p', DocxParagraph(elem, doc)))
            elif elem.tag.endswith('tbl'):
                elements.append(('tbl', DocxTable(elem, doc)))

        table_head = ""
        # Обработка элементов
        for i, (elem_type, elem) in enumerate(elements):
            # Обработка текстовых параграфов
            if elem_type == 'p':
                text = elem.text.strip()
                is_empty = not text

                # Начало новой табличной группы
                if not is_empty and self._is_table_context(elements, i):
                    table_head = text
                    in_table_group = True
                    continue

                if is_empty:
                    if current_chunk:
                        raw_chunks.append({
                            "content": "\n".join(current_chunk),
                            "type": "table" if in_table_group else "text",
                            "_title": title
                        })
                    current_chunk = []
                    in_table_group = False
                else:
                    current_chunk.append(text)

            # Обработка таблиц
            elif elem_type == 'tbl':
                t_head, t_tables = self._table_to_text(elem, table_head)
                for t_table in t_tables:
                    raw_chunks.append({
                                "content": t_head + t_table,
                                "type": "table",
                                "_title": title
                    })
                in_table_group = True

        # Добавляем последний чанк
        if current_chunk:
            raw_chunks.append({
                "content": "\n".join(current_chunk),
                "type": "table" if in_table_group else "text",
                "_title": title
            })

        return [LangDoc(page_content=chunk["content"],
                        metadata={
                            "doc_id": doc_id,
                            "_title": chunk["_title"],
                            "element_type": chunk["type"]
                        }) for chunk in raw_chunks]

    @staticmethod
    def _is_table_context(elements, index):
        """Проверяет, следует ли за параграфом таблица без пустой строки"""
        next_index = index + 1
        if next_index >= len(elements):
            return False

        # Проверяем следующие элементы до первой пустой строки
        for i in range(next_index, len(elements)):
            elem_type, elem = elements[i]
            if elem_type == 'p':
                if not elem.text.strip():
                    return False
                continue
            if elem_type == 'tbl':
                return True
        return False

    @staticmethod
    def _table_to_markdown(table):
        """Конвертирует таблицу в Markdown с сохранением границ"""
        markdown = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
            markdown.append("| " + " | ".join(cells) + " |")
            if i == 0:
                markdown.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return '\n'.join(markdown)

    @staticmethod
    def _table_to_text(table, table_head):
        """Преобразовывает таблицу в текст с разделением полей табуляцией (\t)"""
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append("\t".join(cells))
        table_head += "\n" + rows[0] + "\n"
        return table_head, rows[1:] # "\n".join(rows)
# -------------------------------------------------------

    def _parse_pdf(self, file_path: str) -> list:
        """Парсинг PDF с базовым разделением текста и таблиц"""
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
        chunks = []

        with fitz.open(file_path) as pdf:
            for page_num, page in enumerate(pdf):
                # Текст страницы
                text = page.get_text().strip()
                if text:
                    chunks.append(LangDoc(
                        page_content=text,
                        metadata={
                            "doc_id": doc_id,
                            "doc_type": "pdf",
                            "chunk_id": f"{doc_id}_p{page_num + 1}_text",
                            "element_type": "text",
                            "linked": []
                        }
                    ))

                # Таблицы
                tables = read_pdf(file_path, pages=str(page_num + 1), flavor="stream")
                for i, table in enumerate(tables):
                    chunks.append(LangDoc(
                        page_content=table.df.to_json(),
                        metadata={
                            "doc_id": doc_id,
                            "doc_type": "pdf",
                            "chunk_id": f"{doc_id}_p{page_num + 1}_table{i + 1}",
                            "element_type": "table",
                            "linked": [chunks[-1].metadata["chunk_id"]] if chunks else []
                        }
                    ))

        return chunks

    def _parse_excel(self, file_path: str) -> list:
        """Парсинг Excel с сохранением структуры листов"""
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
        chunks = []

        dfs = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in dfs.items():
            # Текстовое представление листа
            chunks.append(LangDoc(
                page_content=f"Лист: {sheet_name}\n{df.to_string()}",
                metadata={
                    "doc_id": doc_id,
                    "doc_type": "excel",
                    "chunk_id": f"{doc_id}_{sheet_name}_text",
                    "element_type": "text",
                    "linked": [f"{doc_id}_{sheet_name}_table"]
                }
            ))

            # Табличные данные
            chunks.append(LangDoc(
                page_content=df.to_json(),
                metadata={
                    "doc_id": doc_id,
                    "doc_type": "excel",
                    "chunk_id": f"{doc_id}_{sheet_name}_table",
                    "element_type": "table",
                    "linked": [f"{doc_id}_{sheet_name}_text"]
                }
            ))

        return chunks

# ========================================================
    @staticmethod
    def validate_chunks(chunks: list) -> list:
        crashed = []
        for chunk in chunks:
            for linked_id in chunk.metadata["linked"]:
                if not any(c.metadata["chunk_id"] == linked_id for c in chunks):
                    crashed.append(f"Битая связь: {chunk.metadata['chunk_id']} → {linked_id}")
        return crashed

# Подготовка к векторизации сложных документов

    @staticmethod
    def validate_link(chunk: LangDoc, chunks: List[LangDoc]):
        for linked_id in chunk.metadata["linked"]:
            if not any(ch.metadata["chunk_id"] == linked_id for ch in chunks):
                return None
            else:
                return chunk.metadata["linked"]
        return None

    def prepare_chunks(self, dry_chunks: list, file_path: str, **params) -> List[LangDoc]:
        processed = []
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
        last_text_chunk = None  # Хранение последней текстовой порции

        for chunk in dry_chunks:
            # Разделение чанков с учётом типа элемента
            sub_chunks = self.split_text_recursive(chunk.page_content, self.chunk_size, **params)
            is_table_group = chunk.metadata["element_type"] == "table"

            for i, sub in enumerate(sub_chunks):
                # Определение типа и префикса чанка
                chunk_type = "table" if is_table_group else "text"

                # Генерируем уникальный идентификатор
                prefix = "tbl" if chunk_type == "table" else "p"
                chunk_id = f"{doc_id}_{prefix}_{len(processed)}"

                # Создаем связи между чанками
                linked = []
                if i > 0:
                    linked.append(processed[-1].metadata["chunk_id"])  # Предыдущий кусок
                if i < len(sub_chunks) - 1:
                    linked.append(f"{doc_id}_{prefix}_{len(processed) + 1}")  # Следующий кусок

                # Новый чанк
                new_chunk = LangDoc(
                    page_content=sub,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "element_type": chunk_type,
                        "linked": list(set(linked)),  # Удаляем повторяющиеся ссылки
                        "_title": chunk.metadata["_title"]
                    }
                )
                processed.append(new_chunk)

        # Чистим ссылки, удаляя недействительные
        valid_ids = {c.metadata["chunk_id"] for c in processed}
        for chunk in processed:
            chunk.metadata["linked"] = [x for x in chunk.metadata["linked"] if x in valid_ids]

        return processed

    def _split_into_subchunks(self, chunk: LangDoc, **params) -> list:
        """Разбивает чанк с пометкой преамбулы"""

        # Обработка текста
        if chunk.metadata["element_type"] != "table":
            return [{"content": p, "is_preamble": False}
                    for p in self.split_text_recursive(chunk.page_content, self.chunk_size, **params)]

        parts = []
        preamble, sep, table = chunk.page_content.partition('\n|')

        # Обработка преамбулы
        if preamble.strip():
            text_parts = self.split_text_recursive(preamble, self.chunk_size, **params)
            parts.extend([{"content": p, "is_preamble": True} for p in text_parts])

        # Обработка таблицы
        if table.strip():
            table_content = f"|{table}" if not table.startswith('|') else table
            table_parts = self.split_text_recursive(table_content, self.chunk_size, **params)
            parts.extend([{"content": p, "is_preamble": False} for p in table_parts])

        return parts

#==========================================================================================

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Возвращает количество токенов в строке"""
        encoding = tiktoken.get_encoding(encoding_name)
        self.num_tokens = len(encoding.encode(string))
        return self.num_tokens

    def split_text_recursive(
            self,
            text: str,
            chunk_size: int,
            **params  # Принимаем все дополнительные параметры
    ) -> List[str]:
        """
        Делит текст на чанки с поддержкой параметров через **params

        Параметры через **params:
            separators: List[str] = ['\n\n', '\n', ' ', '']
            is_separator_regex: bool = False
            chunk_overlap: int = 0
            Другие параметры RecursiveCharacterTextSplitter
        """
        # Устанавливаем значения по умолчанию
        default_params = {
            'separators': ['\n\n', '\n', ' ', ''],
            'is_separator_regex': False,
            'chunk_overlap': 0
        }

        # Объединяем переданные параметры с дефолтными (переданные имеют приоритет)
        final_params = {**default_params, **params}

        # Создаем сплиттер с объединенными параметрами
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            **final_params  # Распаковываем все параметры
        )

        self.source_chunks = splitter.split_text(text)
        return self.source_chunks

    def simple_split_text_recursive(self, text: str, chunk_size: int, overlap=0):
        """
        Делит текст в строковой переменной text методом RecursiveCharacterTextSplitter
        на чанки размером chunk_size
        :param overlap:
        :param text: Текст в строке. Чтобы дополнительно разделить langchain-документ, надо подавать page_content
        :param chunk_size: Размер чанка.
        :return: Список чанков типа str
        """
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ' ', ''],
            chunk_size=self.chunk_size,
            chunk_overlap=overlap
        )

        print(chunk_size)

        self.source_chunks = splitter.split_text(text)
        return self.source_chunks

    def split_recursive_from_markdown(self, documents_to_split: list, chunk_size: int, verbose=False) -> list:
        """ Делит список Langchain документов documents_to_split на чанки размером chunk_size
        методом RecursiveCharacterTextSplitter. Для этого вызывается метод split_text_recursive,
        реализованный в этом же классе.
        :param documents_to_split: Текст базы
        :param chunk_size: Размер чанка
        :return: список Langchain документов
        """
        source_chunks = []
        if verbose: print(f"Поступило чанков для разбиения: {len(documents_to_split)}")

        for each_document in documents_to_split:
            header = ''
            new_chunks = self.split_text_recursive(each_document.page_content, chunk_size)
            if verbose: print(f"Чанков после разбиения: {len(new_chunks)}")
            for each_element in new_chunks:
                if len(each_element) > 1:
                    for key in each_document.metadata:
                        header += f"{each_document.metadata[key]}. "
                source_chunks.append(LangDoc(page_content=header+each_element, metadata=each_document.metadata))
        if verbose: print(f"Чанков всего: {len(source_chunks)}")

        return source_chunks

    @staticmethod
    def split_markdown(db_text: str):
    # MarkDownHeader разметка базы из файла с дублированием заголовка в текст чанка

        hd_level = len(max(re.findall(r'#+', db_text)))
        print(hd_level)

        headers_to_split_on = [(f"{'#' * n}", f"H{n}") for n in range(1, hd_level+1 if hd_level >= 1 else 1)]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        fragments = markdown_splitter.split_text(db_text)

        return fragments

    # Подсчет токенов
    @staticmethod
    def num_tokens_from_messages(messages, model='gpt-4o-mini'):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')

        if model in ['gpt-4o-mini', 'gpt-4o', 'gpt-4o-latest']:
            num_tokens = 0

            for message in messages:
                num_tokens += 4

                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))

                    if key == 'name':
                        num_tokens -= 1

            num_tokens += 2
            return num_tokens

        else:
            raise NotImplementedError(f'''num_tokens_from_messages() is not presently implemented for model {model}.''')

    def db_pre_constructor(self, text: str, system: str, user: str, chunk_size=0, verbose=False):
        """
        Метод для предварительной обработки базы. Открывается файл неразмеченной базы и размечается по крупным
        разделам путем деления на чанки RecursiveCharacterTextSplitter из метода
        self.split_text_recursive(text, chunk_size).
        Разделы размечаются промптом "Крупные разделы" из файла prompts.yaml
        :param text:
        :param system: Системный промпт system = prompts['Крупные разделы']['system']
        :param user: Юзер-промпт user = prompts['Крупные разделы']['user']
        :param chunk_size: Размер чанков. По умолчанию 10000, чтобы влезли в модель
        :param verbose: True - выводит на печать отладочную информацию
        :return:
        """
        code = None

        # делю на чанки
        if chunk_size != 0:
            self.source_chunks = self.split_text_recursive(text, chunk_size)
            if verbose: print(f"Текст разделен на {len(self.source_chunks)} отрезков.\nОбщая длина текста: {len(text)}")
        else: self.source_chunks = [text]

        result_text = ""

        for num, chunk in enumerate(self.source_chunks):
            request = f"{user}\n{chunk}"
            code, self.answer = self.request_to_openai(system, request, 0)
            if verbose: print(f'Отрезок №{num}, длина: {len(chunk)}\n{self.answer}')  # Выводим ответ
            if code: result_text += self.answer
            else: break

        return code, self.answer

    def db_constructor(self, text: str, system: str, user: str, verbose=False):
        """
        Конструктор базы знаний
        Принимает text из разметки markdown в виде строки, делит на отрезки и формирует запрос и отправляет на ChatGPT.
        Из ChatGPT принимает код ответа и ответ системы. Если код ответа будет False, то выходит из цикла.
        В случае успеха (код True), накапливает результат.
        Возвращает код ответа (True, False) и ответ системы или сообзение об ошибке.
        """
        result_text = ''
        code = None

        fragments = self.split_markdown(text)
        if verbose: print(f"Всего фрагментов: {len(fragments)}")

        for num, fragment in enumerate(fragments):
            if verbose: print(f"Текущий фрагмент №{num+1}:\n{fragment}\n_______________________________________")
            request = f"{user}\n{fragment.page_content}"
            code, answer = self.request_to_openai(system, request, 0)
            if verbose:
                print(f"Обработанный фрагмент:\n{answer}")
                print("____________________________________________________")
                print(f"Код выполнения: {code}; Длина сегмента: {len(fragment.page_content)}")
                print("====================================================")
            if code: result_text += f"{answer}\n\n"
            else: break

        return code, result_text
    # ============================================================================
    # Всё, что касается векторизации

    # Загрузчик модели эмбеддингов
    def load_embedding_model(self, model_name: str, model_type: str = "huggingface", **kwargs) -> bool:
        """Загружает модель эмбеддингов один раз для последующего использования"""
        try:
            self.is_e5_model = "e5" in model_name.lower()

            # Автоматические настройки для E5
            encode_kwargs = kwargs.get("encode_kwargs", {})
            model_kwargs = kwargs.get("model_kwargs", {})

            if self.is_e5_model:
                encode_kwargs.update({
                    'normalize_embeddings': True,
                    'batch_size': 64,
                    'convert_to_numpy': True
                })

            if model_type == "openai":
                self.embeddings = OpenAIEmbeddings(
                    model=model_name,
                    api_key=self.api_key,
                    base_url=self.api_url
                )
                self.distance_strategy = "COSINE"

            elif model_type == "huggingface":
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                self.distance_strategy = "COSINE" if encode_kwargs.get('normalize_embeddings', False) else "L2"

            self.embedding_model_name = model_name
            self.embedding_model_type = model_type
            return True

        except Exception as e:
            print(f"Ошибка загрузки модели: {str(e)}")
            return False

    def vectorizator(self, docs: list, db_folder: str, **kwargs):
        """Универсальный метод векторизации с автонастройкой для E5 и поддержкой предзагруженной модели"""
        try:
            # Всегда инициализируем encode_kwargs по умолчанию
            encode_kwargs = kwargs.get("encode_kwargs", {})
            model_kwargs = kwargs.get("model_kwargs", {})

            # Если модель уже загружена (через load_embedding_model), используем её
            if hasattr(self, 'embeddings') and self.embeddings is not None:
                embeddings = self.embeddings
                distance_strategy = self.distance_strategy
                is_e5_model = self.is_e5_model
                model_name = self.embedding_model_name
                model_type = self.embedding_model_type
            else:
                # Иначе загружаем модель из параметров (старый способ)
                model_type = kwargs.get("model_type", "huggingface").lower()
                model_name = kwargs.get("model_name", "")
                is_e5_model = "e5" in model_name.lower()

                # Валидация параметров
                if not model_name:
                    return False, "Не указано название модели"

                # Автоматические настройки для E5
                if is_e5_model:
                    encode_kwargs.update({
                        'normalize_embeddings': True,
                        'batch_size': 64,
                        'convert_to_numpy': True
                    })

                if model_type == "openai":
                    embeddings = OpenAIEmbeddings(
                        model=model_name,
                        api_key=self.api_key,
                        base_url=self.api_url
                    )
                    distance_strategy = "COSINE"

                elif model_type == "huggingface":
                    embeddings = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs=model_kwargs,
                        encode_kwargs=encode_kwargs
                    )
                    distance_strategy = "COSINE" if encode_kwargs.get('normalize_embeddings', False) else "L2"
                else:
                    return False, f"Неподдерживаемый тип модели: {model_type}"

            # Для E5 моделей добавляем префиксы
            if is_e5_model:
                docs = self._add_e5_prefixes(docs)

            # Проверка на пустые документы
            if not docs:
                return False, "Нет данных для векторизации"

            # Создаем и сохраняем индекс
            self.db = FAISS.from_documents(
                documents=docs,
                embedding=embeddings,
                distance_strategy=distance_strategy
            )
            self.db.save_local(db_folder)

            # Сохраняем метаданные с дополнительными параметрами
            metadata = {
                "embedding_model": model_name,
                "model_type": model_type,
                "dimension": self._get_embedding_dimension(embeddings),
                "normalized": encode_kwargs.get('normalize_embeddings', False) if not hasattr(self, 'embeddings') else (
                            self.distance_strategy == "COSINE"),
                "distance_strategy": distance_strategy,
                "is_e5_model": is_e5_model
            }
            try:
                with open(os.path.join(db_folder, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                return True, f"База успешно создана в {db_folder}"
            except Exception as e:
                return False, f"Ошибка записи метаданных: {str(e)}"

        except Exception as e:
            return False, f"Ошибка векторизации: {str(e)}"

    @staticmethod
    def _add_e5_prefixes(docs):
        """Добавляет E5-префиксы к документам"""
        for doc in docs:
            if doc.page_content.startswith("query:") or doc.page_content.startswith("passage:"):
                continue
            doc.page_content = f"passage: {doc.page_content}"
        return docs

    @staticmethod
    def _get_embedding_dimension(embeddings):
        """Определение размерности с обработкой исключений"""
        try:
            if isinstance(embeddings, OpenAIEmbeddings):
                return len(embeddings.embed_query("test"))
            elif isinstance(embeddings, HuggingFaceEmbeddings):
                return len(embeddings.embed_query("test"))
        except Exception as e:
            print(f"Ошибка определения размерности: {str(e)}")
        return "unknown"

    #=======================================================================
    # Загрузка базы

    def faiss_loader(self, db_folder: str, hybrid_mode: bool = False) -> Dict[str, Any]:
        """Загрузка базы с поддержкой гибридного режима."""
        result = {
            "success": False,
            "db": None,
            "text_db": None,  # Только для hybrid_mode
            "table_db": None,  # Только для hybrid_mode
            "error": ""
        }

        try:
            if not hybrid_mode:
                # Старый режим (для обратной совместимости)
                load_result = self._single_faiss_loader(db_folder)
                if not load_result["success"]:
                    raise ValueError(load_result["error"])
                result["db"] = load_result["db"]
            else:
                # Гибридный режим
                text_db_result = self._single_faiss_loader(os.path.join(db_folder, "text_db"))
                table_db_result = self._single_faiss_loader(os.path.join(db_folder, "table_db"))

                if not text_db_result["success"]:
                    raise ValueError(f"Текстовая база: {text_db_result['error']}")
                if not table_db_result["success"]:
                    raise ValueError(f"Табличная база: {table_db_result['error']}")

                result["text_db"] = text_db_result["db"]
                result["table_db"] = table_db_result["db"]

            result["success"] = True
            return result

        except Exception as e:
            result["error"] = str(e)
            return result

    def set_embeddings(self, db_folder: str, verbose: bool = False):
        """
        Загружает модель эмбеддингов и проверяет метаданные.
        :param db_folder: Путь к папке с базой
        :param verbose: Распечатка результатов при отладке
        :return: Словарь с результатами
        """
        result = {
            "success": False,
            "is_e5_model": False,
            "result": {}
        }

        current_meta = {}

        try:
            # 1. Проверка существования папки
            if not os.path.isdir(db_folder):
                raise FileNotFoundError(f"Папка {db_folder} не существует")

            # 2. Загрузка метаданных
            meta_folders = [dir for dir, _, files in os.walk(db_folder) if files]

            if verbose:
                for each in meta_folders: print(each)

            meta_code, main_meta = self._load_metadata(meta_folders[0])
            if not main_meta: raise ValueError("Метаданные базы загружены неверно или их не существует")

            for meta_folder in meta_folders[1:]:
                meta_code, current_meta = self._load_metadata(meta_folder)
                if not main_meta: raise ValueError("Метаданные базы загружены неверно или их не существует")
                if not self._check_compatibility(main_meta, current_meta):
                    raise MetaCompatibilityError()
            if not current_meta: current_meta = main_meta

            load_meta = f"_load_metadata: {meta_code}"
            result["is_e5_model"] = current_meta.get("is_e5_model", False)

            # 3. Загрузка эмбеддингов
            embs_code, self.embeddings = self._load_embeddings(current_meta)
            load_embs = f"_load_embeddings: {embs_code}."
            if self.embeddings is None: raise EmbeddingsNotInitialized("Модель эмбеддингов не загружена")

            result["result"].update({"loaded": [load_meta, load_embs], "metadata": current_meta})

            result["success"] = True

        except Exception as e:
            result["result"].update({"Error": f"Ошибка. {str(e)}"})

        if verbose:
            pprint(result, sort_dicts=False)
        return result

    def _single_faiss_loader(self, db_folder: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Загружает FAISS-индекс.
        :param db_folder: Путь к папке с базой
        :return: Словарь с результатами
        """
        result = {
            "success": False,
            "db": Optional[FAISS] | None,
            "error": ""
        }

        try:
            # 1. Проверка существования папки
            if not os.path.isdir(db_folder):
                raise FileNotFoundError(f"Папка {db_folder} не существует")

            # 2. Проверка файлов FAISS
            required_files = ["index.faiss", "index.pkl"]
            missing = [f for f in required_files if not os.path.exists(os.path.join(db_folder, f))]
            if missing:
                raise FileNotFoundError(f"Отсутствуют файлы: {missing}")

            # Проверка эмбеддингов
            if self.embeddings is None: raise EmbeddingsNotInitialized()

            # 3. Основная загрузка
            result["db"] = FAISS.load_local(
                db_folder,
                embeddings=self.embeddings,  # Используем глобальную модель эмбеддингов
                allow_dangerous_deserialization=True
            )

            result["success"] = True
            if verbose:
                print(f"_single_faiss_loader: {db_folder}")
                pprint(result)
            return result

        except Exception as e:
            result["error"] = str(e)
            if verbose:
                print(f"Ошибка _single_faiss_loader: {db_folder}\n{result['error']}")
            return result


#============================================================
# Объединение баз

    def merge_databases(self, input_folders: List[str], output_folder: str) -> tuple:
        """
        Объединяет несколько FAISS-баз с проверкой совместимости
        Возвращает (success: bool, message: str)
        """
        try:
            # 1. Проверка минимального количества баз
            if len(input_folders) < 2: return False, "Необходимо минимум 2 базы для объединения"

            # 2. Загрузка и проверка метаданных
            result, main_meta = self._load_metadata(input_folders[0])
            if not main_meta: return False, result

            # 3. Проверка совместимости всех баз
            for folder in input_folders[1:]:
                result, current_meta = self._load_metadata(folder)
                if not self._check_compatibility(main_meta, current_meta):
                    msg = f"Несовместимые базы:\n{main_meta['embedding_model']}\nи\n{current_meta['embedding_model']}"
                    return False, msg

            # 4. Загрузка эмбеддингов
            err_code, embeddings = self._load_embeddings(main_meta)
            if not embeddings: return False, err_code

            # 5. Объединение баз
            merged_db = self._merge_faiss_indexes(input_folders, embeddings)

            # 6. Сохранение результата
            merged_db.save_local(output_folder)
            self._save_merged_metadata(output_folder, main_meta)

            return True, f"Базы успешно объединены в {output_folder}"

        except Exception as e:
            return False, f"Критическая ошибка: {str(e)}"

    def safe_hybrid_merge(self, input_folders: List[str], output_folder: str) -> tuple:
        """Объединение только если есть минимум 2 базы каждого типа"""
        txt_msg = ""
        tab_msg = ""
        result = ""

        try:
            os.makedirs(os.path.join(output_folder, "text_db"), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "table_db"), exist_ok=True)

            # Фильтрация текстовых баз
            text_folders = [
                os.path.join(f, "text_db") for f in input_folders
                if os.path.exists(os.path.join(f, "text_db"))
                   and os.listdir(os.path.join(f, "text_db"))
            ]

            # Объединение текстов (только если >1 базы)
            if len(text_folders) > 1:
                success, result = self.merge_databases(text_folders, os.path.join(output_folder, "text_db"))
                if not success:
                    return False, f"Ошибка объединения текстов: {result}"
            elif text_folders:  # Если ровно 1 база - просто копируем
                shutil.copytree(text_folders[0], os.path.join(output_folder, "text_db"), dirs_exist_ok=True)
                txt_msg = "Скопирована единственная текстовая база"
            else:
                txt_msg = "Нет текстовых баз для объединения"

            # Фильтрация табличных баз
            table_folders = [
                os.path.join(f, "table_db") for f in input_folders
                if os.path.exists(os.path.join(f, "table_db"))
                   and os.listdir(os.path.join(f, "table_db"))
            ]

            # Объединение таблиц (только если >1 базы)
            if len(table_folders) > 1:
                success, result = self.merge_databases(table_folders, os.path.join(output_folder, "table_db"))
                if not success:
                    return False, f"Ошибка объединения таблиц: {result}"
            elif table_folders:  # Если 1 база - копируем
                shutil.copytree(table_folders[0], os.path.join(output_folder, "table_db"), dirs_exist_ok=True)
                tab_msg = "Скопирована единственная табличная база"
            else:
                tab_msg = "Нет табличных баз для объединения"

            return True, f"{result}. {txt_msg+'.'} {tab_msg+'.'}"

        except Exception as e:
            return False, f"Критическая ошибка: {str(e)}"

    @staticmethod
    def _load_metadata(folder: str) -> Any | None:
        """Загружает метаданные из папки с базой"""
        meta_path = os.path.join(folder, "metadata.json")
        try:
            with open(meta_path, "r") as f:
                return "Успешно", json.load(f)
        except FileNotFoundError:
            return f"Файл метаданных не найден: {meta_path}", None
        except json.JSONDecodeError as e:
            return f"Ошибка формата JSON: {e}", None
        except PermissionError:
            return f"Нет прав на чтение файла: {meta_path}", None
        except Exception as e:
            return f"Неизвестная ошибка: {e}", None

    def metadata_loader(self, folder):
        return self._load_metadata(folder)

    @staticmethod
    def _check_compatibility(meta1: dict, meta2: dict) -> bool:
        """Проверяет совместимость метаданных двух баз"""
        required_keys = [
            'embedding_model',
            'model_type',
            'normalized',
            'distance_strategy',
            'is_e5_model'
        ]

        for key in required_keys:
            if meta1.get(key) != meta2.get(key): return False
        return True

    def _load_embeddings(self, metadata: dict) -> (tuple[str, None] | tuple[str, HuggingFaceEmbeddings] |
                                                   tuple[str, OpenAIEmbeddings]):
        """Инициализирует модель эмбеддингов на основе метаданных"""
        try:
            model_type = metadata['model_type']
            model_name = metadata['embedding_model']
            if metadata['model_type'] == "openai":
                return "Успешно", OpenAIEmbeddings(
                    model=model_name,
                    api_key=self.api_key,
                    base_url=self.api_url
                )

            elif metadata['model_type'] == "huggingface":
                return "Успешно", HuggingFaceEmbeddings(
                    model_name=model_name,
                    encode_kwargs={'normalize_embeddings': metadata['normalized']}
                )
            else:
                raise ValueError(f"Неподдерживаемый тип модели: {model_type}. Доступные варианты: openai, huggingface")


        except ValueError as e: # Специфичная обработка ошибок валидации
            return f"Ошибка валидации: {str(e)}", None
        except KeyError as e:
            return f"Отсутствует обязательное поле в метаданных: {str(e)}", None
        except ImportError as e:
            return f"Ошибка импорта модуля: {str(e)}", None
        except Exception as e:
            return f"Неизвестная ошибка при загрузке эмбеддингов: {str(e)}", None

    @staticmethod
    def _merge_faiss_indexes(folders: List[str], embeddings: Embeddings) -> FAISS:
        """Объединяет FAISS-индексы"""
        merged_db = FAISS.load_local(folders[0], embeddings, allow_dangerous_deserialization=True)

        for folder in folders[1:]:
            current_db = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
            merged_db.merge_from(current_db)

        return merged_db

    @staticmethod
    def _save_merged_metadata(output_folder: str, meta: dict):
        """Создает расширенные метаданные для объединенной базы"""
        merged_meta = {
            "embedding_model": meta["embedding_model"],
            "model_type": meta["model_type"],
            "dimension": meta["dimension"],
            "normalized": meta["normalized"],
            "distance_strategy": meta["distance_strategy"],
            "is_e5_model": meta["is_e5_model"]
        }

        with open(os.path.join(output_folder, "metadata.json"), "w") as f:
            json.dump(merged_meta, f, indent=2)

# ==================================================================================================
# Поиск
    @staticmethod
    def formatted_scored_sim_search_by_cos(index: Optional[FAISS], query: str, **search_args) -> list:
        """
        Cинхронный поиск на базе similarity_search_with_relevance_scores.
        :param index: FAISS-индекс из langchain
        :param query: Запрос (вектор)
        :param k:
        :return: список словарей с результатами поиска
        """
        k = search_args.pop("k", 4)
        kwargs = search_args.copy()
        # Стандартный поиск по совпадению на основе косинусных расстояний который возвращает
        results = index.similarity_search_with_relevance_scores(query, k=k, **kwargs)
        # Преобразуем результаты в требуемый формат
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "score": round(float(score), 6),
                "metadata": doc.metadata
            })
        return formatted_results

    # Синхронный поиск по максимальной предельной релевантности

    def formatted_scored_mmr_search_by_vector(self, index: Optional[FAISS], query: str, **search_args: Any) -> list:
        """
        Cинхронный поиск на базе max_marginal_relevance_search_with_score_by_vector.
        :param index: FAISS-индекс из langchain
        :param query: Запрос (вектор)
        :param k:
        :return: список словарей с результатами поиска
        """
        if index is None: return []

        # Получение эмбеддинга запроса
        query_embedding = self.embeddings.embed_query(query)
        # MMR поиск с исходными оценками
        results = index.max_marginal_relevance_search_by_vector(
            query_embedding,
            **search_args
        )

        # Нормализация оценок из [-1, 1] в [0, 1]
        formatted_results = []
        for doc in results:
            doc_embedding = self.embeddings.embed_query(doc.page_content)
            cosine_sim = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            formatted_results.append({
                "content": doc.page_content,
                "score":  cosine_sim,
                "metadata": doc.metadata
            })

        return formatted_results

    # --------------------------------------------------
    # Асинхронный поиск

    @async_wrapper
    def aformatted_scored_sim_search_by_cos(self, index: Optional[FAISS], query: str, **search_args) -> list:
        """Преобразование метода в асинхронный"""
        return self.formatted_scored_sim_search_by_cos(index, query, **search_args)

    @async_wrapper
    def aformatted_scored_mmr_search_by_vector(self, index: Optional[FAISS], query: str, **search_args) -> list:
        return self.formatted_scored_mmr_search_by_vector(index, query, **search_args)

    async def multi_async_search(
            self,
            query: str,
            indexes: List[Optional[FAISS]],
            search_function: Callable,
            **search_args
    ) -> list:
        """
        Асинхронный поиск по нескольким индексам с одним запросом.
        :param query: Запрос (строка)
        :param indexes: Список FAISS-индексов
        :param search_function: Асинхронная функция поиска
        :return: список словарей с результатами поиска
        """
        tasks = [search_function(index, query, **search_args) for index in indexes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = []
        for res in results:
            if isinstance(res, BaseException):  # Учитываем все типы исключений
                print(f"⚠️ Ошибка в поиске: {str(res)}")
                continue
            if isinstance(res, list):  # Явная проверка типа
                valid_results.extend(res)
        return valid_results


    def _process_search_results(self,
                                text_results: List[LangDoc],
                                table_results: List[LangDoc],
                                db_result: dict,
                                k: int
    ) -> List[LangDoc]:
        # 2. Собираем все уникальные чанки (основные + связанные)
        all_chunks = {}
        title_map = {}  # Храним соответствие doc_id -> _title

        for chunk in text_results + table_results:
            doc_id = chunk.metadata["doc_id"]

            # Запоминаем название документа при первом встреченном чанке
            if doc_id not in title_map:
                title_map[doc_id] = chunk.metadata.get("_title", f"Документ {doc_id}")

            # Добавляем основной чанк
            if chunk.metadata["chunk_id"] not in all_chunks:
                all_chunks[chunk.metadata["chunk_id"]] = chunk

            # Добавляем связанные чанки
            for linked_id in chunk.metadata.get("linked", []):
                linked_chunk = self._get_chunk_by_id(linked_id, db_result)
                if linked_chunk and linked_chunk.metadata["chunk_id"] not in all_chunks:
                    all_chunks[linked_chunk.metadata["chunk_id"]] = linked_chunk

                    # Запоминаем название для связанных чанков
                    linked_doc_id = linked_chunk.metadata["doc_id"]
                    if linked_doc_id not in title_map:
                        title_map[linked_doc_id] = linked_chunk.metadata.get("_title", f"Документ {linked_doc_id}")

        # 3. Обогащаем "головные" чанки названиями документов
        final_results = []
        for chunk in all_chunks.values():
            # Создаем копию метаданных, чтобы не изменять оригинал
            enriched_metadata = chunk.metadata.copy()

            # Добавляем название только к "головным" чанкам (тем, которые найдены через MMR)
            if chunk.metadata["chunk_id"] in {c.metadata["chunk_id"] for c in text_results + table_results}:
                enriched_metadata["document_title"] = title_map.get(chunk.metadata["doc_id"], "")

            # Создаем новый документ с обогащенными метаданными
            final_results.append(LangDoc(
                page_content=chunk.page_content,
                metadata=enriched_metadata
            ))

        # 4. Сортируем по chunk_id для сохранения порядка документа
        return sorted(
            final_results,
            key=lambda x: x.metadata["chunk_id"]
        )[:k * 2]

    def _get_chunk_by_id(self, chunk_id: str, db_result: dict) -> Optional[LangDoc]:
        """Поиск чанка по ID с проверкой всех хранилищ"""
        for db_type in ["text_db", "table_db"]:
            if db_result.get(db_type):
                for doc in db_result[db_type].docstore._dict.values():
                    if doc.metadata["chunk_id"] == chunk_id:
                        return doc
        return None
#===================================================================================================
class Tester(DBConstructor):
    def __init__(self):
        super().__init__()

    def db_tester(self, db_markdown_text: list, system: str, user: str, verbose=False):

        questionnaire = []

        for chunk in db_markdown_text:
            if verbose:
                print(user + chunk.page_content)
                print('---------------------------------------------------------------------')
            request = f"{user}\n{chunk.page_content}"
            self.answer = self.request_to_openai(system, request, 0.5)
            if verbose:
                print(f"Вопросы от модели:\n{self.answer}")
                print('---------------------------------------------------------------------')
            questionnaire.append(self.answer)
            time.sleep(2.8)
        test_results = ''.join(questionnaire)
        return test_results

    def quest_handler(self, quest_file: str, system: str, user: str):
        # Метод для обработки пула тестовых вопросов по базе
        # и составления сводки по недостающей информации
        with open(quest_file, 'r') as qf:
            pull_questions = qf.read()
        # Делаю из пула вопросов langchain документ, чтобы подать в request_openai
        pull_questions = LangDoc(page_content=pull_questions, metadata={'pull': 'questions'})
        # Получаю сводку
        request = f"{user}\n{pull_questions}"
        self.summary = self.request_to_openai(system, request, 0)

        return self.summary

