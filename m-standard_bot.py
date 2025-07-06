import os
import asyncio
import traceback
from html import escape
from pprint import pprint
import yaml

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.enums import ParseMode
from rag_processor import *
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from pydantic import BaseModel, Field, ValidationError

import csv
import os
from datetime import datetime
from github import Github, GithubException

class GCProcessor(RAG):
    def __init__(self, gigachat_model: str = "GigaChat"):
        super().__init__()
        self.api_key = os.environ.get("GIGACHAT_API_KEY", None)
        self.scope = os.getenv("GIGACHAT_SCOPE")
        self.giga_chat = GigaChat(credentials=self.api_key, scope=self.scope, verify_ssl_certs=False)
        self.user = MessagesRole.USER
        self.system = MessagesRole.SYSTEM
        self.gigachat_model = gigachat_model

    @property
    def gigachat_model(self):
        return self._gigachat_model

    @gigachat_model.setter
    def gigachat_model(self, value):
        self._gigachat_model = value

    def get_answer(self, user: str, system_prompt: str = "", temperature: float = 0.0):
        """Формирует запрос к GigaChat и возвращает ответ."""
        messages = [
            Messages(role=self.system, content=system_prompt),
            Messages(role=self.user, content=user)
        ]

        # Генерируем ответ от GigaChat
        response = self.giga_chat.chat(Chat(messages=messages, temperature=temperature, model=self.gigachat_model))
        return response.choices[0].message.content

class Config:
    os.environ.clear()
    load_dotenv(".venv/.env")
    FAISS_ROOT = os.path.join(os.getcwd(), "DB_FAISS")
    BOT_TOKEN = os.getenv("BOT_TOKEN")

    GENERATION_K = 4  # Новый параметр для генерации
    TEXT_K = 3
    TABLE_K = 3

# Валидация структуры файла
class PromptsSchema(BaseModel):
    system_prompt: str
    user_template: str
    generation_settings: dict = Field(
        default={"temperature": 0.0, "model_name": "GigaChat"},
        description="Настройки генерации ответов"
    )

class PromptManager:
    def __init__(self, file_path: str = "prompts.yaml"):
        self.file_path = file_path  # Храним как строку
        self._load_prompts()
        self.last_modified = 0

    def _load_prompts(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        try:
            # Валидируем структуру файла
            validated = PromptsSchema(**data)
            self.system = validated.system_prompt
            self.user_template = validated.user_template
            self.temperature = validated.generation_settings.get("temperature", 0.0)
            self.model_name = validated.generation_settings.get("model_name", "GigaChat")
            self.last_modified = os.path.getmtime(self.file_path)

        except Exception as e:
            print(f"⚠️ Ошибка загрузки промптов: {e}")
            # Значения по умолчанию
            self.system = "Ты - большая языковая модель. Твоя задача - отвечать на вопросы пользователей, опираясь на предоставленные документы"
            self.user_template = "Пользователь задал вопрос: {question}. Ответь на него, пользуясь следующими данными: {doci}"
            self.temperature = 0.0
            self.model_name = "GigaChat"

    def get_prompts(self):
        # Проверяем обновление файла
        current_modified = os.path.getmtime(self.file_path)  # Проверяет дату изменения
        if current_modified > self.last_modified:  # Если обнаружено изменение
            self._load_prompts()  # Перезагружает промпты, температуру, модель_name и другие параметры
        return { # Возвращает словарь с параметрами
            "system": self.system,
            "user_template": self.user_template,
            "temperature": self.temperature,
            "model_name": self.model_name
        }

class QueryLogger:
    CSV_DELIMITER = "|"
    def __init__(self, log_file="query_logs.csv", github_token=None, github_repo=None, branch="bot-logs"):
        self.log_file = log_file
        self.github_token = github_token
        self.github_repo = github_repo  # Формат: "username/repo-name"
        self.branch = branch
        self._init_log_file()

    def _init_log_file(self):
        """Создает файл логов с заголовками, если он не существует"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._get_fieldnames())
                writer.writeheader()

    @staticmethod
    def _get_fieldnames():
        return [
            "timestamp",
            "user_id",
            "question",
            "category",
            "document_titles",
            "document_contents",
            "document_scores",
            "generated_answer",
            "user_rating"
        ]

    def log_query(self, **kwargs):
        """Логирует запрос в CSV и отправляет в GitLab (если настроено)"""
        try:
            # Формируем запись
            record = {
                "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                "user_id": kwargs.get("user_id", ""),
                "question": kwargs.get("question", ""),
                "category": kwargs.get("category", ""),
                "document_titles": "\n-----------  \n".join(kwargs.get("document_titles", [])),
                "document_contents": "\n-----------  \n".join(kwargs.get("document_contents", [])),
                "document_scores": "\n-------  \n".join(map(str, kwargs.get("document_scores", []))),
                "generated_answer": kwargs.get("generated_answer", ""),
                "user_rating": kwargs.get("user_rating", "")
            }

            # Локальное логирование
            with open(self.log_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._get_fieldnames(), delimiter=self.CSV_DELIMITER)
                writer.writerow(record)

                # Отправка в GitHub (если настроено)
                if self.github_token and self.github_repo:
                    self._push_to_github(record)

        except Exception as e:
            print(f"⚠️ Ошибка логирования: {e}")

    def _push_to_github(self, record):
        """Отправка лога в репозиторий GitHub"""
        try:
            # 1. Инициализация клиента GitHub
            g = Github(self.github_token)
            repo = g.get_repo(self.github_repo)

            # 2. Формируем CSV строку
            csv_row = ",".join(f'"{value}"' for value in [
                record["timestamp"],
                record["user_id"],
                record["question"],
                record["category"],
                record["document_titles"],
                record["document_contents"],
                record["document_scores"],
                record["generated_answer"],
                record["user_rating"]
            ]) + "\n"

            # 3. Получаем текущее содержимое файла (если существует)
            try:
                file_contents = repo.get_contents(self.log_file, ref=self.branch)
                current_content = file_contents.decoded_content.decode("utf-8")
                new_content = current_content + csv_row
                update = True
            except Exception:  # Файл не существует
                current_content = ""
                new_content = ",".join(self._get_fieldnames()) + "\n" + csv_row
                update = False

            # 4. Обновляем или создаем файл
            commit_message = "Bot log update"
            if update:
                repo.update_file(
                    path=self.log_file,
                    message=commit_message,
                    content=new_content,
                    sha=file_contents.sha,
                    branch=self.branch
                )
            else:
                repo.create_file(
                    path=self.log_file,
                    message="Initial bot logs",
                    content=new_content,
                    branch=self.branch
                )
            print("✅ Логи успешно обновлены в GitHub")

        except GithubException as e:
            print(f"⚠️ GitHub API error: {e.data.get('message', str(e))}")
        except Exception as e:
            print(f"⚠️ Unexpected error: {str(e)}")
            traceback.print_exc()

bot = Bot(token=Config.BOT_TOKEN)
dp = Dispatcher()
processor = DBConstructor()
user_sessions = {}
prompt_manager = PromptManager()  # Читает prompts.yaml в первый раз
answer_generator = GCProcessor(prompt_manager.get_prompts()["model_name"])  # Берёт модель из файла
logger = QueryLogger(
    log_file="query_logs_lite-2_t-03_ver-03_mmr_tx-3_tb-3.csv",
    github_token=os.getenv("GITHUB_TOKEN"),  # Добавить в .env
    github_repo="vlad-alaukhov/MStandard",  # Ваш репозиторий
    branch="bot-logs"  # Существующая ветка
)
filters = [
    {"element_type": "text", "_search_params": {"k": Config.TEXT_K, "fetch_k": (Config.TEXT_K * 10)//2, "lambda_mult": 0.6}},
    {"element_type": "table", "_search_params": {"k": Config.TABLE_K, "fetch_k": (Config.TABLE_K * 10)//2, "lambda_mult": 0.4}}
]

# ====================== Инициализация ======================
async def on_startup(bot: Bot):
    print("🔄 Запуск инициализации эмбеддингов...")
    print(Config.FAISS_ROOT)

    try:
        set_embs_result = processor.set_embeddings(Config.FAISS_ROOT, verbose=False)
        processor.db_metadata = set_embs_result["result"]["metadata"]
        pprint(processor.db_metadata)

        if not set_embs_result["success"]:
            error_msg = set_embs_result.get("result", {}).get("Error", "Неизвестная ошибка")
            print(f"❌ Ошибка инициализации: {error_msg}")
            return

        print("✅ Эмбеддинги успешно загружены")


    except Exception as e:
        print(f"💥 Критическая ошибка при запуске: {str(e)}")
        raise
# ====================== Команды ===========================
# --------------------- Команда /start ---------------------
@dp.message(Command("start"))
async def start(message: types.Message):
    try:
        categories = [
            d for d in os.listdir(Config.FAISS_ROOT)
            if os.path.isdir(os.path.join(Config.FAISS_ROOT, d))
        ]

        if not categories:
            await message.answer("⚠️ Базы данных не найдены!")
            return

        keyboard = types.InlineKeyboardMarkup(
            inline_keyboard=[
                [types.InlineKeyboardButton(
                    text=category,
                    callback_data=f"category_{category}"
                )]
                for category in categories
            ]
        )

        # Загрузка руководства
        with open("guide.yaml", "r", encoding="utf-8") as f:
            guide = yaml.safe_load(f)

        await message.answer(
            guide["brief"],
            parse_mode=ParseMode.MARKDOWN
        )

        await message.answer("📂 Выберите категорию документов:", reply_markup=keyboard)

    except Exception as e:
        await message.answer("⚠️ Ошибка при загрузке категорий")
        print(f"❗ Ошибка в /start: {e}")

# -------------- Команда /getsystem Запрос системного промпта --------------
# Добавляем в существующий Dispatcher
@dp.message(Command("getsystem"))
async def cmd_get_system(message: types.Message):
    prompts = prompt_manager.get_prompts()
    await message.answer(
        "📝 <b>Текущий системный промпт:</b>\n\n"
        f"<code>{escape(prompts['system'])}</code>",
        parse_mode=ParseMode.HTML
    )

# ---------------- Команда /getuserprompt Запрос user-промпта ----------------
@dp.message(Command("getuserprompt"))
async def cmd_get_user_prompt(message: types.Message):
    prompts = prompt_manager.get_prompts()
    await message.answer(
        "📋 <b>Текущий user template:</b>\n\n"
        f"<code>{escape(prompts['user_template'])}</code>",
        parse_mode=ParseMode.HTML
    )

# -------------- Команда /getsettings Запрос системных установок --------------
@dp.message(Command("getsettings"))
async def cmd_get_settings(message: types.Message):
    prompts = prompt_manager.get_prompts()
    await message.answer(
        "⚙️ <b>Текущие настройки генерации:</b>\n\n"
        f"🧠 Модель: <code>{escape(answer_generator.gigachat_model)}</code>\n"
        f"🌡 Температура: <code>{prompts['temperature']}</code>",
        parse_mode=ParseMode.HTML
    )
# ------------------------------- Команда /help -------------------------------
@dp.message(Command("help"))
async def help_command(message: types.Message):
    try:
        # Загрузка руководства
        with open("guide.yaml", "r", encoding="utf-8") as f:
            guide = yaml.safe_load(f)
        await message.answer(
            guide["full_guide"],
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True
        )
    except Exception as e:
        await message.answer("⚠️ Руководство временно недоступно")
        print(f"Ошибка загрузки руководства: {str(e)}")
# ================================ Логика бота ================================
# ----------------------- Обработка категории документов ----------------------
@dp.callback_query(F.data.startswith("category_"))
async def handle_category(callback: types.CallbackQuery):
    try:
        user_id = callback.from_user.id
        category = callback.data.split("_", 1)[1]  # Исправлено разделение

        # Инициализация сессии
        user_sessions[user_id] = {
            "faiss_indexes": [],  # Будет заполнено
            "query_prefix": "",
            "last_results": [],  # Важно: создаем ключ заранее
            "current_category": ""
        }

        # Убедимся, что путь существует
        category_path = os.path.join(Config.FAISS_ROOT, category)
        if not os.path.exists(category_path):
            await callback.answer("❌ Категория не найдена", show_alert=True)
            return

        # Показываем статус "Загрузка..."
        await callback.answer("⏳ Загрузка...")

        # Асинхронная загрузка баз
        faiss_indexes = []
        faiss_paths = [d for d, _, files in os.walk(category_path) for file in files if file.endswith(".faiss")]

        # Прогресс-бар
        progress_msg = await callback.message.answer("🔄 Прогресс: 0%")

        for idx, faiss_dir in enumerate(faiss_paths):
            # Загрузка в отдельном потоке
            load_result = await asyncio.to_thread(
                processor.faiss_loader,
                faiss_dir,
                hybrid_mode=False
            )

            if load_result["success"]:
                faiss_indexes.append(load_result["db"])

            # Обновление прогресса
            progress = (idx + 1) / len(faiss_paths) * 100
            await progress_msg.edit_text(f"🔄 Прогресс: {int(progress)}%")

        # # Сохраняем результат
        user_sessions[user_id].update({
            "faiss_indexes": faiss_indexes,
            "query_prefix": "query: " if processor.db_metadata.get("is_e5_model", False) else "",
            "current_category": category
        })

        # Удаляем сообщения
        await progress_msg.delete()
        await callback.message.answer(f"✅ База '{category}' готова к поиску!")

    except Exception as e:
        await callback.answer("⚠️ Ошибка загрузки", show_alert=True)
        print(f"❗ Ошибка: {str(e)}")
        traceback.print_exc()
# --------------------- Обработка запроса ---------------------
@dp.message(F.text)
async def handle_query(message: types.Message):
    try:
        user_id = message.from_user.id
        if user_id not in user_sessions:
            await message.answer("❌ Сначала выберите категорию через /start")
            return

        # Отправляем индикатор поиска
        search_msg = await message.answer("⏳ Ищу ответ в документах...")

        # Получаем контекст пользователя
        session = user_sessions[user_id]

        print(session["query_prefix"] + message.text)

        # Выполняем поиск
        raw_results = await layered_search(
            query=session["query_prefix"] + message.text,
            indexes=session["faiss_indexes"],
            search_function=processor.aformatted_scored_mrr_search_with_cosine_sorting
        )

        pprint(raw_results)

        # Сортировка и фильтрация найденных чанков
        sorted_results = sorted(
            raw_results,
            key=lambda x: x["score"],
            reverse=True
        )[:Config.GENERATION_K]

        raw_articles = []
        # Собираем полные статьи для всех результатов
        session["articles"] = []
        for result in sorted_results:
            full_content = await assemble_full_content(
                main_chunk=result,
                faiss_indexes=session["faiss_indexes"]
            )
            raw_articles.append({
                "doc_id": result["metadata"]["doc_id"],
                "title": result["metadata"].get("_title", "Без названия"),
                "content": full_content,
                "score": result["score"],
                "element_type": result["metadata"].get("element_type", "text")
            })

        seen = set()

        for article in raw_articles:
            # Создаем кортеж из идентифицирующих полей
            identifier = (
                article["doc_id"],
                article["title"],
                article["content"]  # Если контент одинаковый - это дубликат
            )

            # Если статья уникальна - добавляем
            if identifier not in seen:
                seen.add(identifier)
                session["articles"].append({
                    "title": article["title"],
                    "content": article["content"],
                    "score": article["score"],
                    "element_type": article["element_type"]
                })

        # Отправляем индикатор поиска
        await search_msg.edit_text("⏳ Готовлю ответ...")

        # Формируем промпт для модели
        user_prompt = "\n\n".join(
            f"Статья {i + 1} ({art['score']:.0%}): {art['title']}\n{art['content']}..."
            for i, art in enumerate(session["articles"])
        )

        prompts = prompt_manager.get_prompts()

        if answer_generator.gigachat_model != prompts["model_name"]:
            answer_generator.gigachat_model = prompts["model_name"]  # Просто обновляем имя модели

        # Генерируем ответ с помощью GigaChat
        answer = answer_generator.get_answer(
            user=prompts["user_template"].format(question=message.text, doci=user_prompt),
            system_prompt=prompts["system"],
            temperature=prompts["temperature"]
        )

        # Удаляем индикатор поиска.
        await search_msg.delete()

        # Сохраняем данные для последующего логирования в сессии
        session["last_log_data"] = {
            "user_id": user_id,
            "question": message.text,
            "category": session["current_category"],
            "document_titles": [art['title'] for art in session["articles"]],
            "document_contents": [art['content'][:500] + "..." for art in session["articles"]],
            "document_scores": [art['score'] for art in session["articles"]],
            "generated_answer": answer
        }

        # Создаем кнопки только для источников
        builder = InlineKeyboardBuilder()
        for idx, art in enumerate(session["articles"]):
            builder.button(
                text=f"{art['title']} ({art['score']:.0%})",
                callback_data=f"show_article_{idx}"
            )
        builder.adjust(1)  # Каждый источник на новой строке

        # Форматируем и отправляем ответ
        response = f"🔍 {answer}\n\n" "📚 Использованные источники:\n"
        await message.answer(
            response,
            reply_markup=builder.as_markup(),
            parse_mode=ParseMode.MARKDOWN
        )

        # Отправляем отдельное сообщение с запросом оценки
        rate_builder = InlineKeyboardBuilder()
        for i in range(1, 6):
            rate_builder.button(text=str(i), callback_data=f"rate_{i}")
        rate_builder.adjust(5)  # Все кнопки в один ряд

        # Сохраняем ID сообщения с оценкой для возможного удаления
        rate_message = await message.answer(
            "📊 Пожалуйста, оцените качество ответа (1 - 5):",
            reply_markup=rate_builder.as_markup()
        )

        # Пытаемся закрепить сообщение
        try:
            await bot.pin_chat_message(
                chat_id=message.chat.id,
                message_id=rate_message.message_id,
                disable_notification=True
            )
            session["is_pinned"] = True  # Флаг, что сообщение закреплено
        except Exception as e:
            print(f"⚠️ Не удалось закрепить сообщение: {e}")
            session["is_pinned"] = False

        # Сохраняем ID сообщения с оценкой
        session["rate_message_id"] = rate_message.message_id

    except Exception as e:
        await message.answer(f"⚠️ Ошибка при обработке запроса: {str(e)}")
        print(f"ERROR: {str(e)}")
        traceback.print_exc()

async def layered_search(query: str, indexes: List[Optional[FAISS]], search_function: Callable):
    all_results = []
    global filters
    Config.TEXT_K = 6

    for filter_config in filters:
        search_args = {
            "filter": {k: v for k, v in filter_config.items() if not k.startswith('_')},
            **filter_config.get("_search_params", {})
        }

    chunk_results = await processor.multi_async_search(
        query=query,
        indexes=indexes,
        search_function=search_function,
        **search_args
    )
    all_results.extend(chunk_results)

    return all_results

# Обработчик оценки пользователя
@dp.callback_query(F.data.startswith("rate_"))
async def handle_rating(callback: types.CallbackQuery):
    try:
        user_id = callback.from_user.id
        session = user_sessions.get(user_id)

        if not session or "last_log_data" not in session:
            await callback.answer("❌ Сессия устарела. Выполните новый поиск.")
            return

        # Получаем оценку (1-5)
        rating = int(callback.data.split("_")[1])

        # Добавляем оценку в данные для логирования
        log_data = session["last_log_data"]
        log_data["user_rating"] = rating

        # Логируем запрос с оценкой
        logger.log_query(**log_data)

        # Удаляем данные из сессии
        del session["last_log_data"]

        # Открепляем сообщение с оценкой (если оно было закреплено)
        if "is_pinned" in session and session["is_pinned"]:
            try:
                await bot.unpin_chat_message(
                    chat_id=callback.message.chat.id
                )
            except Exception as e:
                print(f"⚠️ Ошибка при откреплении сообщения: {e}")

        # Удаляем сообщение с кнопками оценки
        if "rate_message_id" in session:
            try:
                await bot.delete_message(
                    chat_id=callback.message.chat.id,
                    message_id=session["rate_message_id"]
                )
            except Exception as e:
                print(f"⚠️ Ошибка при удалении сообщения с оценкой: {e}")
            finally:
                # Очищаем данные о сообщении в сессии
                del session["rate_message_id"]
                if "is_pinned" in session:
                    del session["is_pinned"]

        # Отправляем подтверждение пользователю
        await callback.answer(f"✅ Спасибо за вашу оценку: {rating}!")

        # Дублируем благодарность в чат
        await callback.message.answer(f"🌟 Ваша оценка {rating} принята! Спасибо за обратную связь!")

    except Exception as e:
        await callback.answer("⚠️ Ошибка при обработке оценки")
        print(f"RATING ERROR: {str(e)}")
        traceback.print_exc()

# Обработчик выбора статьи
@dp.callback_query(F.data.startswith("show_article_"))
async def handle_article_selection(callback: types.CallbackQuery):
    try:
        user_id = callback.from_user.id
        session = user_sessions.get(user_id)

        if not session or "articles" not in session:
            await callback.answer("❌ Сессия устарела. Выполните новый поиск.")
            return

        article_idx = int(callback.data.split("_")[-1])
        article = session["articles"][article_idx]

        # Форматируем заголовок
        header = (
            f"📄 Документ: {article['title']}\n"
            f"🔗 Тип: {'таблица' if article['element_type'] == 'table' else 'текст'}\n"
            f"📏 Точность соответствия: {article['score']:.0%}\n\n"
        )

        # Отправляем содержимое с разбивкой
        await callback.message.answer(header)
        await send_long_message(callback.message, article["content"])
        await callback.answer()

    except Exception as e:
        await callback.answer("⚠️ Ошибка при загрузке статьи")
        print(f"CALLBACK ERROR: {str(e)}")
        traceback.print_exc()

async def assemble_full_content(main_chunk: dict, faiss_indexes: list) -> str:
    """Сборка полного контента из связанных чанков"""
    chunks = []
    visited = set()
    queue = [main_chunk["metadata"]["chunk_id"]]

    while queue:
        chunk_id = queue.pop(0)
        if chunk_id in visited:
            continue

        # Поиск чанка во всех индексах
        chunk = None
        for index in faiss_indexes:
            chunk = next(
                (doc for doc in index.docstore._dict.values()
                 if doc.metadata["chunk_id"] == chunk_id),
                None
            )
            if chunk:
                break

        if chunk:
            chunks.append(chunk)
            visited.add(chunk_id)
            queue.extend(
                linked_id
                for linked_id in chunk.metadata.get("linked", [])
                if linked_id not in visited
            )

    # Сортировка по порядку chunk_id (пример: doc1_p1, doc1_p2)
    chunks.sort(key=lambda x: x.metadata["chunk_id"])

    # Сборка контента
    return "\n\n".join(
        chunk.page_content.replace("passage:", "").strip()
        for chunk in chunks
    )

def format_response(main_chunk: dict, content: str) -> str:
    """Форматирование в зависимости от типа"""
    header = f"📄 Документ: {main_chunk['metadata'].get('_title', 'Без названия')}\n"
    element_type = main_chunk["metadata"].get("element_type", "text")

    if element_type == "table":
        return f"{header}📊 Таблица:\n{content}"

    if len(content) > 4000:
        content = content[:3900] + "\n[...текст сокращен...]"

    return f"{header}{content}"


# Функция для отправки длинных сообщений (адаптированная)
async def send_long_message(
        message: types.Message,
        text: str,
        max_length: int = 4000,
        delimiter: str = "\n\n"
) -> None:
    """Умная разбивка текста с сохранением структуры"""
    parts = []
    current_part = []
    current_len = 0

    for paragraph in text.split(delimiter):
        para_len = len(paragraph)

        if current_len + para_len > max_length:
            if current_part:
                parts.append(delimiter.join(current_part))
                current_part = []
                current_len = 0

            # Обработка очень длинных абзацев
            while para_len > max_length:
                parts.append(paragraph[:max_length])
                paragraph = paragraph[max_length:]
                para_len = len(paragraph)

        if para_len > 0:
            current_part.append(paragraph)
            current_len += para_len + len(delimiter)

    if current_part:
        parts.append(delimiter.join(current_part))

    # Отправка с нумерацией
    total = len(parts)
    for i, part in enumerate(parts, 1):
        header = f"📖 Часть {i}/{total}\n\n" if total > 1 else ""
        try:
            await message.answer(
                f"{header}{part}",
                parse_mode=ParseMode.MARKDOWN_V2 if "|" in part else None
            )
        except Exception as e:
            print(f"Ошибка отправки части {i}: {str(e)}")

# --------------------- Запуск
if __name__ == "__main__":
    dp.startup.register(on_startup)  # Явная регистрация обработчика

    print("=== Старт бота ===")
    print(f"🔑 Токен бота: {'установлен' if Config.BOT_TOKEN else 'отсутствует!'}")
    print(f"📁 Путь к базам: {Config.FAISS_ROOT}")

    try:
        asyncio.run(dp.start_polling(
            bot,
            skip_updates=True,
            allowed_updates=dp.resolve_used_update_types()
        ))
    except KeyboardInterrupt:
        print("\n🛑 Бот остановлен пользователем")
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {str(e)}")