import os
import asyncio
import traceback
from html import escape
from pprint import pprint
import yaml

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.enums import ParseMode
from rag_processor import *
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from pydantic import BaseModel, Field, ValidationError

class GCProcessor(RAG):
    def __init__(self, gigachat_model: str = "GigaChat"):
        super().__init__()
        self.api_key = os.environ.get("GIGACHAT_API_KEY", None)
        self.giga_chat = GigaChat(credentials=self.api_key, verify_ssl_certs=False)
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
    DEFAULT_K = 4

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

# config = Config()
bot = Bot(token=Config.BOT_TOKEN)
dp = Dispatcher()
processor = DBConstructor()
user_sessions = {}
prompt_manager = PromptManager()  # Читает prompts.yaml в первый раз
answer_generator = GCProcessor(prompt_manager.get_prompts()["model_name"])  # Берёт модель из файла

# ====================== Инициализация ======================
async def on_startup(bot: Bot):
    print("🔄 Запуск инициализации эмбеддингов...")

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
            "last_results": []  # Важно: создаем ключ заранее
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

        print(faiss_paths)

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

        # Сохраняем результат
        user_sessions[user_id] = {
            "faiss_indexes": faiss_indexes,
            "query_prefix": "query: " if processor.db_metadata.get("is_e5_model", False) else ""
        }

        user_sessions[user_id].update({
            "faiss_indexes": faiss_indexes,
            "query_prefix": "query: " if processor.db_metadata.get("is_e5_model", False) else ""
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

        # Выполняем поиск
        raw_results = await processor.multi_async_search(
            query=session["query_prefix"] + message.text,
            indexes=session["faiss_indexes"],
            search_function=processor.aformatted_scored_sim_search_by_cos,
            k=Config.DEFAULT_K
        )

        # Сортировка и фильтрация найденных чанков
        sorted_results = sorted(
            raw_results,
            key=lambda x: x["score"],
            reverse=True
        )[:3]  # Топ-3 результата

        # Собираем полные статьи для всех результатов
        session["articles"] = []
        for result in sorted_results:
            full_content = await assemble_full_content(
                main_chunk=result,
                faiss_indexes=session["faiss_indexes"]
            )
            session["articles"].append({
                "title": result["metadata"].get("_title", "Без названия"),
                "content": full_content,
                "score": result["score"],
                "element_type": result["metadata"].get("element_type", "text")
            })

        # Формируем промпт для модели
        user_prompt = "\n\n".join(
            f"Статья {i + 1} ({art['score']:.0%}): {art['title']}\n{art['content'][:1500]}..."
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

        # Создаем кнопки для источников
        builder = InlineKeyboardBuilder()
        for idx, art in enumerate(session["articles"]):
            builder.button(
                text=f"{art['title']} ({art['score']:.0%})",
                callback_data=f"show_article_{idx}"
            )
        builder.adjust(1)

        # Форматируем и отправляем ответ
        response = f"🔍 {answer}\n\n" "📚 Использованные источники:\n"

        await message.answer(response, reply_markup=builder.as_markup(), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        await message.answer("⚠️ Ошибка при обработке запроса")
        print(f"ERROR: {str(e)}")
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