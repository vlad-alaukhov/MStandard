# RAGProcessor

Модуль для работы с RAG-системами с поддержкой CPU и GPU.
**Версии зависимостей строго зафиксированы для гарантии стабильности.**

## Как установить пакет для CPU локально из requirements.txt

**Решение:**

1. **Активация виртуального окружения (рекомендуется)**
   `python -m venv .venv && source .venv/bin/activate`

2. **Удаление предыдущих версий (если нужно)**
   `pip uninstall rag-processor -y`

3. **Установка зависимостей**
   `pip install -r requirements.txt`

**Пример содержимого `requirements.txt` для CPU:**
```txt
--index-url https://download.pytorch.org/whl/cpu
torch==2.3.0+cpu
sentence-transformers==2.2.0
faiss-cpu==1.7.0
langchain-core==0.3.51
pymupdf==1.23.0
python-docx==1.1.2
opencv-python-headless==4.11.0.86

**Примечания:**

Если requirements.txt не содержит явных ссылок на CPU-версии, добавьте --index-url для PyTorch.

Для полной изоляции используйте новое виртуальное окружение.

После установки проверьте зависимости:
pip list | grep -E "torch|sentence-transformers|faiss"



