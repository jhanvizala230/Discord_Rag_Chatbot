FROM python:3.11

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY backend_langchain/ ./backend_langchain/

EXPOSE 8000

CMD ["uvicorn", "backend_langchain.app:app", "--host", "0.0.0.0", "--port", "8000"]
