FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn requests pydantic huggingface_hub
COPY app.py /app/app.py
ENV STATE_DIR=/app/state
RUN mkdir -p /app/state
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
