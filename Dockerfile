FROM ultralytics/ultralytics:8.2.48-cpu

WORKDIR /app

RUN pip install fastapi uvicorn pydantic_settings loguru
ADD yolo-doclaynet.pt /app/yolo-doclaynet.pt
ADD main.py /app/main.py

CMD ["python", "main.py"]