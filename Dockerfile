FROM ultralytics/ultralytics:8.2.16-cpu

WORKDIR /app

RUN pip install fastapi uvicorn pydantic_settings loguru -i https://pypi.tuna.tsinghua.edu.cn/simple
ADD main.py /app/main.py
ADD yolov8n-doclaynet.pt /app/yolov8n-doclaynet.pt

CMD ["python", "main.py"]