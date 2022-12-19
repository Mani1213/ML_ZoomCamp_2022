FROM python:3.8-slim

WORKDIR /app

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

COPY ["predict.py", "house_price_model.bin", "./"]

RUN mkdir /app/templates
ADD templates /app/templates

RUN mkdir /app/static
ADD static /app/static

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:9696", "predict:app" ]

