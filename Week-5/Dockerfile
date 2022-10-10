FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile.lock","Pipfile","./"]

RUN pipenv install --system --deploy

COPY ["ques6.py","model1.bin","./"]

EXPOSE 9696

ENTRYPOINT ["python","ques6.py"]
