FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 80

COPY . /app

CMD ["uvicorn", "main:app",  "--host", "0.0.0.0", "--port", "80" , "--reload"]

