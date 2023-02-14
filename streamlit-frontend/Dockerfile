
FROM python:3.9-slim


COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "Streamlit_dashboard.py"]


