FROM python:3.8-slim-buster

COPY . .

RUN pip install -r requirements.txt

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]