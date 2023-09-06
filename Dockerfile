# base image
FROM python:3.10

# making directory of app
WORKDIR /appdemov2

# copy secrets.toml
COPY secrets.toml /root/.streamlit/secrets.toml

# copy of requirements file
COPY requirements.txt ./requirements.txt

# install pacakges
RUN pip3 install -r requirements.txt

# copying all files over
COPY . .

# exposing default port for streamlit
EXPOSE 8501

# command to launch app
CMD streamlit run ./Chatbot.py
