#specify the parent base image which is the anaconda3
FROM continuumio/anaconda3:latest

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
    && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/*

# copy requirements.txt
COPY ./requirements.txt /src/app/requirements.txt

# install dependencies
RUN pip install --no-cache-dir --upgrade pip

# set work directory
WORKDIR /src/app


# install project requirements
RUN pip install -r requirements.txt

# copy project
COPY . /src/app

# set work directory
WORKDIR /src/app

# set app port
EXPOSE 8080

# Run gait_api.py when the container launches
CMD [ "/src/app/gait_api.py","run","--host","0.0.0.0"]