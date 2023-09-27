FROM python:3.9

WORKDIR /app

# Copy the current directory (on your machine) into the container at /app
COPY . /app

# Install any needed Python packages specified in requirements.txt
RUN pip install -r requirements.txt

EXPOSE $PORT 
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT heart:app 
