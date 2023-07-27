FROM ubuntu

# Set the working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3.11 python3-pip 

# Install FastAPI and other Python dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy all the files from the current directory to the container's /app folder
COPY . /app/

# Expose the port that FastAPI will run on
EXPOSE 8000

# Define the command to run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "127.0.0.0", "--port", "8000"]