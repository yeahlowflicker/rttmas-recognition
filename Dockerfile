# Use the Alpine version of Python 3.10 as the base image
FROM python:3.10-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY ./src .

# Command to run your Python application
# Replace 'your_script.py' with the name of your main Python file
CMD ["python", "app.py"]
