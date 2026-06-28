# Use a slim version of Python to keep the image small
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script into the container
COPY matrix_singluarity_game.py .

# Command to run your script
CMD ["python", "matrix_singluarity_game.py"]