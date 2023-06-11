# Use a base image with Python and necessary dependencies
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project directory into the container at /app
COPY . /app

# Install any dependencies required by your project
RUN pip install -r requirements.txt

# Specify the command to run your Python script
CMD ["python", "cryptogram_final.py"]
