# Use a lightweight Python 3 base image
FROM python:3.12-slim

# Install system dependencies required by GeoPandas and PROJ
RUN apt-get update && apt-get install -y \
    build-essential \
    libproj-dev \
    proj-data \
    proj-bin \
    gdal-bin

# Install Python libraries
RUN pip install geopandas libpysal pandas numpy scipy

# Set the working directory in the container
WORKDIR /app

# Copy all files from your project folder to the container
COPY . .

# Run your script when the container starts
CMD ["python", "BiT_updated.py"]
