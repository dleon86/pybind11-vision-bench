# Use a slim Python image with build tools
FROM python:3.10-slim

# Install core build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install pybind11, scikit-build, and poetry for pip builds
RUN pip install --no-cache-dir pybind11 scikit-build poetry

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies using Poetry, without creating a venv
# Update lock file first, then install
RUN poetry config virtualenvs.create false && \
    poetry lock --no-interaction --no-ansi && \
    poetry install --no-interaction --no-ansi

# Default to bash shell
CMD ["/bin/bash"]
