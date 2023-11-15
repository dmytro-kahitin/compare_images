# Compare Images

This project utilizes RabbitMQ for task queuing and MongoDB for data storage to perform Optical Character Recognition (OCR) on images and text comparison. It uses the PaddleOCR library for OCR and the Sklearn library for text comparison.

## Features

 - Perform OCR on images.
 - Compare images using hash algorithms.
 - Compare texts using Bag-of-Words and TF-IDF models.
 - Task queuing using RabbitMQ.
 - Store OCR results in MongoDB.

## Installation

This project requires Conda (Anaconda or Miniconda). Clone this repository and create a new Conda environment:

For Windows:
```
git clone https://github.com/dmytro-kahitin/compare_images.git
cd compare_images
conda create -n compare_images python=3.10
```

For Linux:
```
sudo apt-get update && apt-get install -y ffmpeg libgl1 libsm6 libxext6 gcc git
git clone https://github.com/dmytro-kahitin/compare_images.git
cd compare_images
conda create -n compare_images python=3.10
```

Install all required packages:

```
conda install -c conda-forge pika -y
conda install -c conda-forge python-dotenv -y
conda install -c anaconda pymongo -y
conda install -c anaconda scikit-learn -y
pip install logging
pip install xxhash
pip install imagehash
pip install PyMuPDF==1.20.2
pip install paddleocr
pip install paddlepaddle
```

## Environment Variables

Create a .env file at the app/config and set the following environment variables or just copy example.env:

```
# DEBUG
ENABLE_MAINTENANCE_QUEUE=True
LOGGER_LEVEL=INFO # DEBUG INFO WARNING ERROR FATAL

# TEXT COMPARATOR
SIMILARITY_PERCENTAGE=60 # from 1 to 100
MIN_TEXT_LEN=200
ENABLE_PREPROCESS_TEXT=False

# SIMILARITY_PERCENT 
# Variables define the maximum similarity thresholds for various hash algorithms. Lower values indicate a higher degree of similarity between images.
AHASH_MAX_SIMILARITY_PERCENT = 4
DHASH_MAX_SIMILARITY_PERCENT = 8
WHASH_HAAR_MAX_SIMILARITY_PERCENT = 8
COLORHASH_MAX_SIMILARITY_PERCENT = 0

# RabbitMQ settings (Set these as per your RabbitMQ configuration)
RABBITMQ_HOST=
RABBITMQ_PORT=
RABBITMQ_USERNAME=
RABBITMQ_PASSWORD=
RABBITMQ_VHOST=
RABBITMQ_HEARTBEAT=
RABBITMQ_BLOCKED_CONNECTION_TIMEOUT=

# MongoDB settings (Set these as per your MongoDB configuration)
MONGODB_HOST=
MONGODB_PORT=
MONGODB_USERNAME=
MONGODB_PASSWORD=
MONGODB_DATABASE=
MONGODB_COLLECTION=
MONGODB_SIMILAR_IMAGES_COLLECTION=

```

## Usage

Activate the Conda environment:

```
conda activate compare_images
```

To start the project, run:

```
python main.py
```
