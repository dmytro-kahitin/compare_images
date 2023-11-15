FROM continuumio/miniconda3

# OS dependencies
RUN apt-get update && apt-get install -y ffmpeg libgl1 libsm6 libxext6

WORKDIR /ocr

# Create environment
RUN conda create -n compare_images python=3.10

# Activate environment
SHELL ["conda", "run", "-n", "compare_images", "/bin/bash", "-c"]
ARG INSTALL_CACHEBUST=1
# Install Conda packages from various channels
RUN conda install -c conda-forge pika -y
RUN conda install -c conda-forge python-dotenv -y
RUN conda install -c anaconda pymongo -y
RUN conda install -c anaconda scikit-learn -y
RUN pip install xxhash
RUN pip install imagehash
RUN pip install PyMuPDF==1.20.2
RUN pip install paddleocr
RUN pip install paddlepaddle

# Copy code and fix libGL
# The code to run when container is started:
ARG SRC_CACHEBUST=1

COPY ./ .
RUN ln -s /usr/lib/x86_64-linux-gnu/mesa/libGL.so.1 /usr/lib/libGL.so.1

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "compare_images", "python", "main.py"]