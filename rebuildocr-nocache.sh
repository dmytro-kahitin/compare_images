docker-compose stop imageocr
docker-compose build --build-arg INSTALL_CACHEBUST=$(date +%s) imageocr
docker-compose up -d imageocr