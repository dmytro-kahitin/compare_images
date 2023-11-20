docker compose stop imageocr
docker compose build --build-arg SRC_CACHEBUST=$(date +%s) imageocr
docker compose up -d imageocr