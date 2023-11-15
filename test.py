import os
import random
import time
import uuid

from rapidfuzz.distance.metrics_cpp import levenshtein_distance
from scipy.spatial.distance import euclidean, cityblock
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from app.db.recognized_images_repository import RecognizedImagesRepository
from app.messaging.rabbitmq_connection import RabbitMQConnection
from app.services.image_ocr_service import ImageOCRService
from app.services.image_service import OCR_IMAGE_QUEUE, COMPARE_IMAGES_QUEUE, RESPONSE_QUEUE
from app.services.image_similarity_service import ImageSimilarityService

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image
import imagehash


def test_dhash():
    results_file_path = 'results.txt'
    orig_images_dir = 'images\\orig'
    fake_images_dir = 'images\\fake'
    images_to_recognize = get_images_paths(orig_images_dir) | get_images_paths(fake_images_dir)
    compared_pairs = set()
    hashfuncs = [
        ('ahash', imagehash.average_hash),
        ('phash', imagehash.phash),
        ('dhash', imagehash.dhash),
        ('whash-haar', imagehash.whash),
        ('whash-db4', lambda image: imagehash.whash(image, mode='db4')),
        ('colorhash', imagehash.colorhash),
    ]

    # Initialize OCR and similarity services
    image_ocr_service = ImageOCRService()
    image_similarity_service = ImageSimilarityService()

    start_time = time.time()
    total_operations = len(images_to_recognize) * (len(images_to_recognize) - 1) * len(hashfuncs)
    progress_bar = tqdm(total=total_operations, desc="Processing")

    for recognize_image_name, recognize_image_path in images_to_recognize.items():
        img = Image.open(recognize_image_path)
        image_basename = os.path.basename(recognize_image_path)

        for target_recognize_image_name, target_recognize_image_path in images_to_recognize.items():
            if recognize_image_path == target_recognize_image_path:
                continue

            target_image_basename = os.path.basename(target_recognize_image_path)
            pair = frozenset({image_basename, target_image_basename})

            if pair in compared_pairs:
                continue

            target_image = Image.open(target_recognize_image_path)
            compared_pairs.add(pair)
            text = f'{image_basename} to {target_image_basename}:\n'
            with open(results_file_path, "a") as my_file:
                my_file.write(text)

            for hash_name, hash_func in hashfuncs:
                func_start_time = time.time()
                image_hash = hash_func(img)
                target_image_hash = hash_func(target_image)
                similarity = image_hash - target_image_hash
                func_end_time = time.time()
                text = f'-{hash_name}: Similarity: {similarity:.2f} (Less means more similar) Time: {func_end_time - func_start_time:.2f} seconds\n'
                with open(results_file_path, "a") as my_file:
                    my_file.write(text)
                progress_bar.update(1)

            # OCR Comparison
            ocr_start_time = time.time()
            recognized_original_text = image_ocr_service.get_text_from_image(recognize_image_path)
            recognized_target_text = image_ocr_service.get_text_from_image(target_recognize_image_path)
            ocr_similarity = image_similarity_service.compare_texts(recognized_original_text, recognized_target_text)
            ocr_end_time = time.time()

            text = f'-OCR Similarity: {ocr_similarity:.2f} (More means more similar) Time: {ocr_end_time - ocr_start_time:.2f} seconds\n\n'
            with open(results_file_path, "a") as my_file:
                my_file.write(text)
            progress_bar.update(1)

    # progress_bar.close()
    total_time = time.time() - start_time
    text = f'Total Execution Time: {total_time:.2f} seconds\n'
    with open(results_file_path, "a") as my_file:
        my_file.write(text)


def calculate_euclidean_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    vec1, vec2 = tfidf_matrix.toarray()
    distance = euclidean(vec1, vec2)
    return 1 / (1 + distance) * 100


def calculate_manhattan_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    vec1, vec2 = tfidf_matrix.toarray()
    distance = cityblock(vec1, vec2)
    return 1 / (1 + distance) * 100


def calculate_sorensen_dice_coefficient(text1, text2):
    if not text1 or not text2:
        return 0
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection_size = len(set1.intersection(set2))
    return (2 * intersection_size / (len(set1) + len(set2))) * 100


def calculate_overlap_coefficient(text1, text2):
    if not text1 or not text2:
        return 0
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection_size = len(set1.intersection(set2))
    smaller_set_size = min(len(set1), len(set2))
    return (intersection_size / smaller_set_size) * 100


def calculate_cosine_similarity(vec1, vec2):
    return cosine_similarity(vec1, vec2)[0][0]


def calculate_jaccard_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return (float(len(intersection)) / len(union)) * 100


def calculate_levenshtein_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    edit_distance = levenshtein_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    similarity = 1 - edit_distance / max_len
    return similarity * 100


def calculate_bow_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return calculate_cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1)) * 100


def calculate_tfidf_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return calculate_cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1)) * 100


def calculate_bow_and_tfidf_similarity(text1, text2):
    if not text1 or not text2:
        return 0

    try:
        bow_similarity = calculate_bow_similarity(text1, text2)
    except Exception as e:
        print(f"error: bow_similarity: {e}")
        bow_similarity = 0

    try:
        tfidf_similarity = calculate_tfidf_similarity(text1, text2)
    except Exception as e:
        print(f"error: tfidf_similarity: {e}")
        tfidf_similarity = 0

    return (bow_similarity + tfidf_similarity) / 2


def test_text_compare():
    results_file_path = 'compare_results.txt'
    orig_images_dir = 'images\\orig'
    fake_images_dir = 'images\\fake'
    images_to_recognize = get_images_paths(orig_images_dir) | get_images_paths(fake_images_dir)
    ocr_texts = dict()
    comparefuncs = [
        ('bow', calculate_bow_similarity),
        ('tfidf', calculate_tfidf_similarity),
        ('jaccard', calculate_jaccard_similarity),
        ('levenshtein', calculate_levenshtein_similarity),
        ('euclidean', calculate_euclidean_similarity),
        ('manhattan', calculate_manhattan_similarity),
        ('sorensen_dice', calculate_sorensen_dice_coefficient),
        ('overlap', calculate_overlap_coefficient),
        ('bow+tfidf', calculate_bow_and_tfidf_similarity)
    ]

    # Initialize OCR and similarity services
    image_ocr_service = ImageOCRService()

    start_time = time.time()
    total_operations = len(images_to_recognize) * (len(images_to_recognize) - 1) * len(comparefuncs)
    progress_bar = tqdm(total=total_operations, desc="Processing")

    for recognize_image_name, recognize_image_path in images_to_recognize.items():
        recognized_original_text = ocr_texts.get(recognize_image_name)
        if recognized_original_text is None:
            recognized_original_text = image_ocr_service.get_text_from_image(recognize_image_path)
            ocr_texts.update({recognize_image_name: recognized_original_text})
        image_basename = os.path.basename(recognize_image_path)

        for target_recognize_image_name, target_recognize_image_path in images_to_recognize.items():
            if recognize_image_path == target_recognize_image_path:
                progress_bar.update(1 * len(comparefuncs))
                continue
            recognized_target_text = ocr_texts.get(target_recognize_image_name)
            if recognized_target_text is None:
                recognized_target_text = image_ocr_service.get_text_from_image(target_recognize_image_path)
                ocr_texts.update({target_recognize_image_name: recognized_target_text})

            target_image_basename = os.path.basename(target_recognize_image_path)

            text = f'\n{image_basename} to {target_image_basename}:\n'
            with open(results_file_path, "a") as my_file:
                my_file.write(text)

            for compare_name, compare_func in comparefuncs:
                similarity = compare_func(recognized_original_text, recognized_target_text)
                text = f'-{compare_name}: Similarity: {similarity:.2f} (More means more similar)\n'
                with open(results_file_path, "a") as my_file:
                    my_file.write(text)
                progress_bar.update(1)

    # progress_bar.close()
    total_time = time.time() - start_time
    text = f'Total Execution Time: {total_time:.2f} seconds\n'
    with open(results_file_path, "a") as my_file:
        my_file.write(text)


def send_task(image_name, image_path, queue_name):
    rabbitmq_connection = RabbitMQConnection()
    message = {
        'image_id': f'{image_name}',
        "image_path": image_path,
    }
    rabbitmq_connection.send_message(queue_name, message)
    print(f'Message {image_name} sent to {queue_name}')


def get_images_paths(images_dir):
    image_paths = {}
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name_without_extension = os.path.splitext(filename)[0]
            image_paths[name_without_extension] = f'{images_dir}\\{filename}'
    return image_paths


def ocr_and_compare_all_images_test():
    orig_images_dir = 'images\\orig'
    fake_images_dir = 'images\\fake'
    images_to_recognize = get_images_paths(orig_images_dir)
    for recognize_image_name, recognize_image_path in images_to_recognize.items():
        send_task(recognize_image_name, recognize_image_path, OCR_IMAGE_QUEUE)

    images_to_compare = get_images_paths(fake_images_dir)
    for compare_image_name, compare_image_path in images_to_compare.items():
        send_task(compare_image_name, compare_image_path, COMPARE_IMAGES_QUEUE)


def random_task_test(enable_ocr_tasks=True, ocr_tasks_count=0, enable_compare_tasks=True, compare_tasks_count=0):
    orig_images_dir = 'images\\orig'
    fake_images_dir = 'images\\fake'
    images_to_recognize = get_images_paths(orig_images_dir)
    images_to_compare = get_images_paths(fake_images_dir)

    # Combine both recognize and compare tasks
    tasks = []

    if enable_ocr_tasks and ocr_tasks_count != 0:
        images_to_recognize_task_count = 0
        for recognize_image_name, recognize_image_path in images_to_recognize.items():
            if images_to_recognize_task_count >= ocr_tasks_count:
                break
            tasks.append(('ocr', recognize_image_name, recognize_image_path))
            images_to_recognize_task_count += 1

    if enable_compare_tasks and compare_tasks_count != 0:
        images_to_compare_task_count = 0
        for compare_image_name, compare_image_path in images_to_compare.items():
            if images_to_compare_task_count >= compare_tasks_count:
                break
            tasks.append(('compare', compare_image_name, compare_image_path))
            images_to_compare_task_count += 1

    # Shuffle tasks randomly
    random.shuffle(tasks)

    # Execute the tasks randomly
    for task in tasks:
        task_type, image_name, image_path = task
        if task_type == 'ocr':
            send_task(image_name, image_path, OCR_IMAGE_QUEUE)
        elif task_type == 'compare':
            send_task(image_name, image_path, COMPARE_IMAGES_QUEUE)


def random_ocr_image_test(write_to_db=True):
    image_ocr_service = ImageOCRService()
    if write_to_db is False:
        db_connection = RecognizedImagesRepository()
    else:
        db_connection = None
    orig_images_dir = 'images\\orig'
    fake_images_dir = 'images\\fake'
    all_images = get_images_paths(orig_images_dir)
    all_images.update(get_images_paths(fake_images_dir))
    # Combine both recognize and compare tasks
    tasks = []
    max_task_count = 2

    images_to_compare_task_count = 0
    for compare_image_name, compare_image_path in all_images.items():
        if images_to_compare_task_count >= max_task_count:
            break
        tasks.append((compare_image_name, compare_image_path))
        images_to_compare_task_count += 1

    # Shuffle tasks randomly
    random.shuffle(tasks)

    # Execute the tasks randomly
    for task in tasks:
        image_name, image_path = task
        recognized_text = image_ocr_service.get_text_from_image(image_path)
        if write_to_db is False:
            print(f'Name: {image_name} Text: {recognized_text}')
        else:
            image_id = random.randint(0, 100000)
            if db_connection:
                db_connection.insert_image_details({
                    "_id": str(uuid.uuid4()),
                    "image_id": image_id,
                    "image_path": image_path,
                    "recognized_text": recognized_text
                })


def random_compare_images_test():
    image_ocr_service = ImageOCRService()
    db_connection = RecognizedImagesRepository()
    image_similarity_service = ImageSimilarityService()
    orig_images_dir = 'images\\orig'
    fake_images_dir = 'images\\fake'
    all_images = get_images_paths(orig_images_dir)
    all_images.update(get_images_paths(fake_images_dir))
    tasks = []
    max_task_count = 2

    images_to_compare_task_count = 0
    for compare_image_name, compare_image_path in all_images.items():
        if images_to_compare_task_count >= max_task_count:
            break
        tasks.append((compare_image_name, compare_image_path))
        images_to_compare_task_count += 1

    random.shuffle(tasks)

    for task in tasks:
        image_name, image_path = task
        recognized_text = image_ocr_service.get_text_from_image(image_path)

        all_images = db_connection.get_all_images()

        similar_image_ids = []
        for image in all_images:
            if image_similarity_service.is_similar(recognized_text, image.get('recognized_text')):
                similar_image_ids.append(image["_id"])

        similar_images_data = db_connection.get_images_by_ids(similar_image_ids)

        similar_images = []
        similar_images_ids = []
        for image in similar_images_data:
            similar_images.append({
                "_id": image.get('_id'),
                "image_id": image.get('image_id'),
                "image_path": image.get('image_path'),
                "recognized_text": image.get('recognized_text')
            })
            similar_images_ids.append(image.get('_id'))

        image_id = random.randint(0, 100000)
        image_uuid = str(uuid.uuid4())
        db_connection.insert_image_details({
            "_id": str(uuid.uuid4()),
            "image_id": f'image_id_{image_id}',
            "image_path": image_path,
            "recognized_text": recognized_text
        })

        db_connection.insert_similar_images(image_uuid, similar_images_ids)

        result_message = {
            "image_id": task['image_id'],
            "image_path": task['image_path'],
            "recognized_text": recognized_text,
            "similar_images": similar_images
        }
        print(result_message)


if __name__ == "__main__":
    ocr_and_compare_all_images_test()
    # random_task_test(compare_tasks_count=2, ocr_tasks_count=2)
    # random_ocr_image_test()
    # random_compare_images_test()
    # test_dhash()
    # test_text_compare()
