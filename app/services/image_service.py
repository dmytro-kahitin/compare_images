import logging
import os
import traceback
import uuid
from threading import Thread, Lock

from pika import BasicProperties

from app.config.environment_manager import EnvironmentManager
from app.db.recognized_images_repository import RecognizedImagesRepository
from app.services.image_hash_service import ImageHashService
from app.services.image_ocr_service import ImageOCRService
from app.services.image_similarity_service import ImageSimilarityService

# Constants for queue names
OCR_IMAGE_QUEUE = 'ocr_image_queue'
COMPARE_IMAGES_QUEUE = 'compare_images_queue'
RESPONSE_QUEUE = 'response_queue'
MAINTENANCE_QUEUE = 'maintenance_queue'

# Allowed image extensions for recognize and generate hashes
ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']


class ImageService(EnvironmentManager):
    """
        Main class to handle image services.

        Inherits from EnvironmentManager for environment variable management.
    """

    def __init__(self, messaging_connection):
        """
            Initializes with specified messaging_connection.
        """
        super().__init__(['ENABLE_MAINTENANCE_QUEUE'])
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.logger_level)
        self.logger.info('Initializing Image service...')
        self.enable_maintenance_queue = self.env_vars['ENABLE_MAINTENANCE_QUEUE'].lower() == "true"
        self.messaging_connection = messaging_connection
        self.db_connection = RecognizedImagesRepository()
        self.image_ocr_service = ImageOCRService()
        self.image_similarity_service = ImageSimilarityService()
        self.image_hash_service = ImageHashService()
        self.ocr_queue_empty = False
        self.lock = Lock()

    def consume_queues(self):
        """
            Consumes messages from OCR and Compare queues continuously.
        """
        try:
            while True:
                queue_status = self.messaging_connection.channel.queue_declare(queue=OCR_IMAGE_QUEUE, passive=True)
                while queue_status.method.message_count > 0:
                    self.consume_single_message(OCR_IMAGE_QUEUE)
                    queue_status = self.messaging_connection.channel.queue_declare(queue=OCR_IMAGE_QUEUE, passive=True)
                self.consume_single_message(COMPARE_IMAGES_QUEUE)
                self.consume_single_message(MAINTENANCE_QUEUE)
        except Exception as e:
            self.logger.exception("Exception while consuming messages", exc_info=e)

    def consume_single_message(self, queue_name):
        """
            Consumes a single message from a given queue.

            Args:
                queue_name: Name of the queue to consume from.
        """
        method_frame, properties, body = self.messaging_connection.channel.basic_get(queue=queue_name)
        if method_frame:
            self.logger.debug(f"Consuming single message from {queue_name}")
            self.process_message(queue_name, self.messaging_connection.channel, method_frame, properties, body)

    def process_message(self, queue_name, channel, method, properties, body):
        """
            Processes received message based on its queue.

            Args:
                queue_name: Name of the queue the message is from.
                channel: Channel object for communication.
                method: Method frame received.
                properties: Properties of the message.
                body: The actual message body.
        """
        try:
            task = self.messaging_connection.parse_message(body)
            if queue_name == OCR_IMAGE_QUEUE:
                self.handle_ocr_task(task)
            elif queue_name == COMPARE_IMAGES_QUEUE:
                self.handle_compare_task(task)
            elif queue_name == MAINTENANCE_QUEUE:
                self.handle_maintenance_task(task)
            else:
                self.logger.error(f"Unknown queue: {queue_name}")
                raise ValueError(f"Unknown queue: {queue_name}")
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            self.logger.exception(f'Exception while processing message from {queue_name}', exc_info=e)
            # Negative acknowledgment without requeuing
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            # Publish the failed message to the Dead Letter Exchange
            channel.basic_publish(
                exchange='dlx_exchange',
                routing_key='rejected',
                properties=BasicProperties(delivery_mode=2),
                body=body
            )
        finally:
            if queue_name == OCR_IMAGE_QUEUE:
                queue_status = channel.queue_declare(queue=OCR_IMAGE_QUEUE, passive=True)
                with self.lock:
                    self.ocr_queue_empty = queue_status.method.message_count == 0

    def start_consuming(self):
        """
            Starts a thread to consume messages from queues.
        """
        try:
            self.messaging_connection.connect()
            self.logger.info("Starting to consume messages...")
            consume_thread = Thread(target=self.consume_queues)
            consume_thread.start()
            consume_thread.join()
        except Exception as e:
            self.logger.exception("Exception while consuming messages", exc_info=e)
            traceback.print_exc()
        finally:
            if self.messaging_connection.connection.is_open:
                self.messaging_connection.connection.close()

    def handle_maintenance_task(self, task):
        """
            Handles maintenance tasks, such as clearing collections.

            Args:
                task: The task containing details about the maintenance operation.
        """
        action = task.get('action')
        if not self.enable_maintenance_queue:
            self.logger.warning("Maintenance action not allowed")
            return "Maintenance action not allowed."

        if action == 'clear_all_collections':
            self.db_connection.clear_all_collections()
            self.logger.info("All collections cleared successfully.")
            return "All collections cleared successfully."

        else:
            return "Unknown maintenance action."

    def get_recognized_text_or_none(self, task, image_xxhash):
        """
            Check if the image is already in the database, and return recognized text if present.

            Args:
                task (dict): Dictionary containing image details like path, id, etc.
                image_xxhash (str): The xxhash string of the image.

            Returns:
                tuple: A message string and the recognized text, if available.
        """
        existing_images = self.db_connection.get_images_by_xxhash(image_xxhash)
        for existing_image in existing_images:
            if existing_image['image_path'] == task['image_path']:
                self.logger.info("Image already recognized and saved")
                return 'Image already recognized and saved', existing_image['recognized_text']

        recognized_text = existing_images[0]['recognized_text'] if existing_images else None
        if recognized_text is None:
            recognized_text = self.image_ocr_service.get_text_from_image(task['image_path'])
            if recognized_text == "":
                self.logger.info("Text was not recognized or text length less than required")
                return "Text was not recognized or text len less than required", None
        return None, recognized_text

    def insert_image_to_db(self, task, image_hashes, recognized_text):
        """
            Insert image details into the database.

            Args:
                task (dict): Dictionary containing image details like path, id, etc.
                image_hashes (dict): The hashes dict of the image.
                recognized_text (str): The text recognized from the image.

            Returns:
                str: The generated UUID for the new image record.
        """
        current_image_id = str(uuid.uuid4())
        self.db_connection.insert_image_details({
            "_id": current_image_id,
            "xxhash": image_hashes['xxhash'],
            "ahash": image_hashes['ahash'],
            "dhash": image_hashes['dhash'],
            "whash_haar": image_hashes['whash_haar'],
            "colorhash": image_hashes['colorhash'],
            "image_id": task['image_id'],
            "image_path": task['image_path'],
            "recognized_text": recognized_text
        })
        self.logger.debug(f"Image inserted into database with ID: {current_image_id}")
        return current_image_id

    def handle_ocr_task(self, task):
        """
            Handles OCR tasks and saves recognized text to the database.

            Args:
                task (dict): The task dictionary containing details like image path.

            Returns:
                str: A message indicating the outcome of the operation.
        """
        self.logger.info("Start ocr task")
        image_path = task['image_path']
        if not os.path.exists(image_path):
            self.logger.warning(f"No image found at path: {image_path}")
            return 'No image'
        if not any(image_path.lower().endswith(ext) for ext in ALLOWED_IMAGE_EXTENSIONS):
            self.logger.warning(f"Incorrect file extension for image at path: {image_path}")
            return 'Incorrect file extension'

        image_hashes = self.image_hash_service.generate_image_hashes(image_path)
        message, recognized_text = self.get_recognized_text_or_none(task, image_hashes['xxhash'])
        if message:
            return message
        self.insert_image_to_db(task, image_hashes, recognized_text)
        self.logger.info("OCR task completed")
        return 'Recognition completed'

    def handle_compare_task(self, task):
        """
            Handles image comparison tasks and sends the result to a response queue.

            Args:
                task (dict): The task dictionary containing details like image path.

            Returns:
                str: A message indicating the outcome of the operation.
        """
        self.logger.info("Start comparison task")
        image_path = task['image_path']
        if not os.path.exists(image_path):
            self.logger.warning(f"No image found at path: {image_path}")
            return 'No image'
        if not any(image_path.lower().endswith(ext) for ext in ALLOWED_IMAGE_EXTENSIONS):
            self.logger.warning(f"Incorrect file extension for image at path: {image_path}")
            return 'Incorrect file extension'

        image_hashes = self.image_hash_service.generate_image_hashes(image_path)

        message, recognized_text = self.get_recognized_text_or_none(task, image_hashes['xxhash'])
        if not recognized_text:
            self.logger.warning(f"Image not recognized: {message}")
            return message

        recognized_text = self.image_similarity_service.preprocess_text(recognized_text)

        all_images = self.db_connection.get_all_images()

        similar_images_info = []
        for image in all_images:
            is_similar, similarity_percentage = self.image_hash_service.is_similar(image_hashes, image)
            if not is_similar:
                is_similar, similarity_percentage = self.image_similarity_service.is_similar(recognized_text, image.get('recognized_text'))
            if is_similar:
                similar_images_info.append({"id": image["_id"], "similarity": similarity_percentage})

        similar_images_data = self.db_connection.get_images_by_ids([info['id'] for info in similar_images_info])

        # Use a map for id to similarity linking to prevent any mix-up
        similarity_map = {info['id']: info['similarity'] for info in similar_images_info}
        for image in similar_images_data:
            image['similarity'] = similarity_map[image["_id"]]

        current_image_id = self.insert_image_to_db(task, image_hashes, recognized_text)

        if not similar_images_info:
            self.logger.info("No similar images found.")
            return 'Comparison completed'

        self.db_connection.insert_similar_images(current_image_id, [info['id'] for info in similar_images_info])

        similar_images = []
        for image in similar_images_data:
            similar_images.append({
                "image_id": image.get('image_id'),
                "image_path": image.get('image_path'),
                "similarity": image.get('similarity'),
                "recognized_text": image.get('recognized_text')
            })

        result_message = {
            "image_id": task['image_id'],
            "image_path": task['image_path'],
            "recognized_text": recognized_text,
            "similar_images": similar_images
        }
        self.messaging_connection.send_message(RESPONSE_QUEUE, result_message)
        self.logger.info(f"Comparison task completed successfully. Founded {len(similar_images_info)} similar images")
        return 'Comparison completed'
