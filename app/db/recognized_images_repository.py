import uuid
import logging
from pymongo import MongoClient
from app.config.environment_manager import EnvironmentManager


class RecognizedImagesRepository(EnvironmentManager):
    """
        Repository for managing recognized image data in MongoDB.

        Attributes:
            mongodb_host (str): MongoDB host address.
            mongodb_port (int): MongoDB connection port.
            mongodb_username (str): Username for MongoDB authentication.
            mongodb_password (str): Password for MongoDB authentication.
            mongodb_database (str): Name of the database to connect to.
            mongodb_collection (str): Name of the main collection.
            mongodb_similar_images_collection (str): Name of the collection for similar images.
    """

    def __init__(self):
        """
            Initialize MongoDB connection, its collections, and load environment variables.
        """
        super().__init__([
            'MONGODB_HOST', 'MONGODB_PORT', 'MONGODB_USERNAME',
            'MONGODB_PASSWORD', 'MONGODB_DATABASE', 'MONGODB_COLLECTION',
            'MONGODB_SIMILAR_IMAGES_COLLECTION'
        ])
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.logger_level)
        self.logger.info('Initializing Recognized Images repository...')

        self._initialize_variables()
        self._initialize_mongodb()

        # Ensure collections exist
        self.create_collections()

    def _initialize_variables(self):
        """
            Map loaded environment variables to instance variables.
        """
        self.mongodb_host = self.env_vars['MONGODB_HOST']
        self.mongodb_port = int(self.env_vars['MONGODB_PORT'])
        self.mongodb_username = self.env_vars['MONGODB_USERNAME']
        self.mongodb_password = self.env_vars['MONGODB_PASSWORD']
        self.mongodb_database = self.env_vars['MONGODB_DATABASE']
        self.mongodb_collection = self.env_vars['MONGODB_COLLECTION']
        self.mongodb_similar_images_collection = self.env_vars['MONGODB_SIMILAR_IMAGES_COLLECTION']

    def _initialize_mongodb(self):
        """
            Initialize MongoDB client and its collections.
        """
        try:
            self.mongo_client = MongoClient(
                self.mongodb_host, self.mongodb_port, username=self.mongodb_username,
                password=self.mongodb_password
            )
            self.db = self.mongo_client[self.mongodb_database]
            self.collection = self.db[self.mongodb_collection]
            self.similar_images_collection = self.db[self.mongodb_similar_images_collection]
            self.logger.info("MongoDB client initialized successfully")
        except Exception as e:
            self.logger.exception("Failed to initialize MongoDB client", exc_info=e)

    def create_collections(self):
        """
            Create MongoDB collections if they don't already exist.
        """
        try:
            existing_collections = self.db.list_collection_names()

            if self.mongodb_collection not in existing_collections:
                self.db.create_collection(self.mongodb_collection)
                self.logger.info(f"Created MongoDB collection: {self.mongodb_collection}")

            if self.mongodb_similar_images_collection not in existing_collections:
                self.db.create_collection(self.mongodb_similar_images_collection)
                self.logger.info(f"Created MongoDB similar images collection: {self.mongodb_similar_images_collection}")
        except Exception as e:
            self.logger.exception("Failed to create MongoDB collections", exc_info=e)

    def insert_image_details(self, doc):
        """
            Insert a single image document into the main collection.

            Args:
                doc (dict): Image document to be inserted.
        """
        try:
            self.collection.insert_one(doc)
            self.logger.debug("Inserted image details into MongoDB")
        except Exception as e:
            self.logger.exception("Failed to insert image details into MongoDB", exc_info=e)

    def get_all_images(self):
        """
            Retrieve all image documents from the main collection.

            Returns:
                list[dict]: List of all image documents.
        """
        try:
            images = list(self.collection.find())
            self.logger.debug("Retrieved all images from MongoDB")
            return images
        except Exception as e:
            self.logger.exception("Failed to retrieve all images from MongoDB", exc_info=e)
            return []


    def get_images_by_ids(self, image_ids):
        """
            Retrieve multiple image documents by their IDs.

            Args:
                image_ids (list[str]): List of image document IDs.

            Returns:
                list[dict]: List of image documents matching the IDs.
        """
        try:
            images = list(self.collection.find({"_id": {"$in": image_ids}}))
            self.logger.debug("Retrieved images by specific IDs from MongoDB")
            return images
        except Exception as e:
            self.logger.exception("Failed to retrieve images by IDs from MongoDB", exc_info=e)
            return []

    def get_images_by_xxhash(self, image_xxhash):
        """
            Retrieve all image documents that have a specific xxhash value.

            Args:
                image_xxhash (str): The hash value to look for in image documents.

            Returns:
                list[dict]: List of image documents with the specified xxhash.
        """
        try:
            images = list(self.collection.find({"xxhash": image_xxhash}))
            self.logger.debug(f"Retrieved images by xxhash: {image_xxhash}")
            return images
        except Exception as e:
            self.logger.exception(f"Failed to retrieve images by xxhash: {image_xxhash}", exc_info=e)
            return []

    def insert_similar_images(self, image_id, similar_images_ids):
        """
            Insert records of similar images into a separate collection.

            Args:
                image_id (str): ID of the source image.
                similar_images_ids (list[str]): List of similar image IDs.
        """
        try:
            docs = [{"_id": str(uuid.uuid4()), "source_image_id": image_id, "similar_image_id": img_id} for img_id in
                    similar_images_ids]
            self.similar_images_collection.insert_many(docs)
            self.logger.debug("Inserted similar images details into MongoDB")
        except Exception as e:
            self.logger.exception("Failed to insert similar images into MongoDB", exc_info=e)

    def clear_all_collections(self):
        """
            Clear all collections in database.
        """
        try:
            self.collection.drop()
            self.similar_images_collection.drop()
            self.logger.debug("All collections cleared successfully in MongoDB")
        except Exception as e:
            self.logger.exception("Failed to clear collections in MongoDB", exc_info=e)
