import os
import logging
import xxhash
import imagehash
from PIL import Image

from app.config.environment_manager import EnvironmentManager


class ImageHashService(EnvironmentManager):
    """
        A service class for generating and comparing image hashes.

        This class provides methods to generate different types of image hashes and compare them to determine similarity.
        It inherits from `EnvironmentManager` to utilize environment variable management functionalities.
    """

    def __init__(self):
        """
            Initializes the ImageHashService with environment variables.
        """
        super().__init__([
            'AHASH_MAX_SIMILARITY_PERCENT',
            'DHASH_MAX_SIMILARITY_PERCENT',
            'WHASH_HAAR_MAX_SIMILARITY_PERCENT',
            'COLORHASH_MAX_SIMILARITY_PERCENT',
            'AHASH_SIMILARITY_OUTPUT',
            'DHASH_SIMILARITY_OUTPUT',
            'WHASH_HAAR_SIMILARITY_OUTPUT',
            'COLORHASH_SIMILARITY_OUTPUT'
        ])

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.logger_level)
        self.logger.info('Initializing ImageHashService...')

        self._initialize_variables()

    def _initialize_variables(self):
        """
            Map loaded environment variables to instance variables.
        """
        self.AHASH_MAX_SIMILARITY_PERCENT = float(self.env_vars['AHASH_MAX_SIMILARITY_PERCENT'])
        self.DHASH_MAX_SIMILARITY_PERCENT = float(self.env_vars['DHASH_MAX_SIMILARITY_PERCENT'])
        self.WHASH_HAAR_MAX_SIMILARITY_PERCENT = float(self.env_vars['WHASH_HAAR_MAX_SIMILARITY_PERCENT'])
        self.COLORHASH_MAX_SIMILARITY_PERCENT = float(self.env_vars['COLORHASH_MAX_SIMILARITY_PERCENT'])
        self.AHASH_SIMILARITY_OUTPUT = float(self.env_vars['AHASH_SIMILARITY_OUTPUT'])
        self.DHASH_SIMILARITY_OUTPUT = float(self.env_vars['DHASH_SIMILARITY_OUTPUT'])
        self.WHASH_HAAR_SIMILARITY_OUTPUT = float(self.env_vars['WHASH_HAAR_SIMILARITY_OUTPUT'])
        self.COLORHASH_SIMILARITY_OUTPUT = float(self.env_vars['COLORHASH_SIMILARITY_OUTPUT'])

    def _generate_image_xxhash(self, image_path):
        """
            Generate a 128-bit xxHash for an image.

            Args:
                image_path (str): Path to the image file.

            Returns:
                str: A 128-bit hash string representing the image or None if the file does not exist.
        """
        if not os.path.exists(image_path):
            return None

        hasher1 = xxhash.xxh64()
        hasher2 = xxhash.xxh64()
        with open(image_path, 'rb') as afile:
            while chunk := afile.read(4096):
                hasher1.update(chunk)
                hasher2.update(chunk[::-1])  # Reverse the chunk to create a different hash

        self.logger.debug(f'Generating xxhash for image: {image_path}')
        return hasher1.hexdigest() + hasher2.hexdigest()

    def _generate_ahash(self, image_path):
        """
            Calculate the average hash (aHash) for an image.

            Args:
                image_path (str): Path to the image file.

            Returns:
                str: The calculated aHash of the image.
        """
        img = Image.open(image_path)
        self.logger.debug(f'Generating average hash (aHash) for image: {image_path}')
        return str(imagehash.average_hash(img))

    def _generate_dhash(self, image_path):
        """
            Calculate the difference hash (dHash) for an image.

            Args:
                image_path (str): Path to the image file.

            Returns:
                str: The calculated dHash of the image.
        """
        img = Image.open(image_path)
        self.logger.debug(f'Generating difference hash (dHash) for image: {image_path}')
        return str(imagehash.dhash(img))

    def _generate_whash_haar(self, image_path):
        """
            Calculate the wavelet hash (wHash) using Haar wavelets for an image.

            Args:
                image_path (str): Path to the image file.

            Returns:
                str: The calculated wHash (Haar) of the image.
        """
        img = Image.open(image_path)
        self.logger.debug(f'Generating wavelet hash (wHash) using Haar wavelets for image: {image_path}')
        return str(imagehash.whash(img))

    def _generate_colorhash(self, image_path):
        """
            Calculate the color hash for an image.

            Args:
                image_path (str): Path to the image file.

            Returns:
                str: The calculated color hash of the image.
        """
        img = Image.open(image_path)
        color_hash = imagehash.colorhash(img)

        # Format each element in the hash array as a two-character hexadecimal string
        formatted_hash = ''.join(['{:02x}'.format(pixel) for pixel in color_hash.hash.flatten()])
        self.logger.debug(f'Generating colorhash for image: {image_path}')
        return formatted_hash

    def generate_image_hashes(self, image_path):
        """
            Generate various types of hashes for an image.

            This method combines the generation of xxHash and various image hashes including aHash, dHash, wHash (Haar),
            and colorHash.

            Args:
                image_path (str): Path to the image file.

            Returns:
                dict: A dictionary containing generated hashes or None if the image path does not exist.
        """
        if not os.path.exists(image_path):
            self.logger.warning(f'Image file does not exist: {image_path}')
            return None

        hashes = {
            'xxhash': self._generate_image_xxhash(image_path),
            'ahash': self._generate_ahash(image_path),
            'dhash': self._generate_dhash(image_path),
            'whash_haar': self._generate_whash_haar(image_path),
            'colorhash': self._generate_colorhash(image_path)
        }
        self.logger.debug(f'Generated hashes for image: {image_path}')
        return hashes

    def is_similar(self, target_hashes, hashes_to_compare):
        """
            Compare image hashes to determine if they are similar.

            This method compares various types of image hashes and determines if they are similar based on predefined
            maximum similarity percentages.

            Args:
                target_hashes (dict): Hashes of the target image.
                hashes_to_compare (dict): Hashes of the image to compare.

            Returns:
                tuple: A tuple containing a boolean indicating similarity and the corresponding similarity output value.
        """
        for hash_type in ['ahash', 'dhash', 'whash_haar', 'colorhash']:
            if imagehash.hex_to_hash(target_hashes[hash_type]) - imagehash.hex_to_hash(
                    hashes_to_compare[hash_type]) <= getattr(self, f'{hash_type.upper()}_MAX_SIMILARITY_PERCENT'):
                self.logger.debug(
                    f'Images are similar based on {hash_type}: '
                    f'{getattr(self, f"{hash_type.upper()}_SIMILARITY_OUTPUT")}'
                )
                return True, getattr(self, f'{hash_type.upper()}_SIMILARITY_OUTPUT')

        self.logger.debug('Images are not similar')
        return False, 0
