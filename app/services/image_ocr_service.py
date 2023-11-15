import cv2
import logging
import time
from paddleocr import PaddleOCR
from app.config.environment_manager import EnvironmentManager


class ImageOCRService(EnvironmentManager):
    """
        Class to perform Optical Character Recognition (OCR) on images.

        Initialize OCR service with model paths.
        Inherits from EnvironmentManager for environment variable management.
    """

    def __init__(self):
        """
            Initialize and load environment variables.
        """
        super().__init__(['MIN_TEXT_LEN'])
        self.min_text_len = int(self.env_vars['MIN_TEXT_LEN'])

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.logger_level)
        self.logger.info('Initializing OCR service...')

        self.infer = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, det_model_dir='model')

    def get_text_from_image(self, image_path):
        """
            Extract text content from an image.

            Args:
                image_path (str): Path to the image file.

            Returns:
                str: Extracted text as a string.
        """
        self.logger.info(f'Processing image: {image_path}')
        img = cv2.imread(image_path)
        if img is None:
            print('Wrong path:', image_path)
            return ""

        start_time = time.time()
        try:
            result = self.get_ocr_text(img)
        except (Exception,):
            # Try OCR on upscaled image in case of exception
            upscaled_image = self.upscale_image(img)
            result = self.get_ocr_text(upscaled_image)

        self.logger.info(f'OCR time: {time.time() - start_time}')
        if len(result) <= self.min_text_len:
            self.logger.warning(f'Extracted text too short: {result}')
            return ""
        return result

    def get_ocr_text(self, image):
        """
            Extract text from the given image using OCR Inferencer.

            Args:
                image: Image data.

            Returns:
                str: Extracted text as a string.
        """
        result = self.infer.ocr(image, cls=True)
        recognized_text = ''
        for idx in range(len(result)):
            res = result[idx]
            if res is None:
                continue
            for line in res:
                recognized_text += f' {line[1][0]}'
        return recognized_text

    @staticmethod
    def upscale_image(image):
        """
            Utility function to upscale image.

            Args:
                image: Image data.

            Returns:
                : Upscaled image data.
        """
        height, width = image.shape[:2]
        return cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
