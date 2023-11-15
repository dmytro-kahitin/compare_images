import logging

from app.config.environment_manager import EnvironmentManager
from app.messaging.rabbitmq_connection import RabbitMQConnection
from app.services.image_service import ImageService


class Main(EnvironmentManager):
    """
        Main class to initialize and start services.
    """
    def __init__(self):
        super().__init__([])
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.logger_level)
        self.logger.info('Initializing main...')

    def run(self):
        """
            Main function to initialize and start services.
            Establishes a RabbitMQ connection and initializes an ImageService object,
            then starts consuming messages from the RabbitMQ queue.
        """
        try:
            # Initialize the RabbitMQ connection
            rabbitmq_connection = RabbitMQConnection()

            # Initialize the ImageService with the RabbitMQ connection
            image_service = ImageService(rabbitmq_connection)
            self.logger.info('Starting image processing service...')

            # Start the message consumption process for ImageService
            image_service.start_consuming()
        except Exception as e:
            self.logger.error('Error occurred while running image processing service', exc_info=e)
            raise e


if __name__ == "__main__":
    main = Main()
    main.run()
