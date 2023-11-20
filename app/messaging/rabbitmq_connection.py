import json
import time
import uuid
import pika
import logging
from pika.exceptions import AMQPConnectionError

from app.config.environment_manager import EnvironmentManager
from app.services.image_service import OCR_IMAGE_QUEUE, COMPARE_IMAGES_QUEUE, RESPONSE_QUEUE, MAINTENANCE_QUEUE


class RabbitMQConnection(EnvironmentManager):
    """
        A class responsible for managing RabbitMQ connections and operations.

        This class inherits from EnvironmentManager for environment variable management and provides
        utility methods for managing RabbitMQ operations such as connection setup, message sending,
        and message consumption.
    """

    def __init__(self):
        """
            Initialize RabbitMQ connection parameters and establish a connection.
        """
        super().__init__([
            'RABBITMQ_HOST', 'RABBITMQ_PORT', 'RABBITMQ_USERNAME',
            'RABBITMQ_PASSWORD', 'RABBITMQ_VHOST', 'RABBITMQ_HEARTBEAT',
            'RABBITMQ_BLOCKED_CONNECTION_TIMEOUT'
        ])

        # Assign loaded environment variables to instance variables
        self._set_environment_vars()

        # Setup connection attributes
        self.connection = None
        self.channel = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.logger_level)
        self.logger.info('Initializing RabbitMQConnection...')

        # Setup and connect to RabbitMQ
        self._setup_connection()

    def _set_environment_vars(self):
        """
            Assign loaded environment variables to instance variables.
        """
        self.rabbitmq_host = self.env_vars['RABBITMQ_HOST']
        self.rabbitmq_port = int(self.env_vars['RABBITMQ_PORT'])
        self.rabbitmq_username = self.env_vars['RABBITMQ_USERNAME']
        self.rabbitmq_password = self.env_vars['RABBITMQ_PASSWORD']
        self.rabbitmq_vhost = self.env_vars['RABBITMQ_VHOST']
        self.rabbitmq_heartbeat = int(self.env_vars['RABBITMQ_HEARTBEAT'])
        self.rabbitmq_blocked_connection_timeout = int(self.env_vars['RABBITMQ_BLOCKED_CONNECTION_TIMEOUT'])

    def _setup_connection(self):
        """
        Setup RabbitMQ connection parameters and declare necessary queues with DLX configurations.
        """
        credentials = pika.PlainCredentials(self.rabbitmq_username, self.rabbitmq_password)
        self.parameters = pika.ConnectionParameters(
            self.rabbitmq_host, self.rabbitmq_port, self.rabbitmq_vhost,
            credentials, heartbeat=self.rabbitmq_heartbeat,
            blocked_connection_timeout=self.rabbitmq_blocked_connection_timeout
        )

        self.connection = None
        self.channel = None

        # Attempt to connect to RabbitMQ
        while not self.connection or self.connection.is_closed:
            try:
                self.connection = pika.BlockingConnection(self.parameters)
                self.channel = self.connection.channel()

                # Declare Dead Letter Exchange and Queue
                self.dlx_exchange = 'dlx_exchange'
                self.dlx_queue = 'dlx_queue'
                self.channel.exchange_declare(exchange=self.dlx_exchange, exchange_type='direct', durable=True)
                self.channel.queue_declare(queue=self.dlx_queue, durable=True)
                self.channel.queue_bind(queue=self.dlx_queue, exchange=self.dlx_exchange, routing_key='rejected')

                # Dead Letter Queue Arguments
                dead_letter_arguments = {
                    'x-dead-letter-exchange': self.dlx_exchange,
                    'x-dead-letter-routing-key': 'rejected'
                }

                # Declare application queues with Dead Letter Queue arguments
                self.channel.queue_declare(queue=OCR_IMAGE_QUEUE, durable=True, arguments=dead_letter_arguments)
                self.channel.queue_declare(queue=COMPARE_IMAGES_QUEUE, durable=True, arguments=dead_letter_arguments)
                self.channel.queue_declare(queue=RESPONSE_QUEUE, durable=True, arguments=dead_letter_arguments)
                self.channel.queue_declare(queue=MAINTENANCE_QUEUE, durable=True, arguments=dead_letter_arguments)

                self.channel.basic_qos(prefetch_count=1)
                self.logger.info('Connected to RabbitMQ')
            except AMQPConnectionError:
                self.logger.error('Failed to connect to RabbitMQ, retrying...')
                time.sleep(5)

    def connect(self):
        """
            Establish a connection to RabbitMQ and declare the necessary queues.
        """
        while not self.connection or self.connection.is_closed:
            try:
                self.connection = pika.BlockingConnection(self.parameters)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=OCR_IMAGE_QUEUE, durable=True)
                self.channel.queue_declare(queue=COMPARE_IMAGES_QUEUE, durable=True)
                self.channel.queue_declare(queue=RESPONSE_QUEUE, durable=True)
                self.channel.queue_declare(queue=MAINTENANCE_QUEUE, durable=True)
                self.channel.basic_qos(prefetch_count=1)
                self.logger.info('Connected to RabbitMQ')
            except AMQPConnectionError:
                self.logger.error('Failed to connect to RabbitMQ, retrying...')
                time.sleep(5)

    def close(self):
        """
            Close the RabbitMQ connection if it exists.
        """
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            self.logger.info('Closed RabbitMQ connection')

    def send_message(self, queue_name, message):
        """
            Send a message to a specified RabbitMQ queue.

            Args:
                queue_name (str): The name of the destination queue.
                message (dict): The message payload.

            Returns:
                tuple: The callback queue name and correlation id.
        """
        callback_queue = 'response_queue'
        self.channel.queue_declare(queue=callback_queue, durable=True)
        correlation_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            properties=pika.BasicProperties(
                reply_to=callback_queue,
                correlation_id=correlation_id,
            ),
            body=json.dumps(message)
        )
        self.logger.debug('Sent message to queue: %s', queue_name)
        return callback_queue, correlation_id

    def consume_response(self, callback_queue, correlation_id):
        """
            Consume messages from the callback queue and filter them by correlation_id.

            Args:
                callback_queue (str): The name of the callback queue.
                correlation_id (str): The correlation id to filter the messages by.

            Returns:
                str: The decoded message body that matches the given correlation_id.
        """
        while True:
            try:
                method_frame, properties, body = self.channel.basic_get(callback_queue)
                if method_frame:
                    self.channel.basic_ack(method_frame.delivery_tag)
                    if properties.correlation_id == correlation_id:
                        self.logger.debug('Received response message')
                        self.logger.debug('Response message: %s', body.decode())
                        return body.decode()
            except AMQPConnectionError:
                self.logger.error('Connection error to RabbitMQ, reconnecting...')
                self.connect()

    @staticmethod
    def parse_message(body):
        """
            Parse the received RabbitMQ message body into a JSON object.

            Args:
                body (bytes): The message body.

            Returns:
                dict: The parsed JSON message.
        """
        return json.loads(body.decode())
