import logging
import os
from dotenv import load_dotenv


class EnvironmentManager:
    """
        Class for managing environment variables from a .env file.

        Initialize and load environment variables specified in env_vars list.

        Args:
            env_vars (list): List of environment variable names to load.
    """

    def __init__(self, env_vars):
        """
            Initialize and load environment variables.

            Args:
                env_vars (list): List of environment variable names to load.
        """
        env_vars.append('LOGGER_LEVEL')
        load_dotenv()
        self.env_vars = {}

        for var in env_vars:
            value = os.getenv(var)
            if value is None:
                raise EnvironmentError(f'{var} is not set in .env file')
            self.env_vars[var] = value

        self.logger_level = None
        self.setup_logger_level()

    def setup_logger_level(self):
        """
            Setup default logger level
        """
        if 'LOGGER_LEVEL' in self.env_vars:
            self.logger_level = self.env_vars['LOGGER_LEVEL']
        if self.logger_level == "DEBUG":
            self.logger_level = logging.DEBUG
        elif self.logger_level == "WARNING":
            self.logger_level = logging.WARNING
        elif self.logger_level == "ERROR":
            self.logger_level = logging.ERROR
        elif self.logger_level == "FATAL":
            self.logger_level = logging.FATAL
        else:
            self.logger_level = logging.INFO
