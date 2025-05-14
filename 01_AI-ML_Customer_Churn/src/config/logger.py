import logging
import os


def setup_logging(logging_config):
    logger = logging.getLogger()
    if not logger.handlers:
        level = getattr(logging, logging_config["level"].upper())
        logger.setLevel(level)
        log_dir = os.path.dirname(logging_config.get("log_file", "logs/project.log"))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        formatter = logging.Formatter(logging_config["format"])
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(
            logging_config.get("log_file", "logs/project.log")
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Logging setup complete.")
    else:
        logger.info("Logging already configured.")


# Example usage
if __name__ == "__main__":
    logger = setup_logging(
        {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "log_file": "logs/test.log",
        }
    )
    logger.info("This is a test log message.")
