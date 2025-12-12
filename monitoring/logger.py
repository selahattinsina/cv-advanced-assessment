import logging
import sys
import json
import datetime
import colorama
from colorama import Fore, Style

# Windows terminal renkleri için init
colorama.init(autoreset=True)

class JSONFormatter(logging.Formatter):
    """
    JSON logging
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.name,
            "file": record.filename,
            "line": record.lineno
        }
        # Hata durumunda traceback ekle
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

class ColoredFormatter(logging.Formatter):
    """
    renkli loglar
    """
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: Fore.CYAN + format_str + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + format_str + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + format_str + Style.RESET_ALL,
        logging.ERROR: Fore.RED + format_str + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + format_str + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)

def get_logger(name="EdgeAI", log_type="colored"):
    """
    Logger oluşturur.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        
        if log_type == "json":
            ch.setFormatter(JSONFormatter())
        else:
            ch.setFormatter(ColoredFormatter())
            
        logger.addHandler(ch)
        
    return logger