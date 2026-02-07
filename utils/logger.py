import logging
import os

def setup_logger(save_dir: str):
    """
    Sets up a logger that outputs to both console and a log file.
    """
    logger = logging.getLogger('WBEVFusion')
    logger.setLevel(logging.INFO)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # File Handler
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fh = logging.FileHandler(os.path.join(save_dir, 'train.log'))
    fh.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger
