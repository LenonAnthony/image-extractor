"""
Utilities for retrying operations that might fail due to transient network issues.
"""
import time
import random
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def retry_with_exponential_backoff(
    max_retries=5, 
    base_delay=1, 
    max_delay=60, 
    backoff_factor=2,
    exceptions=(Exception,)
):
    """
    Retry decorator with exponential backoff for handling transient network errors.
    
    Args:
        max_retries (int): Maximum number of retries before giving up
        base_delay (float): Initial delay between retries in seconds
        max_delay (float): Maximum delay between retries in seconds
        backoff_factor (float): Multiplicative factor for delay after each retry
        exceptions (tuple): Exception types to catch and retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) exceeded. Last error: {str(e)}")
                        raise
                    
                    # Calculate jitter (random value between 0 and 0.1*delay)
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = min(delay + jitter, max_delay)
                    
                    logger.warning(
                        f"Attempt {retries}/{max_retries} failed with error: {str(e)}. "
                        f"Retrying in {sleep_time:.2f} seconds..."
                    )
                    
                    time.sleep(sleep_time)
                    delay = min(delay * backoff_factor, max_delay)
        
        return wrapper
    
    return decorator