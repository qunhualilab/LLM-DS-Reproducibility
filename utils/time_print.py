from datetime import datetime

def time_print(*args, **kwargs):
    """
    A custom print function that prepends the current time to the message.
    
    Args:
        *args: Arguments to print.
        **kwargs: Keyword arguments to pass to the built-in print function.
    """
    current_time = datetime.now().strftime("[%H:%M:%S]")  # Format current time
    print(current_time, *args, **kwargs)

if __name__ == '__main__':
    pass
