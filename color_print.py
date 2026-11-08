GRAY = "\033[90m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BASE_COLOR = "\033[0m"

def print_warning(text, **kwargs):
    print(warning(text), **kwargs)

def print_error(text, **kwargs):
    print(error(text), **kwargs)

def print_success(text, **kwargs):
    print(success(text), **kwargs)

def print_label(text, **kwargs):
    print(label(text), **kwargs)


def warning(text):
    return f"{YELLOW}{text}{BASE_COLOR}"

def error(text):
    return f"{RED}{text}{BASE_COLOR}"

def success(text):
    return f"{GREEN}{text}{BASE_COLOR}"

def label(text):
    return f"{GRAY}{text}{BASE_COLOR}"