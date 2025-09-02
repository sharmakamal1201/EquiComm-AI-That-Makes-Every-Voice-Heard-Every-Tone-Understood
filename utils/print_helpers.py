"""
Terminal color print utility
"""

def print_colored(text, color="default"):
    colors = {
        "default": "\033[0m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    print(f"{colors.get(color, colors['default'])}{text}\033[0m")

# Logging helpers
def log_debug(msg):
    print_colored("DEBUG: " + msg, "yellow")

def log_info(msg):
    print_colored(msg, "cyan")

def log_error(msg):
    print_colored("ERROR: " + msg, "red")

def log_output(msg):
    print_colored("OUTPUT: " + msg, "green")
