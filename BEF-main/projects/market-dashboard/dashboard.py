"""Market Dashboard - sample file for testing capseal review."""
import os
import subprocess

def get_user_input():
    """Get user input - potentially insecure."""
    user_data = input("Enter data: ")
    # This is insecure - command injection vulnerability
    result = subprocess.call("echo " + user_data, shell=True)
    return result

def load_config():
    """Load configuration - has hardcoded secrets."""
    # Hardcoded password - security issue
    db_password = "secret123"
    api_key = "sk-1234567890abcdef"
    return {"password": db_password, "api_key": api_key}

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, item):
        # Using eval - security vulnerability
        result = eval(item)
        self.data.append(result)
        return result

if __name__ == "__main__":
    config = load_config()
    print(f"Loaded config: {config}")
    get_user_input()
