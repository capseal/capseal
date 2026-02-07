"""Sample app with known security issues for testing."""
import subprocess
import hashlib
import os


def run_command(user_input):
    """Command injection vulnerability - Semgrep: python.lang.security.audit.dangerous-subprocess-use"""
    result = subprocess.call(user_input, shell=True)
    return result


def weak_hash(data):
    """Weak hash - Semgrep: python.lang.security.insecure-hash.insecure-md5"""
    return hashlib.md5(data.encode()).hexdigest()


def read_file(filename):
    """Path traversal - reads arbitrary files."""
    path = os.path.join("/data", filename)
    with open(path) as f:
        return f.read()


def make_temp():
    """Insecure temp file."""
    import tempfile
    return tempfile.mktemp()
