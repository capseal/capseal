"""Sample server with security issues for testing."""
import yaml
import pickle


def load_config(data):
    """Unsafe YAML load - Semgrep: python.lang.security.deserialization.avoid-pyyaml-load"""
    return yaml.load(data)


def deserialize(data):
    """Unsafe pickle - Semgrep: python.lang.security.deserialization.avoid-pickle"""
    return pickle.loads(data)


def format_query(table, user_input):
    """SQL injection - Semgrep: python.lang.security.audit.formatted-sql-query"""
    query = "SELECT * FROM %s WHERE id = '%s'" % (table, user_input)
    return query
