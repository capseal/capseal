"""File with shell=True - should FAIL shell injection check."""

import subprocess


def run_command(cmd):
    """Run a command - DANGEROUS: uses shell=True."""
    # This is unsafe and should be flagged
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return result.stdout.decode()


def safe_run(args):
    """Run a command safely without shell."""
    # This is safe - no shell=True
    result = subprocess.run(args, capture_output=True)
    return result.stdout.decode()
