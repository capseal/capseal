import sys
sys.path.append(".")
try:
    import bef_zk.bef_rust as bef_rust
    print("Direct import attributes:", dir(bef_rust))
except ImportError:
    pass

try:
    from bef_zk import bef_rust
    print("Module import attributes:", dir(bef_rust))
except ImportError:
    print("Cannot import bef_zk.bef_rust")
