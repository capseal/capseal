import importlib.util, os, sys
sys.path.append(".")
so_path = os.path.join(os.getcwd(), "bef_zk", "bef_rust.so")
spec = importlib.util.spec_from_file_location("bef_rust", so_path)
bef_rust = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bef_rust)

print("PyFriState methods:", dir(bef_rust.PyFriState))
