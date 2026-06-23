import sys
sys.path.append(".")
from src.db.connections import test_connections
print(test_connections())