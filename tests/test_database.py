import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.database import get_connection


connection = get_connection()

tables = connection.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
""").fetchall()

for table in tables:
    print(table["name"])

connection.close()