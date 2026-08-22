import sqlite3

DATABASE_NAME = "career_os.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_NAME)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            education TEXT,
            skills TEXT,
            interests TEXT,
            career_goal TEXT,
            experience TEXT,
            location TEXT
        )
    """)

    connection.commit()
    connection.close()