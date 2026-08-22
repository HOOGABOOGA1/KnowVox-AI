from fastapi import FastAPI
from database.database import initialize_database, get_connection

app = FastAPI()

# Initialize database when CareerOS starts
initialize_database()


# -------------------------
# HOME
# -------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to CareerOS!",
        "status": "online"
    }


# -------------------------
# CREATE USER
# -------------------------

@app.post("/users")
def create_user(
    name: str,
    education: str,
    skills: str,
    interests: str,
    career_goal: str,
    experience: str,
    location: str
):
    connection = get_connection()

    cursor = connection.execute("""
        INSERT INTO users
        (name, education, skills, interests, career_goal, experience, location)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        education,
        skills,
        interests,
        career_goal,
        experience,
        location
    ))

    connection.commit()

    user_id = cursor.lastrowid

    connection.close()

    return {
        "message": "User created successfully",
        "user_id": user_id
    }


# -------------------------
# GET USER
# -------------------------

@app.get("/users/{user_id}")
def get_user(user_id: int):
    connection = get_connection()

    user = connection.execute("""
        SELECT *
        FROM users
        WHERE id = ?
    """, (user_id,)).fetchone()

    connection.close()

    if user is None:
        return {
            "error": "User not found"
        }

    return dict(user)


# -------------------------
# UPDATE USER
# -------------------------

@app.put("/users/{user_id}")
def update_user(
    user_id: int,
    name: str,
    education: str,
    skills: str,
    interests: str,
    career_goal: str,
    experience: str,
    location: str
):
    connection = get_connection()

    cursor = connection.execute("""
        UPDATE users
        SET
            name = ?,
            education = ?,
            skills = ?,
            interests = ?,
            career_goal = ?,
            experience = ?,
            location = ?
        WHERE id = ?
    """, (
        name,
        education,
        skills,
        interests,
        career_goal,
        experience,
        location,
        user_id
    ))

    connection.commit()

    updated = cursor.rowcount

    connection.close()

    if updated == 0:
        return {
            "error": "User not found"
        }

    return {
        "message": "User updated successfully",
        "user_id": user_id
    }