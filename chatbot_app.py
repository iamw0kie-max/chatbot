# -------------------------------------------------
# 🧩 Personal Information Assistant with NLP + MySQL
# -------------------------------------------------

import streamlit as st
import re
import spacy
import mysql.connector
from datetime import datetime

# -----------------------------------------
# ⚙️ Database Connection
# -----------------------------------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Quickpwd@123",
    database="chatbot_db"
)
cursor = conn.cursor()

# -----------------------------------------
# 📦 Load NLP Model
# -----------------------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------------------
# 🧱 Create Tables
# -----------------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS personal_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(50),
    middle_name VARCHAR(50),
    last_name VARCHAR(50),
    age INT,
    name VARCHAR(100),
    birth_date DATE,
    birth_place VARCHAR(100),
    address VARCHAR(255),
    occupation VARCHAR(100),
    organization VARCHAR(100),
    blood_group VARCHAR(10),
    health_status VARCHAR(255),
    spouse_name VARCHAR(100),
    health_insurance VARCHAR(255),
    hobbies TEXT,
    fav_food VARCHAR(100),
    fav_game VARCHAR(100),
    games_played TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS family_members (
    id INT AUTO_INCREMENT PRIMARY KEY,
    person_id INT,
    name VARCHAR(100),
    relation VARCHAR(50),
    FOREIGN KEY (person_id) REFERENCES personal_info(id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS friends (
    id INT AUTO_INCREMENT PRIMARY KEY,
    person_id INT,
    name VARCHAR(100),
    FOREIGN KEY (person_id) REFERENCES personal_info(id)
)
""")
conn.commit()

# -----------------------------------------
# 🧠 NLP Extraction
# -----------------------------------------
def extract_personal_info(text):
    doc = nlp(text)
    info = {
        "first_name": None,
        "middle_name": None,
        "last_name": None,
        "age": None,
        "name": None,
        "birth_place": None,
        "address": None,
        "occupation": None,
        "organization": None,
        "blood_group": None,
        "health_status": None,
        "spouse_name": None,
        "health_insurance": None,
        "hobbies": None,
        "fav_food": None,
        "fav_game": None,
        "games_played": None
    }

    # Named entities
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            full_name = ent.text.strip()
            parts = full_name.split()
            if len(parts) == 1:
                info["first_name"] = parts[0]
            elif len(parts) == 2:
                info["first_name"], info["last_name"] = parts
            elif len(parts) >= 3:
                info["first_name"], info["middle_name"], info["last_name"] = parts[0], " ".join(parts[1:-1]), parts[-1]
            info["name"] = full_name
        elif ent.label_ in ["GPE", "LOC"]:
            info["birth_place"] = ent.text
        elif ent.label_ == "ORG":
            info["organization"] = ent.text

    # Regex extraction patterns
    patterns = {
        "age": r"\b(?:i am|i'm|age is|my age is)\s+(\d{1,3})\b",
        "blood_group": r"\b(A|B|AB|O)[+-]\b",
        "occupation": r"\b(engineer|teacher|developer|manager|doctor|designer|student)\b",
        "fav_food": r"favorite food is ([a-zA-Z ]+)",
        "fav_game": r"favorite game is ([a-zA-Z ]+)",
        "games_played": r"i (?:play|love playing) ([a-zA-Z ,]+)",
        "spouse_name": r"my (?:wife|husband|spouse) is ([A-Z][a-z]+)",
        "health_insurance": r"(?:insurance|policy) (?:from|with) ([A-Za-z0-9 ]+)",
        "hobbies": r"(?:i like|i love|my hobbies are) ([a-zA-Z ,]+)",
        "address": r"(?:address is|i live at) (.+)"
    }

    for key, pat in patterns.items():
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            info[key] = match.group(1).strip()

    return info


# -----------------------------------------
# 💾 Store / Update Personal Info
# -----------------------------------------
def store_personal_info(user_input):
    info = extract_personal_info(user_input)
    if not any(info.values()):
        return None

    cursor.execute("SELECT id FROM personal_info ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()

    if result:
        person_id = result[0]
        for key, value in info.items():
            if value:
                cursor.execute(f"UPDATE personal_info SET {key}=%s WHERE id=%s", (value, person_id))
    else:
        columns = ", ".join([k for k, v in info.items() if v])
        values = tuple(v for v in info.values() if v)
        placeholders = ", ".join(["%s"] * len(values))
        cursor.execute(f"INSERT INTO personal_info ({columns}) VALUES ({placeholders})", values)

    conn.commit()
    return "Got it! I’ve updated your personal information."


# -----------------------------------------
# 👨‍👩‍👧 Family & Friends Info
# -----------------------------------------
def store_relation_info(user_input):
    cursor.execute("SELECT id FROM personal_info ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return None
    person_id = result[0]

    # Family member
    family_match = re.search(r"my (\w+) is ([A-Z][a-z]+)", user_input)
    if family_match:
        relation, name = family_match.groups()
        cursor.execute("INSERT INTO family_members (person_id, name, relation) VALUES (%s, %s, %s)", (person_id, name, relation))
        conn.commit()
        return f"Got it! Added your {relation} named {name}."

    # Friend
    friend_match = re.search(r"my friend ([A-Z][a-z]+)", user_input)
    if friend_match:
        name = friend_match.group(1)
        cursor.execute("INSERT INTO friends (person_id, name) VALUES (%s, %s)", (person_id, name))
        conn.commit()
        return f"Nice! I’ve added your friend {name}."

    return None


# -----------------------------------------
# 🔍 Fetch Info from Database
# -----------------------------------------
def fetch_personal_info():
    cursor.execute("""
    SELECT first_name, middle_name, last_name, age, birth_place, occupation, organization, address, fav_food, fav_game, hobbies
    FROM personal_info ORDER BY id DESC LIMIT 1
    """)
    result = cursor.fetchone()
    if not result:
        return "I don’t know you yet. Tell me something about yourself!"

    first, middle, last, age, birth_place, occupation, org, addr, food, game, hobbies = result
    full_name = " ".join(filter(None, [first, middle, last]))

    parts = []
    if full_name: parts.append(f"Your full name is {full_name}")
    if age: parts.append(f"and you are {age} years old")
    if occupation: parts.append(f"you work as a {occupation}")
    if org: parts.append(f"at {org}")
    if birth_place: parts.append(f"from {birth_place}")
    if addr: parts.append(f"and you live at {addr}")
    if food: parts.append(f"you love eating {food}")
    if game: parts.append(f"your favorite game is {game}")
    if hobbies: parts.append(f"you enjoy {hobbies}")

    return ", ".join(parts) + "."


def fetch_friends():
    cursor.execute("SELECT name FROM friends")
    data = cursor.fetchall()
    if data:
        friends = [f[0] for f in data]
        return f"Your friends are: {', '.join(friends)}."
    return "I don't know your friends yet."


def fetch_family():
    cursor.execute("SELECT name, relation FROM family_members")
    data = cursor.fetchall()
    if data:
        family = [f"{n} ({r})" for n, r in data]
        return f"Your family members are: {', '.join(family)}."
    return "I don't know your family yet."


# -----------------------------------------
# 🧹 Clear All History
# -----------------------------------------
def clear_history():
    cursor.execute("DELETE FROM family_members")
    cursor.execute("DELETE FROM friends")
    cursor.execute("DELETE FROM personal_info")
    conn.commit()
    return "✅ All history has been cleared successfully!"


# -----------------------------------------
# 💬 Streamlit Chat Interface
# -----------------------------------------
st.markdown("### 💬 Chat with your assistant:")
user_input = st.text_input("You:", placeholder="Type something like 'My name is Ravi Kumar Sharma' or 'I am 29 years old'")

col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🧹 Clear History"):
        msg = clear_history()
        st.warning(msg)

if user_input:
    u = user_input.lower()

    if "who am i" in u or "tell me about myself" in u:
        st.info(fetch_personal_info())
    elif "my friends" in u:
        st.info(fetch_friends())
    elif "my family" in u:
        st.info(fetch_family())
    else:
        # Try to store or update info
        response = store_relation_info(user_input)
        if not response:
            response = store_personal_info(user_input)
        st.success(response or "Sorry, I couldn't understand that.")
