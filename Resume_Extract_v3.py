# resume_parser_llm_sql.py
import re
import io
import os
import streamlit as st
import docx
import mysql.connector
from mysql.connector import Error
import pandas as pd

# Optional LLM
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# ======================================================
# CONFIG / UI
# ======================================================
st.set_page_config(page_title="Resume Parser + LLM-SQL Chat Agent", layout="wide", page_icon="🤖")

SKILLS_DB = [
    "Python", "Java", "C", "C++", "C#", "R", "Scala", "Go", "PHP", "JavaScript", "TypeScript",
    "SQL", "MySQL", "PostgreSQL", "Oracle", "Teradata", "MongoDB", "Cassandra", "Snowflake",
    "Hadoop", "Hive", "Pig", "HBase", "Spark", "PySpark", "Databricks",
    "Tableau", "Power BI", "QlikView", "Qlik Sense", "Looker",
    "Informatica", "Talend", "Alteryx", "SSIS", "DataStage", "Airflow",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "GitHub", "Bitbucket", "Jenkins",
    "Machine Learning", "Deep Learning", "AI", "Artificial Intelligence",
    "TensorFlow", "Keras", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
    "Excel", "VBA", "MS Office", "Outlook", "PowerPoint", "SAS", "Unix", "Linux",
    "Shell Script", "Automation", "Data Engineering"
]

# PDF extraction
HAS_FITZ = False
try:
    import fitz
    HAS_FITZ = True
except Exception:
    from PyPDF2 import PdfReader
    HAS_FITZ = False

# spaCy
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    st.warning("spaCy model missing — run: python -m spacy download en_core_web_sm")

# ======================================================
# DATABASE CLASS
# ======================================================
class ResumeDatabase:
    def __init__(self, host="localhost", user="root", password="", database="employee_db"):
        try:
            self.connection = mysql.connector.connect(
                host=host, user=user, password=password, database=database
            )
            if self.connection.is_connected():
                st.sidebar.success("✅ Connected to MySQL")
                self.create_table()
        except Error as e:
            self.connection = None
            st.sidebar.error(f"❌ DB connection failed: {e}")

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS employees (
            id INT AUTO_INCREMENT PRIMARY KEY,
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            full_name VARCHAR(255),
            job_title VARCHAR(255),
            company VARCHAR(255),
            skills TEXT,
            experience_years FLOAT,
            email VARCHAR(255),
            phone VARCHAR(50),
            summary TEXT,
            experience TEXT
        )
        """
        cur = self.connection.cursor()
        cur.execute(query)
        self.connection.commit()
        cur.close()

    def insert_employee(self, rec):
        cur = self.connection.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM employees WHERE email=%s", (rec["email"],))
            if cur.fetchone()[0] > 0:
                st.warning(f"Duplicate email: {rec['email']}, skipped.")
                return
            q = """INSERT INTO employees
            (first_name,last_name,full_name,job_title,company,skills,experience_years,email,phone,summary,experience)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
            vals = (
                rec["first_name"], rec["last_name"], rec["full_name"], rec["job_title"],
                rec["company"], rec["skills"], rec["experience_years"], rec["email"],
                rec["phone"], rec["summary"], rec["experience"]
            )
            cur.execute(q, vals)
            self.connection.commit()
            st.success(f"✅ Inserted {rec['full_name']}")
        except Exception as e:
            st.error(f"Insert error: {e}")
        finally:
            cur.close()

    def list_employees(self):
        cur = self.connection.cursor(dictionary=True)
        cur.execute("SELECT * FROM employees ORDER BY id DESC")
        rows = cur.fetchall()
        cur.close()
        return rows

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

# ======================================================
# TEXT EXTRACTION
# ======================================================
def extract_text_from_pdf_bytes(file_bytes):
    text = ""
    try:
        if HAS_FITZ:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (round(b[1]), round(b[0])))
                for b in blocks:
                    t = b[4].strip()
                    if t:
                        text += t + "\n"
            doc.close()
        else:
            reader = PdfReader(io.BytesIO(file_bytes))
            for p in reader.pages:
                text += (p.extract_text() or "") + "\n"
    except Exception as e:
        st.warning(f"PDF extraction error: {e}")
    return text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.warning(f"DOCX extraction error: {e}")
    return text

# ======================================================
# FIELD EXTRACTION
# ======================================================
def extract_name(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for ln in lines[:10]:
        if not re.search(r'\d|@|skills|experience|summary|profile', ln, re.IGNORECASE):
            parts = ln.split()
            if 1 < len(parts) <= 4 and all(p[0].isupper() for p in parts if p.isalpha()):
                return parts[0], parts[-1], ln.strip()
    return "Unknown", "Unknown", "Unknown"

def extract_job_title_and_company(text):
    job, company = "", ""
    exp_block = re.search(r'(experience|summary)\s*[:\n-]*([\s\S]{0,500})', text, re.IGNORECASE)
    block = exp_block.group(2) if exp_block else text[:600]
    title_match = re.search(r'\b(lead|senior|data|software|developer|engineer|consultant|analyst|manager)\b[^\n,;:]{0,50}', block, re.IGNORECASE)
    if title_match:
        job = title_match.group(0).strip()
    known = ["Atos", "Eviden", "TCS", "Infosys", "Accenture", "IBM", "Wipro", "Cognizant", "HCL", "Oracle", "Moody"]
    for k in known:
        if k.lower() in text.lower():
            company = k
            break
    return job, company

def extract_skills(text):
    text_lower = text.lower()
    found = []
    for skill in SKILLS_DB:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found.append(skill)
    return ", ".join(sorted(set(found)))

# ======================================================
# PARSE RESUME FUNCTION
# ======================================================
def parse_resume(text):
    first_name, last_name, full_name = extract_name(text)
    job_title, company = extract_job_title_and_company(text)
    exp_match = re.search(r'(\d{1,2})\s*(?:\+)?\s*(?:years?|yrs?)', text, re.IGNORECASE)
    experience_years = int(exp_match.group(1)) if exp_match else 0
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    phone_match = re.search(r'\+?\d[\d \-\(\)]{7,}\d', text)
    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0) if phone_match else ""
    skills = extract_skills(text)
    summary_match = re.search(r'(summary|profile)\s*[:\n-]*([\s\S]{0,800})', text, re.IGNORECASE)
    summary = summary_match.group(2).strip() if summary_match else text[:400]
    return {
        "first_name": first_name,
        "last_name": last_name,
        "full_name": full_name,
        "job_title": job_title,
        "company": company,
        "skills": skills,
        "experience_years": experience_years,
        "email": email,
        "phone": phone,
        "summary": summary[:400],
        "experience": ""
    }

# ======================================================
# CHAT AGENT SUPPORT
# ======================================================
def query_chat_agent_rule(db, question):
    q = question.lower().strip()
    cur = db.connection.cursor(dictionary=True)
    for s in SKILLS_DB:
        if s.lower() in q and ("how many" in q or "count" in q):
            cur.execute("SELECT COUNT(*) AS c FROM employees WHERE skills LIKE %s", (f"%{s}%",))
            c = cur.fetchone()["c"]
            return f"There are {c} employees skilled in {s}."
    for s in SKILLS_DB:
        if s.lower() in q and ("list" in q or "who" in q):
            cur.execute("SELECT full_name, experience_years, company FROM employees WHERE skills LIKE %s", (f"%{s}%",))
            rows = cur.fetchall()
            if rows:
                response = f"Employees with {s}:\n"
                for r in rows:
                    response += f"- {r['full_name']} ({r['experience_years']} yrs, {r['company']})\n"
                return response
            return f"No employees found with {s}."
    return "I couldn't parse that. Try: 'How many know Python?', or 'List employees with SQL'."

# ======================================================
# STREAMLIT UI
# ======================================================
st.title("📄 Resume Parser + LLM→SQL Chat Agent")

st.sidebar.header("⚙️ Database Settings")
db_host = st.sidebar.text_input("MySQL Host", "localhost")
db_user = st.sidebar.text_input("MySQL User", "root")
db_password = st.sidebar.text_input("MySQL Password", type="password")
db_name = st.sidebar.text_input("Database", "employee_db")
resume_folder = st.sidebar.text_input("/Users/iamw0.W0KIE/PycharmProjects/PythonProject/chatbot/", "./synthetic_resumes")

db = ResumeDatabase(host=db_host, user=db_user, password=db_password, database=db_name) if db_password else None

tabs = st.tabs(["📂 Parse Folder", "🗂️ Database", "💬 Chat Agent"])

# =========================
# 📂 PARSE FOLDER TAB
# =========================
with tabs[0]:
    st.subheader("📁 Parse all resumes from folder")
    if st.button("Start Parsing Folder"):
        if not os.path.exists(resume_folder):
            st.error(f"Invalid folder path: {resume_folder}")
        else:
            files = [f for f in os.listdir(resume_folder) if f.lower().endswith((".pdf", ".docx"))]
            if not files:
                st.warning("No PDF/DOCX resumes found in this folder.")
            else:
                for file in files:
                    path = os.path.join(resume_folder, file)
                    st.info(f"Parsing {file} ...")
                    if file.lower().endswith(".pdf"):
                        with open(path, "rb") as f:
                            raw = f.read()
                            text = extract_text_from_pdf_bytes(raw)
                    else:
                        text = extract_text_from_docx(path)
                    rec = parse_resume(text)
                    st.json(rec)
                    if db:
                        db.insert_employee(rec)
                st.success("✅ Completed parsing all resumes!")

# =========================
# 🗂️ DATABASE TAB
# =========================
with tabs[1]:
    if db and db.connection:
        rows = db.list_employees()
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)
            st.download_button("⬇️ Download CSV", df.to_csv(index=False), "employees.csv", "text/csv")
        else:
            st.info("No employee records yet.")
    else:
        st.warning("Connect to DB first (enter credentials in sidebar).")

# =========================
# 💬 CHAT AGENT TAB
# =========================
with tabs[2]:
    st.subheader("💬 Ask about resumes")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask (e.g., 'List employees from TCS with >5 years')")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not db or not db.connection:
            reply = "⚠️ Connect to the database first (sidebar)."
        else:
            reply = query_chat_agent_rule(db, prompt)

        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

if db:
    db.close()
