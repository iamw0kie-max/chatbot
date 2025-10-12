import re
import io
import time
import streamlit as st
import docx
import mysql.connector
from mysql.connector import Error
import pandas as pd

# ======================================================
# 1️⃣ CONFIG
# ======================================================
st.set_page_config(page_title="Resume Parser + Chat Agent", layout="wide", page_icon="🤖")

SKILLS_DB = [
    "Python", "PySpark", "SAS", "SQL", "Tableau", "Teradata",
    "GitHub", "Excel", "AWS", "Power BI", "Java", "Spark"
]

# Try importing PyMuPDF (fitz), else fallback to PyPDF2
HAS_FITZ = False
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    from PyPDF2 import PdfReader
    HAS_FITZ = False

# Try loading spaCy
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    st.warning("⚠️ spaCy model missing — run: python -m spacy download en_core_web_sm")

# ======================================================
# 2️⃣ DATABASE CLASS
# ======================================================
class ResumeDatabase:
    def __init__(self, host="localhost", user="root", password="", database="employee_db"):
        try:
            self.connection = mysql.connector.connect(
                host=host, user=user, password=password, database=database
            )
            if self.connection.is_connected():
                st.sidebar.success("✅ Connected to MySQL database")
                self.create_table()
        except Error as e:
            self.connection = None
            st.sidebar.error(f"❌ Database connection failed: {e}")

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
        try:
            cur = self.connection.cursor()
            cur.execute("SELECT COUNT(*) FROM employees WHERE email=%s", (rec["email"],))
            if cur.fetchone()[0] > 0:
                st.warning(f"⚠️ Duplicate entry for {rec['email']}")
                return

            query = """
            INSERT INTO employees 
            (first_name, last_name, full_name, job_title, company, skills, experience_years, email, phone, summary, experience)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            vals = (
                rec["first_name"], rec["last_name"], rec["full_name"],
                rec["job_title"], rec["company"], rec["skills"],
                rec["experience_years"], rec["email"], rec["phone"],
                rec["summary"], rec["experience"]
            )
            cur.execute(query, vals)
            self.connection.commit()
            st.success(f"✅ Inserted: {rec['first_name']} {rec['last_name']}")
        except Exception as e:
            st.error(f"❌ Insert error: {e}")
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
# 3️⃣ TEXT EXTRACTION
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
                    txt = b[4].strip()
                    if txt:
                        text += txt + "\n"
            doc.close()
        else:
            reader = PdfReader(io.BytesIO(file_bytes))
            for p in reader.pages:
                text += (p.extract_text() or "") + "\n"
    except Exception as e:
        st.warning(f"⚠️ PDF extraction error: {e}")
    return text


def extract_text_from_docx(file):
    text = ""
    try:
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.warning(f"⚠️ DOCX extraction error: {e}")
    return text


# ======================================================
# 4️⃣ NAME + JOB EXTRACTION
# ======================================================
def extract_name(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidate = None

    top_block = []
    for ln in lines:
        if re.search(r'^\s*experience\b', ln, re.IGNORECASE):
            break
        top_block.append(ln)
        if len(top_block) > 20:
            break

    for ln in top_block:
        if re.search(r'\d|@|www|gmail|profile|summary|skills|experience|model|monitor', ln, re.IGNORECASE):
            continue
        if 1 < len(ln.split()) <= 4 and all(w[0].isupper() for w in ln.split() if w.isalpha()):
            candidate = ln.strip()
            break

    if not candidate and nlp:
        doc = nlp(" ".join(top_block))
        for ent in doc.ents:
            if ent.label_ == "PERSON" and 2 <= len(ent.text.split()) <= 4:
                candidate = ent.text
                break

    if not candidate:
        return "Unknown", "Unknown", "Unknown"

    parts = candidate.split()
    return parts[0], parts[-1], candidate


def extract_job_title_and_company(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    job = ""
    company = ""

    for ln in lines:
        if re.search(r'\b(Assistant|Senior|Lead|Manager|Consultant|Engineer|Analyst|Developer)\b', ln, re.IGNORECASE):
            job = re.sub(r'\b(Current|Present)\b', '', ln, flags=re.IGNORECASE)
            job = re.sub(r'[-|]', ' ', job)
            job = re.sub(r'\s+', ' ', job).strip()
            break

    comp = re.search(r'\b(TCS|Infosys|Accenture|Wipro|Cognizant|HCL|IBM|Tech\s?Mahindra|Capgemini)\b', text, re.IGNORECASE)
    if comp:
        company = comp.group(1)
    return job, company


# ======================================================
# 5️⃣ PARSE RESUME
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

    skills_found = [s for s in SKILLS_DB if re.search(r'\b' + re.escape(s) + r'\b', text, re.IGNORECASE)]
    skills = ", ".join(skills_found)

    summary_match = re.search(
        r'(Personal\s+Summary|Profile|Summary)\s*[:\n-]*([\s\S]*?)(?=\n[A-Z][A-Z\s]+:|Experience|Education|$)',
        text, re.IGNORECASE
    )
    summary = summary_match.group(2).strip() if summary_match else text[:300] + "..."

    exp_block = re.search(r'Experience\s*([\s\S]*)', text, re.IGNORECASE)
    experience = exp_block.group(1).strip()[:2000] if exp_block else ""

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
        "summary": summary,
        "experience": experience
    }


# ======================================================
# 6️⃣ CHAT AGENT
# ======================================================
def query_chat_agent(db, question):
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

    if "most experience" in q or "highest experience" in q or "most experienced" in q:
        cur.execute("SELECT full_name, experience_years, company FROM employees ORDER BY experience_years DESC LIMIT 1")
        r = cur.fetchone()
        if r:
            return f"The most experienced employee is {r['full_name']} with {r['experience_years']} years at {r['company']}."

    name_match = re.search(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b', question)
    if name_match:
        full = name_match.group(0)
        cur.execute("SELECT * FROM employees WHERE full_name LIKE %s", (f"%{full}%",))
        rec = cur.fetchone()
        if rec:
            return (
                f"**{rec['full_name']}**\n"
                f"• Job Title: {rec['job_title'] or 'N/A'}\n"
                f"• Company: {rec['company'] or 'N/A'}\n"
                f"• Experience: {rec['experience_years']} years\n"
                f"• Skills: {rec['skills']}\n"
                f"• Email: {rec['email']}\n"
                f"• Phone: {rec['phone']}\n"
            )
    return "🤖 Try asking: 'How many know Python?', 'Who has the most experience?', or 'Show details for Ravi Jatav'."


# ======================================================
# 7️⃣ STREAMLIT UI
# ======================================================
st.title("📄 Resume Parser + Chat Agent (PDF / Word → MySQL)")

st.sidebar.header("🗄️ Database Connection")
db_password = st.sidebar.text_input("Enter MySQL Password", type="password")

db = ResumeDatabase(password=db_password) if db_password else None

tab1, tab2, tab3 = st.tabs(["📂 Upload Resumes", "📋 Database", "💬 Chat Agent"])

# --- Upload & Parse ---
with tab1:
    uploaded = st.file_uploader("Upload resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            st.info(f"📄 Processing: {uf.name}")
            text = extract_text_from_pdf_bytes(uf.read()) if uf.name.lower().endswith(".pdf") else extract_text_from_docx(uf)
            rec = parse_resume(text)
            st.json(rec)
            if db:
                if st.button(f"Insert {rec['first_name']} {rec['last_name']}", key=uf.name):
                    db.insert_employee(rec)

# --- Database View ---
with tab2:
    if db:
        data = db.list_employees()
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df)
            st.download_button("⬇️ Download CSV", df.to_csv(index=False), "employees.csv", "text/csv")
        else:
            st.info("No employees yet.")
    else:
        st.warning("⚠️ Connect to database first.")

# --- Chat Agent ---
with tab3:
    st.subheader("💬 Resume Chat Agent")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me about uploaded resumes..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if db and db.connection and db.connection.is_connected():
            response = query_chat_agent(db, prompt)
        else:
            response = "⚠️ Please connect to MySQL first."

        time.sleep(0.3)
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if db:
    db.close()
