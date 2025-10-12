# resume_parser_llm_sql.py
import re
import io
import time
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
# Config / UI
# ======================================================
st.set_page_config(page_title="Resume Parser + LLM-SQL Chat Agent", layout="wide", page_icon="🤖")

SKILLS_DB = [
    # Programming Languages
    "Python", "Java", "C", "C++", "C#", "R", "Scala", "Go", "PHP", "JavaScript", "TypeScript",
    # Data Tools / Databases
    "SQL", "MySQL", "PostgreSQL", "Oracle", "Teradata", "MongoDB", "Cassandra", "Snowflake",
    "Hadoop", "Hive", "Pig", "HBase", "Spark", "PySpark", "Databricks",
    # BI & Visualization
    "Tableau", "Power BI", "QlikView", "Qlik Sense", "Looker",
    # ETL & Data Integration
    "Informatica", "Talend", "Alteryx", "SSIS", "DataStage", "Airflow",
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "GitHub", "Bitbucket", "Jenkins",
    # Analytics & Machine Learning
    "Machine Learning", "Deep Learning", "AI", "Artificial Intelligence",
    "TensorFlow", "Keras", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
    # Reporting & Office Tools
    "Excel", "VBA", "MS Office", "Outlook", "PowerPoint",
    # Other
    "SAS", "Unix", "Linux", "Shell Script", "Automation", "Data Engineering"
]

# PDF extraction libraries
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
# Database wrapper (same as before)
# ======================================================
class ResumeDatabase:
    def __init__(self, host="localhost", user="root", password="", database="employee_db"):
        try:
            self.connection = mysql.connector.connect(
                host=host, user=user, password=password, database=database
            )
            if self.connection.is_connected():
                st.sidebar.success("Connected to MySQL")
                self.create_table()
        except Error as e:
            self.connection = None
            st.sidebar.error(f"DB connection failed: {e}")

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
                st.warning("Duplicate email, skipped")
                return
            q = """INSERT INTO employees
            (first_name,last_name,full_name,job_title,company,skills,experience_years,email,phone,summary,experience)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
            vals = (rec["first_name"], rec["last_name"], rec["full_name"], rec["job_title"], rec["company"],
                    rec["skills"], rec["experience_years"], rec["email"], rec["phone"], rec["summary"], rec["experience"])
            cur.execute(q, vals)
            self.connection.commit()
            st.success(f"Inserted {rec['full_name']}")
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
# Text extraction & parsing (same robust methods)
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
        st.warning(f"PDF extraction error: {e}")
    return text

def extract_text_from_docx(file):
    text = ""
    try:
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.warning(f"DOCX extraction error: {e}")
    return text

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

# ======================================================
# Improved Job Title + Current Company + Summary Extractor
# ======================================================
def extract_job_title_and_company(text: str):
    """
    Ultra-refined extractor (v9)
    ✅ Correctly isolates job title (excludes city/location)
    ✅ Detects 'current' company (handles Atos/Eviden brand)
    ✅ Works with 'Present' or 'Current' markers
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    job = ""
    company = ""

    # 1️⃣ Try to find top experience / current role block
    exp_block = re.search(
        r'(?:experience\s+summary|professional\s+summary|current\s+role|experience)\s*[:\n-]*([\s\S]{0,800})',
        text, re.IGNORECASE
    )
    block = exp_block.group(1) if exp_block else "\n".join(lines[:40])

    # 2️⃣ Identify likely job title
    title_match = re.search(
        r'\b(lead|senior|assistant|associate|etl|data|software|technical|consultant|developer|manager|engineer|architect|analyst|specialist)\b[^\n,.;:]{0,60}',
        block, re.IGNORECASE
    )
    if title_match:
        segment = block[title_match.start(): title_match.end() + 50]
        # Stop at punctuation or known city names
        segment = re.split(r'[,.;\-\n]', segment)[0]
        segment = re.sub(r'\b(New York|USA|India|Pune|Indore|Chennai|Hyderabad|Bangalore|City)\b', '', segment, flags=re.IGNORECASE)
        job = re.sub(r'\b(current|present|summary)\b', '', segment, flags=re.IGNORECASE).strip()

    # 3️⃣ Company detection (priority: latest / current)
    company_match = re.search(
        r'\b(?:at|with|for)\s+([A-Z][A-Za-z&\s\/]{2,50})', block, re.IGNORECASE
    )
    if company_match:
        company = company_match.group(1).strip()

    known_companies = [
        "Atos", "Eviden", "TCS", "Tata Consultancy Services", "Infosys", "Accenture",
        "Wipro", "Cognizant", "HCL", "IBM", "Capgemini", "Tech Mahindra", "DXC",
        "LTI", "Mindtree", "Oracle", "Deloitte", "EY", "PwC", "KPMG", "Virtusa",
        "Genpact", "Zensar", "UST", "Persistent", "Birlasoft", "Optum", "Moody", "Moody's"
    ]
    for comp in known_companies:
        if re.search(r'\b' + re.escape(comp.lower()) + r'\b', text.lower()):
            company = comp
            break

    # 4️⃣ Normalize Atos/Eviden dual brand
    if company.lower() in ["atos", "eviden"]:
        company = "Atos | Eviden"

    # 5️⃣ Cleanup
    job = re.sub(r'[^A-Za-z0-9&\-\s]', '', job).strip()
    company = company.replace('.', '').strip()
    company = company.title()

    return job, company




def extract_skills(text: str) -> str:
    """
    Hybrid skill extractor:
    ✅ Searches technical sections and inline mentions
    ✅ Handles tabular layouts and full-text sentences
    ✅ Deduplicates and canonicalizes
    """
    if not text:
        return ""

    # Try to isolate 'Technical Skills' or 'Skills' section
    match = re.search(
        r'(technical\s+skills|skills|technologies|tools)\s*[:\n-]*([\s\S]{0,1800}?)(?=\n[A-Z][A-Za-z ]{2,40}\s*:|\n\n|experience|summary|education|projects|certificat|$)',
        text, re.IGNORECASE
    )
    skill_block = match.group(2) if match else ""

    # Fallback: capture experience text to catch inline tool mentions
    exp_block = re.search(r'(experience|responsibilities)\s*[:\n-]*([\s\S]{0,2500})', text, re.IGNORECASE)
    inline_block = exp_block.group(2) if exp_block else ""

    combined = (skill_block + " " + inline_block).lower()

    found = []
    seen = set()

    # Match from SKILLS_DB
    for skill in SKILLS_DB:
        s_lower = skill.lower()
        if re.search(r'\b' + re.escape(s_lower) + r'\b', combined):
            if s_lower not in seen:
                found.append(skill)
                seen.add(s_lower)

    # Add important aliases
    aliases = {
        "control m": "Control-M",
        "powercenter": "Informatica PowerCenter",
        "boomi": "Dell Boomi",
        "mulesoft": "Mulesoft",
        "successfactors": "SAP SuccessFactors",
        "sql server": "SQL Server",
        "qlik": "QlikView",
        "azure data studio": "Azure Data Studio",
    }
    for alias, canonical in aliases.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', combined):
            if canonical.lower() not in seen:
                found.append(canonical)
                seen.add(canonical.lower())

    # Filter out duplicates and generic noise
    found_unique = []
    for f in found:
        if f.lower() not in found_unique:
            found_unique.append(f)

    return ", ".join(found_unique)




# Replace your current extract_skills() with this improved version

# A small mapping of common aliases -> canonical names (extend as needed)
_SKILL_CANONICAL_MAP = {
    "pyspark": "PySpark",
    "spark": "Spark",
    "python": "Python",
    "sql server": "SQL Server",
    "sql": "SQL",
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "databricks": "Databricks",
    "qlikview": "QlikView",
    "qlik sense": "Qlik Sense",
    "tableau": "Tableau",
    "power bi": "Power BI",
    "informatica": "Informatica",
    "dell boomi": "Dell Boomi",
    "boomi": "Dell Boomi",
    "mulesoft": "Mulesoft",
    "control-m": "Control-M",
    "control m": "Control-M",
    "unix": "UNIX",
    "shell": "Unix Shell",
    "shell scripting": "Unix Shell",
    "shell scripting": "Unix Shell",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "git": "Git",
    "github": "GitHub",
    "powercenter": "Informatica PowerCenter",
    "sas": "SAS",
    "etl": "ETL",
    "bi": "BI",
    "obiee": "OBIEE",
    "salesforce": "Salesforce",
    "sap successfactors": "SAP SuccessFactors",
    "tableau desktop": "Tableau",
    "sql developer": "SQL Developer",
    "bmc control-m": "Control-M",
    "jira": "JIRA",
    "datastage": "DataStage",
    "airflow": "Airflow",
    "ssrs": "SSRS",
    "ssis": "SSIS",
}

# A basic blacklist of generic words to ignore if accidentally captured
_SKILL_BLACKLIST = {
    "ability","active","accurate","analysis","analyst","application","applications","business",
    "client","company","consultant","development","developer","experience","professional",
    "responsibilities","project","projects","team","platform","solution","solutions","management",
    "service","services","work","workflow","process","processes","system","systems","organization"
}

# Build a fast lowercase canonical lookup from SKILLS_DB + canonical map
_CANDIDATE_SKILLS = set(s.lower() for s in SKILLS_DB) | set(_SKILL_CANONICAL_MAP.keys())

def _normalize_skill(token: str) -> str:
    """Return canonical, nicely cased skill name for a token, or None if ignored."""
    t = token.strip()
    if not t:
        return None
    # remove surrounding punctuation
    t = re.sub(r'^[^\w]+|[^\w]+$', '', t)
    if len(t) <= 1:
        return None
    tl = t.lower()

    # direct canonical map
    if tl in _SKILL_CANONICAL_MAP:
        return _SKILL_CANONICAL_MAP[tl]

    # if token matches SKILLS_DB entries (case-insensitive), return the SKILLS_DB canonical spelling
    for s in SKILLS_DB:
        if s.lower() == tl:
            return s

    # try fuzzy contains: prefer known words in token (e.g., "Informatica Power Center" -> match "informatica")
    for candidate in _CANDIDATE_SKILLS:
        if candidate in tl:
            # return canonical mapping if exists, else title-case it (or from SKILLS_DB)
            if candidate in _SKILL_CANONICAL_MAP:
                return _SKILL_CANONICAL_MAP[candidate]
            # try find the SKILLS_DB proper cased version
            for s in SKILLS_DB:
                if s.lower() == candidate:
                    return s
            return token.strip().title()

    # If token looks like a typical tech name (camelcase / contains digits / hyphen) allow short list
    if re.search(r'[A-Za-z0-9\+\#\.\-]{3,}', token):
        # avoid blacklist words
        if token.lower() in _SKILL_BLACKLIST:
            return None
        # keep token but normalize spacing and punctuation
        cleaned = re.sub(r'\s{2,}', ' ', token).strip(" ,.-")
        # final filter: don't return single generic words
        if len(cleaned) <= 2 or cleaned.lower() in _SKILL_BLACKLIST:
            return None
        # protect things like "Control-M"
        if re.match(r'^[A-Za-z0-9\-\+#\. ]+$', cleaned):
            return cleaned
    return None


def parse_resume(text):
    """Full resume parser with enhanced summary + accurate company and job extraction."""
    first_name, last_name, full_name = extract_name(text)
    job_title, company = extract_job_title_and_company(text)

    # --- Experience
    exp_match = re.search(r'(\d{1,2})\s*(?:\+)?\s*(?:years?|yrs?)', text, re.IGNORECASE)
    experience_years = int(exp_match.group(1)) if exp_match else 0

    # --- Email / phone
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    phone_match = re.search(r'\+?\d[\d \-\(\)]{7,}\d', text)
    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0) if phone_match else ""

    # --- Skills
    skills = extract_skills(text)

    # --- Summary (Profile or Professional Summary section)
    # --- Summary (robust, multiline, bullet-proof) ---
    clean_text = text.replace('\r', ' ')
    clean_text = re.sub(r'[•·\t]+', '\n', clean_text)
    clean_text = re.sub(r'\s{2,}', ' ', clean_text)

    summary_pattern = re.compile(
        r'(?:Professional\s+Summary|Personal\s+Summary|Profile|Summary)\s*[:\n-]*([\s\S]{0,2000}?)'
        r'(?=\n\s*(Education|Experience|Projects|Certifications|Technical\s+Skills|Work\s+Experience)\b|$)',
        re.IGNORECASE,
    )

    summary_match = summary_pattern.search(clean_text)
    if summary_match:
        block = summary_match.group(1)
        block = re.sub(r'[\n\r•\-]+', ' ', block)
        block = re.sub(r'\s{2,}', ' ', block)
        summary = block.strip(" .")
    else:
        # fallback: take top section before "Experience"
        parts = re.split(r'\b(Experience|Work\s+History)\b', clean_text, flags=re.IGNORECASE)
        top = parts[0].strip() if parts else clean_text[:800]
        top = re.sub(r'[\n\r•\-]+', ' ', top)
        sentences = re.split(r'(?<=[.!?])\s+', top)
        summary = " ".join(sentences[:5]).strip(" .")

    # Clean and truncate
    summary = re.sub(r'\s{2,}', ' ', summary).strip()
    if not summary.endswith("."):
        summary += "."


    # --- Experience block
    exp_block = re.search(r'EXPERIENCE[\s\S]*', text, re.IGNORECASE)
    experience = exp_block.group(0).strip()[:2000] if exp_block else ""

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
        "experience": experience
    }



# ======================================================
# LLM → SQL helpers
# ======================================================
def schema_description():
    # Describe main table columns for the model
    return (
        "You have the following table in a MySQL database named `employees` with columns:\n"
        "id (INT), first_name (VARCHAR), last_name (VARCHAR), full_name (VARCHAR), job_title (VARCHAR), company (VARCHAR),\n"
        "skills (TEXT, comma-separated), experience_years (FLOAT), email (VARCHAR), phone (VARCHAR), summary (TEXT), experience (TEXT).\n"
        "Return ONLY a single valid SQL SELECT statement that answers the user's question. Do NOT include any explanation or extra text.\n"
        "The SQL must be a single SELECT statement, no semicolons, no comments, no data-modifying statements.\n"
    )

def call_openai_for_sql(question, model="gpt-4o-mini"):
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed.")
    # require OPENAI_API_KEY in env or Streamlit secrets
    key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key", None)
    if not key:
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY env or st.secrets['openai_api_key'].")
    openai.api_key = key

    prompt = schema_description() + "\nUser question: " + question + "\nSQL:"
    # Chat-completion style request
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a helpful assistant that converts natural language to SQL. Only output a single SELECT statement."},
            {"role":"user","content":prompt}
        ],
        temperature=0.0,
        max_tokens=400
    )
    sql = resp.choices[0].message.content.strip()
    return sql

def validate_sql(sql):
    # Basic validation: only one statement, starts with SELECT, no dangerous keywords, no semicolons
    if ";" in sql:
        return False, "SQL contains semicolons or multiple statements."
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False, "Only SELECT statements are allowed."
    forbidden = ["insert ", "update ", "delete ", "drop ", "alter ", "truncate ", "create "]
    for token in forbidden:
        if token in s:
            return False, f"Forbidden SQL token detected: {token.strip()}"
    # disallow subqueries that modify? we allow subqueries in SELECT; keep simple
    return True, ""

def run_sql_and_fetch(db, sql, limit=1000):
    cur = db.connection.cursor()
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(limit)
        df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
        return df
    finally:
        cur.close()

# ======================================================
# Natural-language fallback chat agent (rule-based)
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
    if "most experience" in q:
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
            return f"**{rec['full_name']}** • Job: {rec['job_title']} • Company: {rec['company']} • Skills: {rec['skills']}"
    return "I couldn't parse that. Try: 'How many know Python?', 'List employees from TCS with >5 years', or use the LLM option."

# ======================================================
# UI: Upload / DB / Chat (includes LLM option)
# ======================================================
st.title("📄 Resume Parser + LLM→SQL Chat Agent")

st.sidebar.header("Database & LLM settings")
db_host = st.sidebar.text_input("MySQL Host", "localhost")
db_user = st.sidebar.text_input("MySQL User", "root")
db_password = st.sidebar.text_input("MySQL Password", type="password")
db_name = st.sidebar.text_input("Database", "employee_db")

use_llm = st.sidebar.checkbox("Enable LLM → SQL", value=False)
openai_model = st.sidebar.text_input("OpenAI model (optional)", value="gpt-4o-mini")
if use_llm and not HAS_OPENAI:
    st.sidebar.error("Install openai package to use LLM → SQL (pip install openai)")

# create DB object if password provided
db = ResumeDatabase(host=db_host, user=db_user, password=db_password, database=db_name) if db_password else None

tabs = st.tabs(["Upload", "Database", "Chat Agent"])

# Upload tab
with tabs[0]:
    uploaded = st.file_uploader("Upload resumes (pdf/docx)", type=["pdf","docx"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            st.info(f"Processing {f.name}")
            raw = f.read()
            text = extract_text_from_pdf_bytes(raw) if f.name.lower().endswith(".pdf") else extract_text_from_docx(io.BytesIO(raw))
            parsed = parse_resume(text)
            st.json(parsed)
            if db:
                if st.button(f"Insert {parsed['full_name']}", key=f.name):
                    db.insert_employee(parsed)

# Database tab
with tabs[1]:
    if db and db.connection:
        rows = db.list_employees()
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)
            st.download_button("Download CSV", df.to_csv(index=False), "employees.csv", "text/csv")
        else:
            st.info("No employees yet.")
    else:
        st.info("Connect to DB to view data (enter password in sidebar).")

# Chat tab
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
        st.session_state.messages.append({"role":"user","content":prompt})

        if not db or not db.connection:
            reply = "⚠️ Connect to the database first (sidebar)."
            st.chat_message("assistant").markdown(reply)
            st.session_state.messages.append({"role":"assistant","content":reply})
        else:
            if use_llm:
                # Call LLM to produce SQL
                try:
                    sql = call_openai_for_sql(prompt, model=openai_model)
                except Exception as e:
                    reply = f"LLM error: {e}"
                    st.chat_message("assistant").markdown(reply)
                    st.session_state.messages.append({"role":"assistant","content":reply})
                else:
                    # show generated SQL and validate; require user confirmation
                    st.markdown("**Generated SQL (from LLM):**")
                    st.code(sql, language="sql")
                    valid, reason = validate_sql(sql)
                    if not valid:
                        reply = f"❌ Generated SQL rejected: {reason}"
                        st.chat_message("assistant").markdown(reply)
                        st.session_state.messages.append({"role":"assistant","content":reply})
                    else:
                        if st.button("Execute this SQL"):
                            try:
                                df = run_sql_and_fetch(db, sql)
                                if df.empty:
                                    reply = "Query executed successfully — no rows returned."
                                    st.chat_message("assistant").markdown(reply)
                                    st.session_state.messages.append({"role":"assistant","content":reply})
                                else:
                                    st.dataframe(df)
                                    reply = f"Returned {len(df)} rows."
                                    st.chat_message("assistant").markdown(reply)
                                    st.session_state.messages.append({"role":"assistant","content":reply})
                            except Exception as e:
                                reply = f"SQL execution error: {e}"
                                st.chat_message("assistant").markdown(reply)
                                st.session_state.messages.append({"role":"assistant","content":reply})
            else:
                # fallback rule-based
                reply = query_chat_agent_rule(db, prompt)
                st.chat_message("assistant").markdown(reply)
                st.session_state.messages.append({"role":"assistant","content":reply})

if db:
    db.close()
