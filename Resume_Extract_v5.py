# ======================================================
# resume_parser_llm_sql.py — Optimized for real-world resumes
# ======================================================
import re
import io
import os
import streamlit as st
import docx
import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import datetime
from dateutil import parser as dateparser
import matplotlib.pyplot as plt


# ======================================================
# Optional OpenAI Support
# ======================================================
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# ======================================================
# UI Config
# ======================================================
st.set_page_config(page_title="Resume Parser + LLM-SQL Chat Agent", layout="wide", page_icon="🤖")

# ======================================================
# Skills Database
# ======================================================
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

# ======================================================
# PDF Extraction
# ======================================================
HAS_FITZ = False
try:
    import fitz
    HAS_FITZ = True
except Exception:
    from PyPDF2 import PdfReader
    HAS_FITZ = False

# ======================================================
# spaCy Setup
# ======================================================
import spacy
from spacy.matcher import PhraseMatcher
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    st.warning("spaCy model missing — run: python -m spacy download en_core_web_sm")

# ======================================================
# Build spaCy Skill Matcher
# ======================================================
def build_skill_matcher(nlp, skills_list):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skills_list]
    matcher.add("SKILL", patterns)
    return matcher

matcher = build_skill_matcher(nlp, SKILLS_DB) if nlp else None

# ======================================================
# Database Class
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
            experience TEXT,
            job_title_conf FLOAT DEFAULT 0,
            company_conf FLOAT DEFAULT 0,
            skills_conf FLOAT DEFAULT 0,
            summary_conf FLOAT DEFAULT 0,
            source_file VARCHAR(255),
            parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            (first_name,last_name,full_name,job_title,company,skills,experience_years,
             email,phone,summary,experience,job_title_conf,company_conf,skills_conf,summary_conf,source_file)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
            vals = (
                rec["first_name"], rec["last_name"], rec["full_name"], rec["job_title"],
                rec["company"], rec["skills"], rec["experience_years"], rec["email"],
                rec["phone"], rec["summary"], rec["experience"],
                rec["job_title_conf"], rec["company_conf"], rec["skills_conf"], rec["summary_conf"],
                rec["source_file"]
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

# ======================================================
# Text Extraction
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
    try:
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        st.warning(f"DOCX extraction error: {e}")
        return ""

# ======================================================
# Parsing Helpers
# ======================================================
def parse_date_token(s):
    s = s.strip().replace("’", "'").replace("‘", "'")
    s = re.sub(r"'\s*(\d{2})\b", lambda m: " 20" + m.group(1), s)
    if re.search(r"present|current", s, re.IGNORECASE):
        return "PRESENT"
    try:
        return dateparser.parse(s, fuzzy=True, default=datetime(1900, 1, 1))
    except Exception:
        return None

# ======================================================
# Advanced Job + Company Extraction (v3)
# ======================================================

def parse_experience_entries(text):
    """
    Extracts likely job entries from resume text.
    Handles multiple patterns (with/without date), blocks skills-as-company, assigns confidence.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    entries = []

    # Define recognizers
    date_pattern = re.compile(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|\d{4})[^\n]{0,30})\s*(?:-|–|to|until)\s*([^\n,;]{2,30})', re.IGNORECASE)
    role_keywords = ["Engineer", "Developer", "Consultant", "Manager", "Lead", "Architect", "Analyst", "Specialist", "Scientist", "Expert", "Administrator"]
    skill_blacklist = set(s.lower() for s in SKILLS_DB)

    for i, line in enumerate(lines):
        if len(line) < 10 or "page" in line.lower():
            continue
        # --- Pattern: Numbered experience line like "1. MOODY’S CORPORATION | BNFS ..."
        if re.match(r"^\d+\.", line):
            # Split at pipe or dash
            parts = re.split(r"[|–\-]", line)
            if len(parts) >= 1:
                comp_candidate = parts[0]
                # Remove numeric prefix and date range
                comp_candidate = re.sub(r"^\d+\.\s*", "", comp_candidate).strip()
                comp_candidate = re.sub(r"\b(Aug|Sep|Sept|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul)['’]?\s*\d{2,4}\b.*",
                                        "", comp_candidate)
                # Validate that it's not a skill or a role
                if len(comp_candidate.split()) <= 6 and not any(k.lower() in comp_candidate.lower() for k in
                                                                ["engineer", "developer", "lead", "consultant",
                                                                 "manager"]):
                    entries.append({
                        "role": "",
                        "company": comp_candidate,
                        "start": None,
                        "end": None,
                        "confidence": 0.95
                    })
        # Capture date info
        date_match = date_pattern.search(line)
        start = end = None
        if date_match:
            start, end = parse_date_token(date_match.group(1)), parse_date_token(date_match.group(2))
            line = line[:date_match.start()].strip()

        # Try both formats
        m1 = re.search(r"(?P<role>[\w\s\/,&\-]+?)\s+(?:at|with|for|in)\s+(?P<company>[A-Z][\w&\.\s\-]{2,})", line)
        m2 = re.search(r"(?P<company>[A-Z][\w&\.\s\-]{2,})\s+(?:\-|–|—|,)\s+(?P<role>[\w\s\/,&\-]+)", line)

        role = company = ""
        conf = 0.0

        if m1:
            role, company = m1.group("role"), m1.group("company")
            conf = 0.9
        elif m2:
            company, role = m2.group("company"), m2.group("role")
            conf = 0.9
        else:
            # fallback: look for known job words
            found_role = [kw for kw in role_keywords if re.search(rf"\b{kw}\b", line, re.IGNORECASE)]
            if found_role:
                role = line.strip(" ,.-")
                conf = 0.6

        # Company post-processing
        company = re.sub(r'\b(for|labs|solutions|technologies|systems|services|ai|ml|data|analytics|science|warehouse)\b.*', '', company, flags=re.IGNORECASE)
        company = re.sub(r'\b(Expert|Engineer|Developer|Consultant|Manager|Lead)\b', '', company, flags=re.IGNORECASE)
        company = re.sub(r'\s{2,}', ' ', company).strip(" ,.-")

        # Filter out skill words pretending as company
        if company.lower() in skill_blacklist:
            company = ""

        # Clean role
        role = re.sub(r'^\d+\.\s*', '', role)
        role = re.sub(r'\b(for|at|in|on)\b\s+[A-Z].*', '', role, flags=re.IGNORECASE).strip(" ,.-")

        if company or role:
            entries.append({
                "role": role,
                "company": company,
                "start": start,
                "end": end,
                "confidence": conf
            })

    # Deduplicate
    unique_entries = {(e['role'], e['company']): e for e in entries}
    return list(unique_entries.values())


def choose_current_job(entries):
    """
    Chooses the most reliable job entry.
    Prefers PRESENT end dates, else highest confidence + recent pattern.
    """
    if not entries:
        return "", "", 0.0

    # Priority 1: end == PRESENT
    present_jobs = [e for e in entries if e.get("end") == "PRESENT"]
    if present_jobs:
        return max(present_jobs, key=lambda e: e["confidence"]).values()

    # Priority 2: latest dated entry
    dated = [e for e in entries if isinstance(e.get("start"), datetime)]
    if dated:
        latest = max(dated, key=lambda e: e["start"])
        return latest["role"], latest["company"], latest["confidence"]

    # Priority 3: fallback on confidence
    best = max(entries, key=lambda e: e["confidence"])
    return best["role"], best["company"], best["confidence"]



def extract_skills_spacy(text):
    if not (nlp and matcher):
        return "", 0.0
    doc = nlp(text)
    found = sorted({span.text for _, start, end in matcher(doc) for span in [doc[start:end]]})
    return ", ".join(found), 0.9 if found else 0.4

def extract_summary_smart(text):
    clean = re.sub(r"[•·\t]+", "\n", text.replace("\r", " "))
    m = re.search(r'(Professional Summary|Profile|Summary)\s*[:\n-]*([\s\S]{0,1500}?)(?=\n[A-Z][a-z]|Education|Experience|Projects|$)', clean, re.IGNORECASE)
    if m:
        block = re.sub(r"[\n\r]+", " ", m.group(2)).strip()
        return block[:800], 0.9
    if nlp:
        doc = nlp(clean[:1200])
        sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
        summary = " ".join(sents[:4])
        return summary[:800], 0.6
    return clean[:400], 0.4
# ----------------------------
# Company cleaning & normalization
# ----------------------------
DOMAIN_MAP = {
    "tcs": "Tata Consultancy Services",
    "wipro": "Wipro",
    "ibm": "IBM",
    "ms": "Microsoft",
    "microsoft": "Microsoft",
    "google": "Google",
    "amazon": "Amazon",
    "aws": "Amazon Web Services",
    "salesforce": "Salesforce",
    "accenture": "Accenture",
    # add more common mappings if you like
}

def clean_company(raw_company, full_text=""):
    """
    Normalize and sanitize company string. Removes URLs/emails,
    transforms domain-like strings to readable names,
    removes skill-like false positives.
    """
    if not raw_company:
        return ""

    s = raw_company.strip()

    # 1) Remove any embedded emails/urls that may have leaked in
    s = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', s, flags=re.IGNORECASE).strip()

    # 2) If the extraction accidentally captured trailing labels, remove them
    s = re.sub(r'\b(Job\s*Title|Experience|Email|Phone|Summary|Skills)\b.*', '', s, flags=re.IGNORECASE).strip(" ,.-")

    # 3) If string is mostly a single token with dots (domain-like) or ends with common tld, handle domain
    #    Examples: 'salesforce.com', 'selseforce.com', 'salesforce.co.in', 'salseforce'
    #    Strategy: strip tld, take first token, correct via map or title-case
    # If there's a dot and no spaces, treat as domain
    if '.' in s and ' ' not in s:
        token = s.split('/')[0]  # remove any path
        token = re.sub(r'^www\.', '', token, flags=re.IGNORECASE)
        token = re.sub(r'\.(com|co|in|io|net|org|biz|us|co\.in|co\.uk)$', '', token, flags=re.IGNORECASE)
        token = token.strip().lower()
        # If token is small (typo like 'selseforce'), try small fuzzy correction heuristics:
        token = token.replace('-', '').replace('_', '')
        # map known domains
        if token in DOMAIN_MAP:
            return DOMAIN_MAP[token]
        # handle obvious typos or near-matches heuristically (basic)
        # e.g., if token contains 'sales' or 'force' return 'Salesforce'
        if re.search(r'(sales.*force|force|sales)', token):
            return "Salesforce"
        # fallback: title-case the token (Salesforce, ByteScale, Nextiq)
        return token.title()

    # 4) Remove any stray tld suffix from multi-word string
    s = re.sub(r'\.(com|co|in|io|net|org|biz|us|co\.in|co\.uk)\b', '', s, flags=re.IGNORECASE).strip()

    # 5) Remove phone-looking or email-like junk again (defensive)
    s = re.sub(r'[\w\.-]+@[\w\.-]+', '', s).strip()
    s = re.sub(r'https?://\S+|www\.\S+', '', s).strip()

    # 6) If the cleaned company is actually a technical skill or very short and not business-like, drop it
    skill_blacklist = set(w.lower() for w in SKILLS_DB)
    # If any token in company equals a skill (or company is single short token from skills), discard
    tokens = [t.strip().lower() for t in re.split(r'[\s,/&\-]+', s) if t.strip()]
    if any(t in skill_blacklist for t in tokens):
        return ""

    # 7) If result is a single uppercase token of 2-6 letters, consider mapping to common acronyms (TCS etc.)
    if len(tokens) == 1 and re.fullmatch(r'[A-Z]{2,6}', tokens[0] if isinstance(tokens[0], str) else ""):
        key = tokens[0].lower()
        if key in DOMAIN_MAP:
            return DOMAIN_MAP[key]
        # otherwise return as-is (e.g., 'TCS')
        return tokens[0].upper()

    # 8) Final cleanup: remove trailing words that are generic noise
    s = re.sub(r'\b(inc|llc|ltd|limited|private|services|solutions|technologies|systems|consultancy)\b\.?$', '', s, flags=re.IGNORECASE).strip(" ,.-")
    # title-case appropriate multi-word company names
    return " ".join([w.capitalize() if len(w) > 1 else w.upper() for w in s.split()]).strip()

# ======================================================
# Full Parser
# ======================================================
def parse_resume(text, source_file=""):
    # Name
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name_line = next((l for l in lines if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+", l)), "Unknown Unknown")
    parts = name_line.split()
    first, last = parts[0], parts[-1]
    full = " ".join(parts)

    # === Direct label-based company and job title extraction ===
    # === Direct label-based extraction (non-greedy, stops at next label) ===
    company_label = re.search(
        r'Company\s*[:\-]\s*([A-Z][A-Za-z0-9&\.\s\-]+?)(?=\s*(?:Job\s*Title|Experience|Email|Phone|$))',
        text, re.IGNORECASE
    )
    job_label = re.search(
        r'(Job\s*Title|Designation)\s*[:\-]\s*([A-Za-z0-9&\.\s\-]+?)(?=\s*(?:Experience|Email|Phone|Company|$))',
        text, re.IGNORECASE
    )

    company_direct = company_label.group(1).strip() if company_label else ""
    job_direct = job_label.group(2).strip() if job_label else ""

    # Experience parsing
    entries = parse_experience_entries(text)
    job_title, company, conf_job = choose_current_job(entries)

    # ✅ Prefer direct values when available
    if company_direct:
        company = company_direct
        conf_job = 0.95
    if job_direct:
        job_title = job_direct
        conf_job = 0.95
    if not company:
        # --- Pattern 1: Common "at / with / for" phrasing ---
        m = re.search(r"\b(?:at|with|for)\s+([A-Z][A-Za-z&\s]{2,60})", text)
        if m:
            company = m.group(1).strip(" ,.-")

    # --- Pattern 2: Role followed by dash and company (e.g., 'Assistant Consultant – Tata Consultancy Services') ---
    if not company:
        m = re.search(
            r"(?:Consultant|Engineer|Developer|Manager|Architect|Analyst|Lead|Expert|Specialist)[^.\n\r]{0,40}[-–—:|]\s*([A-Z][A-Za-z&\s]{2,60})",
            text
        )
        if m:
            company = m.group(1).strip(" ,.-")

    # --- Pattern 3: Company before role (e.g., 'Tata Consultancy Services - Assistant Consultant') ---
    if not company:
        m = re.search(
            r"([A-Z][A-Za-z&\s]{2,60})\s*[-–—:|]\s*(?:Assistant|Senior|Lead|Data|DevOps|Software|Consultant|Engineer|Developer|Manager|Architect|Analyst|Specialist)",
            text
        )
        if m:
            company = m.group(1).strip(" ,.-")

    # --- Pattern 4: Short uppercase companies (TCS, IBM, WIPRO, etc.) ---
    if not company:
        m = re.search(r"\b([A-Z]{2,10})\b", text)
        if m and m.group(1).lower() not in [s.lower() for s in SKILLS_DB]:
            company = m.group(1).strip()

    if not company:
        # Look for contextual company mentions inside summary or experience paragraphs
        exp_section = ""
        m = re.search(r"(?:Professional Summary|Experience|Work Experience)\s*[:\n-]*([\s\S]{0,800})", text,
                      re.IGNORECASE)
        if m:
            exp_section = m.group(1)

        # Detect patterns implying current employment
        patterns = [
            r"(?:currently|presently)\s+(?:working|employed)\s+(?:at|with|for|in)\s+([A-Z][A-Za-z&\s]{2,60})",
            r"(?:joined|serving)\s+(?:at|with|for|in)\s+([A-Z][A-Za-z&\s]{2,60})",
            r"(?:working|worked)\s+(?:at|for|with|in)\s+([A-Z][A-Za-z&\s]{2,60})",
            r"employed\s+(?:with|at|by)\s+([A-Z][A-Za-z&\s]{2,60})",
        ]

        for pat in patterns:
            mm = re.search(pat, exp_section, re.IGNORECASE)
            if mm:
                company = mm.group(1).strip(" ,.-")
                break

    company = clean_company(company, text)

    # Skills
    skills, conf_skills = extract_skills_spacy(text)
    # Summary
    summary, conf_sum = extract_summary_smart(text)
    # Experience years
    exp_match = re.search(r"(\d{1,2})\s*(?:\+)?\s*(?:years?|yrs?)", text, re.IGNORECASE)
    exp_years = int(exp_match.group(1)) if exp_match else 0
    # Contact
    email = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    phone = re.search(r"\+?\d[\d \-\(\)]{7,}\d", text)
    email, phone = email.group(0) if email else "", phone.group(0) if phone else ""
    company = re.sub(r"\b(for|labs|technologies|systems|solutions|services)\b.*", "", company,
                     flags=re.IGNORECASE).strip(" ,.-")
    job_title = re.sub(r"^\d+\.\s*", "", job_title).strip(" ,.-")
    # Normalize and clean results
    company = re.sub(r'\b(for|labs|solutions|technologies|systems|ai|ml|data|analytics)\b.*', '', company,
                     flags=re.IGNORECASE).strip(" ,.-")
    job_title = re.sub(r'^\d+\.\s*', '', job_title).strip(" ,.-")
    if company.lower() in [s.lower() for s in SKILLS_DB]:
        company = ""

    return {
        "first_name": first, "last_name": last, "full_name": full,
        "job_title": job_title, "company": company, "skills": skills,
        "experience_years": exp_years, "email": email, "phone": phone,
        "summary": summary, "experience": "",
        "job_title_conf": conf_job, "company_conf": conf_job,
        "skills_conf": conf_skills, "summary_conf": conf_sum,
        "source_file": source_file
    }

# ======================================================
# Simple Chat Agent
# ======================================================
def query_chat_agent_rule(db, question):
    import matplotlib.pyplot as plt
    import pandas as pd
    import re
    import streamlit as st

    q = question.lower().strip()
    cur = db.connection.cursor(dictionary=True)

    # --- 🧠 Detect multiple skills ---
    matched_skills = [s for s in SKILLS_DB if len(s) > 1 and s.lower() in q]
    has_and = " and " in q
    has_or = " or " in q
    skill_logic = "AND" if has_and else "OR"
    skill_filter = ""
    if matched_skills:
        skill_filter = f" {skill_logic} ".join(["skills LIKE %s" for _ in matched_skills])

    # --- ⚙️ Detect experience conditions ---
    exp_pattern = re.search(
        r"(?:experience|exp)\s*(?:>=|<=|>|<|=|between)?\s*([\d]+)(?:\s*(?:and|to|-)\s*(\d+))?",
        q,
    )
    exp_clause = ""
    exp_params = []
    desc = ""
    if exp_pattern:
        num1 = int(exp_pattern.group(1))
        num2 = exp_pattern.group(2)
        if ">=" in q:
            exp_clause, desc = "experience_years >= %s", f"at least {num1}"
            exp_params.append(num1)
        elif "<=" in q:
            exp_clause, desc = "experience_years <= %s", f"at most {num1}"
            exp_params.append(num1)
        elif ">" in q:
            exp_clause, desc = "experience_years > %s", f"more than {num1}"
            exp_params.append(num1)
        elif "<" in q:
            exp_clause, desc = "experience_years < %s", f"less than {num1}"
            exp_params.append(num1)
        elif "=" in q:
            exp_clause, desc = "experience_years = %s", f"exactly {num1}"
            exp_params.append(num1)
        elif "between" in q or "to" in q or "-" in q:
            exp_clause, desc = "experience_years BETWEEN %s AND %s", f"between {num1} and {num2}"
            exp_params.extend([num1, int(num2)])
        else:
            exp_clause, desc = "experience_years > %s", f"more than {num1}"
            exp_params.append(num1)

    # === 🏢 Company + Experience (+ optional Skill) Filtering ===
    if re.search(r"(list|show|find|display|employees|people|who)", q) and re.search(
        r"(company|tcs|infosys|wipro|accenture|bytescale|zen|dataverse|nextiq|quantedge|tech|solutions|consultancy)",
        q,
    ):
        company_match = re.search(
            r"(?:from|at|in|company)\s+([A-Za-z&\.\- ]{2,60}?)(?=\s+(?:with|having|where|who|exp|experience|years|and|,|$))",
            q,
            re.IGNORECASE,
        )
        company = ""
        if company_match:
            company = company_match.group(1).strip()
            company = re.sub(
                r"\b(list|show|display|people|employees|who|with|having|in|at|the|company)\b",
                "",
                company,
                flags=re.IGNORECASE,
            ).strip()

        company_aliases = {
            "tata consultancy": "Tata Consultancy",
            "tata consultancy services": "Tata Consultancy Services",
            "tcs": "Tata Consultancy Services",
            "bytescale it": "ByteScale IT",
            "dataverse": "DataVerse",
            "nextiq": "NextIQ Software",
            "quantedge": "QuantEdge Innovations",
        }
        for key, val in company_aliases.items():
            if key in company.lower():
                company = val
                break

        where_clauses = []
        params = []
        if company:
            where_clauses.append("company LIKE %s")
            params.append(f"%{company}%")
        if exp_clause:
            where_clauses.append(exp_clause)
            params.extend(exp_params)
        if skill_filter:
            where_clauses.append(f"({skill_filter})")
            params.extend([f"%{s}%" for s in matched_skills])

        query = f"""
            SELECT full_name, company, skills, experience_years
            FROM employees
            WHERE {' AND '.join(where_clauses) if where_clauses else '1=1'}
            ORDER BY experience_years DESC
        """
        cur.execute(query, tuple(params))
        rows = cur.fetchall()

        if not rows:
            return f"🤖 No employees found for company '{company}' with {desc} years of experience."

        df = pd.DataFrame(rows)
        st.dataframe(df)
        if matched_skills:
            return f"📋 Employees skilled in {', '.join(matched_skills)} from {company.upper()} with {desc} of experience."
        else:
            return f"📋 Employees from {company.upper()} with {desc} of experience."

    # --- 📊 Chart Detection ---
    if re.search(r"(chart|plot|graph|visual|bar|pie|trend)", q):

        # === Skill Chart ===
        if "skill" in q:
            query = """
                SELECT skill, COUNT(*) AS freq
                FROM (
                    SELECT TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(skills, ',', n.n), ',', -1)) AS skill
                    FROM employees
                    JOIN (
                        SELECT a.N + b.N * 10 + 1 AS n
                        FROM (SELECT 0 AS N UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
                              UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) a,
                             (SELECT 0 AS N UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
                              UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) b
                    ) n
                    WHERE n.n <= 1 + LENGTH(skills) - LENGTH(REPLACE(skills, ',', ''))
                ) AS derived
                WHERE skill <> ''
                GROUP BY skill
                ORDER BY freq DESC
                LIMIT 10
            """
            cur.execute(query)
            rows = cur.fetchall()
            if not rows:
                return "No data available to plot."

            skills = [r["skill"] for r in rows]
            freqs = [r["freq"] for r in rows]

            plt.figure(figsize=(8, 4))
            if "pie" in q:
                plt.pie(freqs, labels=skills, autopct="%1.1f%%")
                plt.title("Skill Distribution (Top 10)")
            elif "trend" in q or "line" in q:
                plt.plot(skills, freqs, marker="o")
                plt.title("Skill Trend (Top 10)")
                plt.ylabel("Profiles")
                plt.xlabel("Skill")
            else:
                plt.barh(skills[::-1], freqs[::-1])
                plt.title("Top 10 Skills in Database")
                plt.xlabel("Profiles")
                plt.ylabel("Skill")
            st.pyplot(plt)
            return "📊 Skill distribution chart displayed."

        # === Company Chart ===
        if "company" in q:
            query = """
                SELECT company, COUNT(*) AS total
                FROM employees
                WHERE company <> ''
                GROUP BY company
                ORDER BY total DESC
                LIMIT 10
            """
            cur.execute(query)
            rows = cur.fetchall()
            if not rows:
                return "No company data to plot."

            companies = [r["company"] for r in rows]
            counts = [r["total"] for r in rows]

            plt.figure(figsize=(8, 4))
            if "pie" in q:
                plt.pie(counts, labels=companies, autopct="%1.1f%%")
                plt.title("Company Distribution (Top 10)")
            elif "trend" in q or "line" in q:
                plt.plot(companies, counts, marker="o")
                plt.title("Company Profile Trend")
                plt.ylabel("Profiles")
                plt.xlabel("Company")
            else:
                plt.barh(companies[::-1], counts[::-1])
                plt.title("Top 10 Companies by Profiles")
                plt.xlabel("Profiles")
                plt.ylabel("Company")
            st.pyplot(plt)
            return "🏢 Company chart displayed."

        # === Average Experience Chart ===
        if "experience" in q:
            query = """
                SELECT company, ROUND(AVG(experience_years),1) AS avg_exp
                FROM employees
                WHERE company <> ''
                GROUP BY company
                HAVING COUNT(*) > 1
                ORDER BY avg_exp DESC
                LIMIT 10
            """
            cur.execute(query)
            rows = cur.fetchall()
            if not rows:
                return "No experience data to plot."

            companies = [r["company"] for r in rows]
            avg_exp = [r["avg_exp"] for r in rows]

            plt.figure(figsize=(8, 4))
            if "pie" in q:
                plt.pie(avg_exp, labels=companies, autopct="%1.1f%%")
                plt.title("Average Experience by Company")
            elif "trend" in q or "line" in q:
                plt.plot(companies, avg_exp, marker="o")
                plt.title("Experience Trend by Company")
                plt.ylabel("Average Experience (Years)")
                plt.xlabel("Company")
            else:
                plt.barh(companies[::-1], avg_exp[::-1])
                plt.title("Top 10 Companies by Average Experience")
                plt.xlabel("Average Experience (Years)")
                plt.ylabel("Company")
            st.pyplot(plt)
            return "📈 Average experience chart displayed."

    # --- 👩‍💻 NEW: Skill-Only Query ---
    if re.search(r"(list|show|find|display|who|people|developers|employees)", q) and matched_skills and not re.search(r"(company|experience|exp|years)", q):
        query = f"""
            SELECT full_name, company, experience_years
            FROM employees
            WHERE {skill_filter}
            ORDER BY experience_years DESC
            LIMIT 20
        """
        cur.execute(query, tuple(f"%{s}%" for s in matched_skills))
        rows = cur.fetchall()
        if not rows:
            return f"🤖 No employees found skilled in {', '.join(matched_skills)}."

        df = pd.DataFrame(rows)
        st.dataframe(df)
        if len(matched_skills) > 1 and has_and:
            return f"📋 Employees skilled in all of ({', '.join(matched_skills)}) shown above."
        elif len(matched_skills) > 1:
            return f"📋 Employees skilled in any of ({', '.join(matched_skills)}) shown above."
        else:
            return f"📋 Here are employees skilled in {', '.join(matched_skills)}."

    # --- 🧮 Top Companies by Skill ---
    # --- 🏢 Top Companies by Skill ---
    if re.search(r"(company|companies|organization|organizations)", q) and (
        re.search(r"(most|top|best|leading|popular|for|in)", q)
    ):
        if matched_skills:
            query = f"""
                SELECT company, COUNT(*) AS total, ROUND(AVG(experience_years),1) AS avg_exp
                FROM employees
                WHERE ({skill_filter}) AND company <> ''
                GROUP BY company
                ORDER BY total DESC
                LIMIT 10
            """
            cur.execute(query, tuple(f"%{s}%" for s in matched_skills))
            rows = cur.fetchall()
            if not rows:
                return f"🤖 No companies found for {', '.join(matched_skills)}."

            df = pd.DataFrame(rows)
            st.dataframe(df)
            lines = [f"- {r['company']} ({r['total']} profiles, avg {r['avg_exp']} yrs exp)" for r in rows]
            return f"🏢 Top companies for {', '.join(matched_skills)}:\n" + "\n".join(lines)


    # --- 🧩 Average Experience Analytics ---
    if "average" in q and ("company" in q or "organization" in q):
        query = """
            SELECT company, ROUND(AVG(experience_years),1) AS avg_exp, COUNT(*) AS count
            FROM employees
            WHERE company <> ''
            GROUP BY company
            HAVING COUNT(*) > 1
            ORDER BY avg_exp DESC
            LIMIT 10
        """
        cur.execute(query)
        rows = cur.fetchall()
        if rows:
            lines = [f"- {r['company']} ({r['avg_exp']} yrs avg, {r['count']} profiles)" for r in rows]
            return "📊 Top companies by average experience:\n" + "\n".join(lines)
        else:
            return "No company experience data found."

    # --- 🧠 Skill Distribution Analytics ---
    if re.search(r"(skill|skills).*(most|top|popular|common|distribution)", q):
        query = """
            SELECT skill, COUNT(*) AS freq
            FROM (
                SELECT TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(skills, ',', n.n), ',', -1)) AS skill
                FROM employees
                JOIN (
                    SELECT a.N + b.N * 10 + 1 AS n
                    FROM (SELECT 0 AS N UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
                          UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) a,
                         (SELECT 0 AS N UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
                          UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) b
                ) n
                WHERE n.n <= 1 + LENGTH(skills) - LENGTH(REPLACE(skills, ',', ''))
            ) AS derived
            WHERE skill <> ''
            GROUP BY skill
            ORDER BY freq DESC
            LIMIT 15
        """
        cur.execute(query)
        rows = cur.fetchall()
        if rows:
            lines = [f"- {r['skill']} ({r['freq']} profiles)" for r in rows]
            return "🧠 Most common skills:\n" + "\n".join(lines)
        else:
            return "No skill data available."
    # --- 👨‍💻 Experience-Only Queries (no skill or company mentioned) ---
    if re.search(r"(experience|exp)", q) and not matched_skills and not re.search(r"(company|organization|firm|enterprise)", q):
        if exp_clause:
            query = f"""
                SELECT full_name, company, experience_years
                FROM employees
                WHERE {exp_clause}
                ORDER BY experience_years DESC
                LIMIT 25
            """
            cur.execute(query, tuple(exp_params))
            rows = cur.fetchall()

            if not rows:
                return f"🤖 No employees found with {desc} experience."

            df = pd.DataFrame(rows)
            st.dataframe(df)

            count = len(rows)
            avg_exp = sum([r["experience_years"] for r in rows]) / count if count else 0

            # --- 📊 Add Experience Distribution Chart ---
            plt.figure(figsize=(6, 3))
            plt.hist([r["experience_years"] for r in rows], bins=6, edgecolor="black")
            plt.title(f"Experience Distribution ({desc})")
            plt.xlabel("Experience (Years)")
            plt.ylabel("Number of Employees")
            st.pyplot(plt)

            return (
                f"📋 Found **{count} employees** with {desc} experience "
                f"(average experience: {avg_exp:.1f} years)."
            )
        else:
            return "⚙️ Please specify an experience filter (e.g., '> 5 years', '< 10 years', 'between 3 and 8 years')."
    # --- 📊 Count Queries (e.g. "how many resumes", "how many employees from TCS") ---
    if re.search(r"how many|count|number of", q):
        where_clauses = []
        params = []

        # Skill filter
        if matched_skills:
            where_clauses.append(f"({skill_filter})")
            params.extend([f"%{s}%" for s in matched_skills])

        # Company filter
        company_match = re.search(
            r"(?:from|at|in|company)\s+([A-Za-z&\.\- ]{2,60})",
            q,
            re.IGNORECASE,
        )
        company = ""
        if company_match:
            company = company_match.group(1).strip()
            company_aliases = {
                "tata consultancy": "Tata Consultancy",
                "tata consultancy services": "Tata Consultancy Services",
                "tcs": "Tata Consultancy Services",
                "bytescale it": "ByteScale IT",
                "dataverse": "DataVerse",
                "nextiq": "NextIQ Software",
                "quantedge": "QuantEdge Innovations",
            }
            for key, val in company_aliases.items():
                if key in company.lower():
                    company = val
                    break
            where_clauses.append("company LIKE %s")
            params.append(f"%{company}%")

        # Experience filter
        if exp_clause:
            where_clauses.append(exp_clause)
            params.extend(exp_params)

        # Build query
        query = f"""
            SELECT COUNT(*) AS total
            FROM employees
            WHERE {' AND '.join(where_clauses) if where_clauses else '1=1'}
        """
        cur.execute(query, tuple(params))
        result = cur.fetchone()
        total = result["total"] if result else 0

        # Build response message
        parts = []
        if matched_skills:
            parts.append(f"skilled in {', '.join(matched_skills)}")
        if company:
            parts.append(f"from {company}")
        if desc:
            parts.append(f"with {desc} experience")

        filter_text = " ".join(parts) if parts else ""
        return f"📄 There are **{total} resumes** {filter_text.strip()} in the database."

    # --- Default Fallback ---
    return "🤖 I couldn’t understand that question. Try asking: 'list employees from company TCS with exp > 5', 'show skill distribution chart', or 'average experience per company'."






# ======================================================
# Streamlit UI
# ======================================================
st.title("📄 Resume Parser + Enhanced LLM-SQL Chat Agent")

st.sidebar.header("⚙️ Database Settings")
db_host = st.sidebar.text_input("MySQL Host", "localhost")
db_user = st.sidebar.text_input("MySQL User", "root")
db_password = st.sidebar.text_input("MySQL Password", type="password")
db_name = st.sidebar.text_input("Database", "employee_db")
resume_folder = st.sidebar.text_input("C:/Users/iamw0.W0KIE/PycharmProjects/PythonProject/chatbot/", "./synthetic_resumes")

db = ResumeDatabase(host=db_host, user=db_user, password=db_password, database=db_name) if db_password else None
tabs = st.tabs(["📂 Parse Folder", "🗂️ Database", "💬 Chat Agent"])

# === Parse Folder Tab ===
with tabs[0]:
    st.subheader("📁 Parse all resumes from folder")
    if st.button("Start Parsing Folder"):
        if not os.path.exists(resume_folder):
            st.error(f"Invalid folder path: {resume_folder}")
        else:
            files = [f for f in os.listdir(resume_folder) if f.lower().endswith((".pdf", ".docx"))]
            if not files:
                st.warning("No resumes found.")
            else:
                for file in files:
                    path = os.path.join(resume_folder, file)
                    st.info(f"Parsing {file} ...")
                    text = extract_text_from_docx(path) if file.endswith(".docx") else extract_text_from_pdf_bytes(open(path, "rb").read())
                    rec = parse_resume(text, file)
                    st.json(rec)
                    if db:
                        db.insert_employee(rec)
                st.success("✅ Completed parsing all resumes!")

# === Database Tab ===
with tabs[1]:
    if db and db.connection:
        rows = db.list_employees()
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)
            st.download_button("⬇️ Download CSV", df.to_csv(index=False), "employees.csv", "text/csv")
        else:
            st.info("No employee records found.")
    else:
        st.warning("Connect to DB first.")

# === Chat Agent Tab ===
with tabs[2]:
    import re
    import pandas as pd
    import matplotlib.pyplot as plt

    st.subheader("💬 Resume Chat Agent")

    # 1️⃣ Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []        # list of (role, content)
    if "last_result" not in st.session_state:
        st.session_state.last_result = None       # stores last query result (table/chart/text)

    # 2️⃣ Display existing chat history
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # 3️⃣ Chat input — always rendered last, never inside containers
    user_query = st.chat_input("Ask something... (e.g., 'list people from TCS with >5 years exp')")

    # 4️⃣ Process user query
    if user_query:
        # --- show user message immediately ---
        with st.chat_message("user"):
            st.markdown(user_query)

        # --- run your rule-based logic ---
        reply = query_chat_agent_rule(db, user_query)

        # --- clear last result safely ---
        st.session_state.last_result = None

        # --- display assistant reply ---
        with st.chat_message("assistant"):
            st.markdown("🔍 Processing your query...")
            st.divider()
            st.markdown(reply)

        # --- append conversation ---
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("assistant", reply))

        # --- render structured output (optional) ---
        matched_skills = [s for s in SKILLS_DB if len(s) > 1 and s.lower() in user_query.lower()]
        company_match = re.search(r"(?:from|at|in|company)\s+([A-Za-z&.\- ]{2,60})", user_query, re.IGNORECASE)
        company = company_match.group(1).strip() if company_match else ""
        exp_match = re.search(r"(\d{1,2})\s*(?:\+)?\s*(?:years?|yrs?)", user_query, re.IGNORECASE)
        exp_val = exp_match.group(1) if exp_match else ""

        with st.expander("🧠 Interpreted Query", expanded=False):
            st.write(f"- **Skill:** {', '.join(matched_skills) or 'Not detected'}")
            st.write(f"- **Company:** {company or 'Not detected'}")
            st.write(f"- **Experience:** {exp_val + ' years' if exp_val else 'Not specified'}")

    # 5️⃣ Utility buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.last_result = None
            st.rerun()
    with col2:
        st.caption("Chat is local and rule-based — no external API used.")

if db:
    db.connection.close()
