import os
import random
import pandas as pd
from faker import Faker
from fpdf import FPDF

fake = Faker()
Faker.seed(42)

# =========================================
# CONFIGURATION
# =========================================
OUTPUT_DIR = "synthetic_resumes"
N_RESUMES = 40  # change this for more

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample skill sets by domain
DOMAINS = {
    "Data Science": ["Python", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "Keras", "Machine Learning", "Deep Learning", "Statistics"],
    "DevOps": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Jenkins", "Git", "CI/CD"],
    "Data Engineering": ["Python", "PySpark", "Airflow", "Informatica", "Snowflake", "ETL", "SQL", "Data Warehouse"],
    "Java": ["Java", "Spring Boot", "Microservices", "REST API", "Hibernate", "MySQL", "Kafka", "Maven", "JUnit"],
    "BI / Analytics": ["SQL", "Power BI", "Tableau", "Excel", "Snowflake", "Data Modeling", "Reporting", "Visualization"]
}

# Generate random fake company names
COMPANIES = [
    "CloudEdge Analytics", "TechNova Systems", "DataVerse Labs", "AIverse Solutions",
    "CodeCraft Technologies", "NeuralBridge AI", "QuantEdge Innovations", "InfoNexus Global",
    "ByteScale IT", "DeepMatrix Systems", "VirtuSpark Labs", "FinCore Technologies",
    "NextIQ Software", "BlueLogic IT", "ZenData Solutions"
]

# =========================================
# Helper function to create fake resumes
# =========================================
def generate_resume(i):
    role_domain = random.choice(list(DOMAINS.keys()))
    skills = random.sample(DOMAINS[role_domain], k=min(6, len(DOMAINS[role_domain])))
    company = random.choice(COMPANIES)
    full_name = fake.name()
    email = fake.user_name() + "@" + company.replace(" ", "").lower() + ".com"
    phone = fake.phone_number()
    years = random.randint(2, 15)

    job_title = f"{random.choice(['Senior', 'Lead', 'Principal', 'Associate', ''])} {role_domain} Engineer".replace("  ", " ").strip()
    summary = (
        f"{full_name} is a {job_title} with over {years} years of professional experience. "
        f"Expert in {', '.join(skills[:4])}, and experienced in designing and implementing scalable solutions "
        f"for {company}'s clients. Passionate about automation, optimization, and data-driven innovation."
    )

    # Create a simple structured resume (PDF)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, full_name, ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Email: {email} | Phone: {phone}", ln=True)
    pdf.cell(0, 10, f"Company: {company}", ln=True)
    pdf.cell(0, 10, f"Job Title: {job_title}", ln=True)
    pdf.cell(0, 10, f"Experience: {years} years", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Professional Summary", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, summary)
    pdf.ln(6)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Technical Skills", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, ", ".join(skills))
    pdf.ln(6)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Experience Summary", ln=True)
    pdf.set_font("Arial", '', 12)
    for y in range(random.randint(2, 4)):
        exp_company = random.choice(COMPANIES)
        exp_role = f"{random.choice(['ETL Developer', 'Data Engineer', 'ML Engineer', 'DevOps Specialist', 'Java Developer', 'BI Analyst'])}"
        pdf.multi_cell(0, 8, f"- {exp_role} at {exp_company} for {random.randint(1,5)} years, focusing on {random.choice(skills)}.")
    pdf.ln(6)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Education", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, f"Bachelor of Technology in Computer Science, {fake.company()} University")

    # Save PDF
    filename = f"Resume_{i+1:02d}_{full_name.replace(' ', '_')}.pdf"
    filepath = os.path.join(OUTPUT_DIR, filename)
    pdf.output(filepath)

    # Return metadata
    return {
        "file_name": filename,
        "full_name": full_name,
        "role": job_title,
        "company": company,
        "experience_years": years,
        "email": email,
        "phone": phone,
        "skills": ", ".join(skills)
    }

# =========================================
# Main generation loop
# =========================================
metadata = []
for i in range(N_RESUMES):
    rec = generate_resume(i)
    metadata.append(rec)

# Save metadata CSV
df = pd.DataFrame(metadata)
df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
print(f"✅ Generated {N_RESUMES} resumes in '{OUTPUT_DIR}/' with metadata.csv")
