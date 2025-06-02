import streamlit as st
st.set_page_config(page_title="Chest X-ray Report Generator", layout="centered")

from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
import os
import re
import pandas as pd
from datetime import datetime
from utils.dino_embedding import load_rad_dino, extract_rad_dino_prompt

# === CONFIG ===
MODEL_DIR = "models/clinical_t5_final"
REPORT_DIR = "outputs/reports"
LOGO_PATH = "assets/logo.png"
SIGN_PATH = "assets/signature.png"
LOG_CSV = "outputs/report_logs.csv"
os.makedirs(REPORT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load models ===
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    processor, dino_model = load_rad_dino()
    return tokenizer, model, processor, dino_model

tokenizer, model, processor, dino_model = load_models()

# === Post-process report for grammar and readability ===
def clean_findings(text):
    lines = text.strip().split("\n")
    seen = set()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Capitalize and punctuate
        line = line[0].upper() + line[1:] if line else ""
        if not line.endswith("."):
            line += "."

        # Remove duplicates
        line_key = re.sub(r"[^\w\s]", "", line.lower())
        if line_key in seen:
            continue
        seen.add(line_key)

        # Bullet point
        cleaned.append(f"• {line}")

    return "\n".join(cleaned)

# === Generate Report ===
def generate_report(image: Image.Image) -> str:
    prompt = extract_rad_dino_prompt(image, processor, dino_model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_findings(raw_output)

# === Log Report to CSV ===
def log_report(metadata, file_path):
    df = pd.DataFrame([{
        "Patient Name": metadata["name"],
        "Referred By": metadata["ref_by"],
        "Date Taken": metadata["date_taken"],
        "Date of Report": metadata["date_report"],
        "Complaint": metadata.get("complaint", ""),
        "History": metadata.get("history", ""),
        "PDF Path": file_path,
        "Generated At": str(datetime.now())
    }])

    df.to_csv(LOG_CSV, mode="a", index=False, header=not os.path.exists(LOG_CSV))

# === Create Letterhead PDF ===
def create_letterhead_pdf(report_text, image: Image.Image, filename: str, meta: dict) -> str:
    pdf_path = os.path.join(REPORT_DIR, f"{filename}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    styles = getSampleStyleSheet()

    def draw_header():
        c.setFillColorRGB(0.2, 0.4, 0.6)
        c.rect(0, height - 80, width, 80, fill=1)
        c.setFillColorRGB(1, 1, 1)
        text_x = 100
        c.setFont("Helvetica-Bold", 16)
        c.drawString(text_x, height - 50, "DIAGNOSTIC X-RAY CONSULTATION SERVICES®")
        c.setFont("Helvetica", 11)
        c.drawString(text_x, height - 68, "2525 W. Carefree Highway, Suite 114, Phoenix, AZ")
        c.drawString(400, height - 68, "Phone: (602) 274-3331")
        if os.path.exists(LOGO_PATH):
            c.drawImage(LOGO_PATH, 40, height - 75, width=40, height=40)

    def draw_footer():
        c.setFillColorRGB(0.85, 0.85, 0.85)
        c.rect(0, 0, width, 50, fill=1)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 9)
        c.drawRightString(width - 40, 30, "Email: glongmuir@diagnosticx-ray.com")

    def draw_signature():
        y_offset = 90
        if os.path.exists(SIGN_PATH):
            c.drawImage(SIGN_PATH, width - 180, y_offset, width=120, height=50, mask='auto')
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(width - 50, y_offset - 10, "Dr. GARY A. LONGMUIR")
        c.setFont("Helvetica-Oblique", 9)
        c.drawRightString(width - 50, y_offset - 22, "D.C., Ph.D., D.A.C.B.R.")

    # === Page 1 ===
    draw_header()
    draw_footer()
    draw_signature()

    frame = Frame(50, 100, width - 100, height - 200, showBoundary=0)
    content = []
    meta_style = ParagraphStyle("Meta", fontSize=13, spaceAfter=10)
    italic_style = ParagraphStyle("Italic", fontSize=13, spaceAfter=10, fontName="Helvetica-Oblique")
    heading_style = ParagraphStyle("Heading", fontSize=14, spaceAfter=12, fontName="Helvetica-Bold")

    def add_field(label, value, style=meta_style):
        content.append(Paragraph(f"<b>{label}:</b> {value}", style))

    add_field("Patient’s Name", meta["name"])
    add_field("Referred by", meta["ref_by"])
    add_field("Date Taken", meta["date_taken"])
    add_field("Date of Report", meta["date_report"])
    add_field("Patient’s Complaint", meta["complaint"], italic_style)
    add_field("Patient’s History", meta["history"], italic_style)
    content.append(Paragraph("Findings:", heading_style))
    content.append(Paragraph(report_text.replace("\n", "<br/>"), meta_style))
    frame.addFromList(content, c)

    # === Page 2 ===
    c.showPage()
    draw_header()
    draw_footer()
    draw_signature()

    # Draw X-ray image
    image_path = os.path.join(REPORT_DIR, f"{filename}_img.jpg")
    image.save(image_path)
    img_w, img_h = 5.8 * inch, 5.8 * inch
    img_x = (width - img_w) / 2
    img_y = 230

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, img_y + img_h + 15, "X-ray Image:")
    c.drawImage(image_path, img_x, img_y, width=img_w, height=img_h)

    c.save()
    return pdf_path

# === Streamlit UI ===
st.title("🩻 Chest X-ray Report Generator")
st.write("Upload a chest X-ray image and fill in patient details to generate a medically formatted PDF report.")

with st.form("report_form"):
    st.subheader("🧾 Patient Details")
    name = st.text_input("Patient’s Name")
    ref_by = st.text_input("Referred by")
    date_taken = st.date_input("Date Taken")
    date_report = st.date_input("Date of Report")
    complaint = st.text_area("Patient’s Complaint")
    history = st.text_area("Patient’s History")
    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])
    submitted = st.form_submit_button("🧠 Generate Report")

if submitted and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    with st.spinner("Analyzing image and generating report..."):
        report = generate_report(image)
        st.subheader("Generated Findings")
        st.text(report)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_data = {
            "name": name,
            "ref_by": ref_by,
            "date_taken": str(date_taken),
            "date_report": str(date_report),
            "complaint": complaint,
            "history": history,
        }

        pdf_file = create_letterhead_pdf(report, image, f"report_{timestamp}", meta=meta_data)

    
        log_report(meta_data, pdf_file)

        with open(pdf_file, "rb") as f:
            st.download_button("Download Report as PDF", f, file_name=os.path.basename(pdf_file))
