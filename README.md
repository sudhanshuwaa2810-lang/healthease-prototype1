# app.py - HealthEase (Render-ready single-file backend)
# Features:
# - Signup/login (patient / doctor) with session
# - Upload docs (PDF/JPG/PNG). OCR: PyMuPDF for PDFs, pytesseract for images (if available)
# - OpenAI summarization hook (if OPENAI_API_KEY set)
# - Patient can share a doc with a doctor (by doctor's username)
# - Symptom checker (simple rule-based + OpenAI optional)
# - /seed endpoint to create demo accounts
#
# NOTE: This is a prototype for testing. Do NOT use as-is in production for sensitive data.

import os
import io
import uuid
import datetime
import sqlite3
import functools
import json
from pathlib import Path
from flask import (
    Flask, request, redirect, url_for, render_template_string,
    session, send_from_directory, flash, jsonify
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image

# Optional libs
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    import openai
except Exception:
    openai = None

# ---------- Config ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DOCS_DIR = os.path.join(UPLOAD_DIR, "docs")
PROFILE_DIR = os.path.join(UPLOAD_DIR, "profile_pics")
DB_PATH = os.path.join(BASE_DIR, "healthease.db")
ALLOWED = {"pdf", "png", "jpg", "jpeg"}

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(PROFILE_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("HEALTHEASE_SECRET", "change_me_in_prod")
app.config.update(MAX_CONTENT_LENGTH=60 * 1024 * 1024)

# ---------- DB helpers ----------
def get_db():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        password_hash TEXT,
        role TEXT,
        created_at TIMESTAMP
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        owner_id TEXT,
        filename TEXT,
        original_name TEXT,
        uploaded_at TIMESTAMP,
        shared_with TEXT,
        ocr_text TEXT,
        summary TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

# ---------- Utilities ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def save_doc_file(file_storage, owner_id):
    filename = secure_filename(file_storage.filename)
    ext = filename.rsplit(".", 1)[1].lower()
    unique = f"{uuid.uuid4().hex}.{ext}"
    user_dir = os.path.join(DOCS_DIR, owner_id)
    os.makedirs(user_dir, exist_ok=True)
    path = os.path.join(user_dir, unique)
    file_storage.save(path)
    return unique, path

def extract_text_from_pdf(path):
    if fitz is None:
        return ""
    try:
        doc = fitz.open(path)
        texts = []
        for p in doc:
            texts.append(p.get_text())
        return "\n".join(texts)
    except Exception as e:
        print("PDF OCR error:", e)
        return ""

def extract_text_from_image(path):
    if pytesseract is None:
        return ""
    try:
        img = Image.open(path).convert("RGB")
        return pytesseract.image_to_string(img)
    except Exception as e:
        print("Image OCR error:", e)
        return ""

def simple_summarize(text):
    if not text or not text.strip():
        return "No readable text found."
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    top = lines[:6]
    out = []
    for l in top:
        if ":" in l:
            k,v = l.split(":",1)
            out.append(f"{k.strip()}: {v.strip()} (please consult a doctor).")
        else:
            out.append(l)
    return " ".join(out)

def openai_summarize(text):
    if openai is None:
        return None, "openai package not installed"
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None, "OPENAI_API_KEY not set"
    try:
        openai.api_key = key
        prompt = (
            "You are HealthEase assistant. Convert the following medical report text into a short "
            "(3-5 sentence) easy-to-understand summary for a non-medical person. Keep cautious language and add 'see a doctor' if abnormal values are likely.\n\n"
            f"Report:\n{text[:4000]}"
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=300,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip(), None
    except Exception as e:
        return None, str(e)

def ocr_and_summarize(filepath):
    ext = filepath.rsplit(".", 1)[1].lower()
    text = ""
    if ext == "pdf":
        text = extract_text_from_pdf(filepath) or ""
    else:
        text = extract_text_from_image(filepath) or ""
    summary, err = None, None
    if openai is not None and os.getenv("OPENAI_API_KEY"):
        summary, err = openai_summarize(text)
    if summary is None:
        summary = simple_summarize(text)
    return text, summary

# ---------- Simple UI template ----------
BASE_HTML = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>HealthEase</title>
  <style>
    body{font-family:Arial;background:#f3f7fb;margin:0;padding:12px}
    .wrap{max-width:760px;margin:12px auto}
    header{background:#0b61ff;color:#fff;padding:12px;border-radius:10px}
    .card{background:#fff;padding:12px;border-radius:10px;margin-top:12px;box-shadow:0 6px 18px rgba(0,0,0,0.06)}
    .big{display:block;width:100%;padding:12px;background:#0b61ff;color:#fff;border:0;border-radius:8px}
    input,textarea,select{width:100%;padding:10px;margin-top:8px;border-radius:6px;border:1px solid #ddd}
    .small{color:#666;font-size:13px}
    a.button{display:inline-block;padding:8px 12px;background:#1f9f46;color:#fff;border-radius:8px;text-decoration:none}
  </style>
</head>
<body>
  <div class="wrap">
    <header><h2>HealthEase — Prototype</h2></header>
    <div class="card">
      {% with messages = get_flashed_messages() %}
        {% if messages %}<div class="small">{{ messages[0] }}</div>{% endif %}
      {% endwith %}
      {{ body|safe }}
    </div>
    <div style="text-align:center;margin-top:8px" class="small">Prototype — not for production use</div>
  </div>
</body>
</html>
"""

# ---------- Routes ----------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    body = "<h3>Welcome to HealthEase</h3><p class='small'>Upload and summarize medical reports easily.</p>"
    body += "<p><a href='/signup' class='button'>Create account</a> <a href='/login' class='button' style='background:#444'>Login</a></p>"
    return render_template_string(BASE_HTML, body=body)

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","").strip()
        role = request.form.get("role","patient")
        if not (username and password):
            flash("Fill required fields")
            return redirect(url_for("signup"))
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username=?", (username,))
        if cur.fetchone():
            flash("Username exists")
            conn.close()
            return redirect(url_for("signup"))
        uid = uuid.uuid4().hex
        pw = generate_password_hash(password)
        cur.execute("INSERT INTO users (id,username,password_hash,role,created_at) VALUES (?,?,?,?,?)",
                    (uid, username, pw, role, datetime.datetime.utcnow()))
        conn.commit(); conn.close()
        flash("Account created, please login")
        return redirect(url_for("login"))
    body = """
      <h3>Create account</h3>
      <form method="post">
        <label>Username</label><input name="username" required />
        <label>Password</label><input name="password" type="password" required />
        <label>Role</label>
        <select name="role"><option value="patient">Patient</option><option value="doctor">Doctor</option></select>
        <button class="big" type="submit">Create account</button>
      </form>
      <p class='small'>Already have an account? <a href='/login'>Login</a></p>
    """
    return render_template_string(BASE_HTML, body=body)

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","").strip()
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT id,password_hash,role FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        conn.close()
        if row and check_password_hash(row[1], password):
            session['user_id'] = row[0]; session['username'] = username; session['role'] = row[2]
            flash("Logged in")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials")
        return redirect(url_for("login"))
    body = """
      <h3>Login</h3>
      <form method="post">
        <label>Username</label><input name="username" required />
        <label>Password</label><input name="password" type="password" required />
        <button class="big" type="submit">Login</button>
      </form>
      <p class='small'>No account? <a href='/signup'>Create one</a></p>
    """
    return render_template_string(BASE_HTML, body=body)

@app.route("/dashboard")
def dashboard():
    if 'user_id' not in session: return redirect(url_for("login"))
    name = session.get("username"); role = session.get("role")
    body = f"<h3>Hi {name} ({role})</h3>"
    if role == "patient":
        body += "<p><a href='/upload' class='button'>Upload report</a> <a href='/mydocs' class='button' style='background:#1f9f46'>My documents</a></p>"
    else:
        body += "<p><a href='/mydocs' class='button'>Documents shared with me</a></p>"
    body += "<p><a href='/symptom' class='button' style='background:#ff7a59'>Symptom checker</a> <a href='/logout' class='button' style='background:#666'>Logout</a></p>"
    return render_template_string(BASE_HTML, body=body)

@app.route("/logout")
def logout():
    session.clear(); flash("Logged out"); return redirect(url_for("index"))

@app.route("/upload", methods=["GET","POST"])
def upload():
    if 'user_id' not in session or session.get("role") != "patient":
        return redirect(url_for("login"))
    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            flash("No file selected"); return redirect(url_for("upload"))
        if not allowed_file(f.filename):
            flash("File type not allowed"); return redirect(url_for("upload"))
        owner = session['user_id']
        unique, path = save_doc_file(f, owner)
        text, summary = ocr_and_summarize(path)
        doc_id = uuid.uuid4().hex
        conn = get_db(); cur = conn.cursor()
        cur.execute("INSERT INTO documents (id,owner_id,filename,original_name,uploaded_at,ocr_text,summary) VALUES (?,?,?,?,?,?,?)",
                    (doc_id, owner, unique, f.filename, datetime.datetime.utcnow(), text, summary))
        conn.commit(); conn.close()
        flash("Uploaded and processed")
        return redirect(url_for("mydocs"))
    body = """
      <h3>Upload report</h3>
      <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".pdf,.jpg,.jpeg,.png" required />
        <button class="big" type="submit">Upload</button>
      </form>
      <p class='small'><a href='/dashboard'>Back</a></p>
    """
    return render_template_string(BASE_HTML, body=body)

@app.route("/mydocs")
def mydocs():
    if 'user_id' not in session: return redirect(url_for("login"))
    uid = session['user_id']
    conn = get_db(); cur = conn.cursor()
    if session.get("role") == "patient":
        cur.execute("SELECT id,original_name,filename,uploaded_at,shared_with,summary FROM documents WHERE owner_id=? ORDER BY uploaded_at DESC", (uid,))
    else:
        cur.execute("SELECT d.id,d.original_name,d.filename,d.uploaded_at,u.username as patient FROM documents d JOIN users u ON d.owner_id=u.id WHERE d.shared_with=?", (uid,))
    docs = cur.fetchall(); conn.close()
    items = ""
    for d in docs:
        if session.get("role") == "patient":
            items += f"<div class='card'><strong>{d['original_name']}</strong><div class='small'>Uploaded: {d['uploaded_at']}</div>"
            items += f"<div style='margin-top:8px'><a class='button' href='/file/{uid}/{d['filename']}'>Open file</a></div>"
            items += f"<form method='post' action='/share/{d['id']}' style='margin-top:8px'><input name='doctor_username' placeholder='Doctor username' style='width:100%;padding:8px;margin-top:6px'/><button class='big' type='submit'>Share with doctor</button></form>"
            items += f"<div class='small' style='margin-top:6px'>Summary: { (d['summary'][:200] + '...') if d['summary'] else 'No summary' }</div></div>"
        else:
            items += f"<div class='card'><strong>{d['original_name']}</strong><div class='small'>Patient: {d['patient']}</div>"
            items += f"<div style='margin-top:8px'><a class='button' href='/file/{d['patient']}/{d['filename']}'>Open file</a></div></div>"
    body = f"<h3>Documents</h3>{items}<div class='small' style='margin-top:8px'><a href='/dashboard'>Back</a></div>"
    return render_template_string(BASE_HTML, body=body)

@app.route("/file/<owner_id>/<filename>")
def serve_file(owner_id, filename):
    path = os.path.join(DOCS_DIR, owner_id, filename)
    if not os.path.exists(path):
        return "Not found", 404
    uid = session.get("user_id")
    if uid == owner_id:
        return send_from_directory(os.path.join(DOCS_DIR, owner_id), filename)
    # check share
    conn = get_db(); cur = conn.cursor(); cur.execute("SELECT shared_with FROM documents WHERE filename=? AND owner_id=?", (filename, owner_id)); r = cur.fetchone(); conn.close()
    if r and r['shared_with'] == uid:
        return send_from_directory(os.path.join(DOCS_DIR, owner_id), filename)
    return "Forbidden", 403

@app.route("/share/<doc_id>", methods=["POST"])
def share_doc(doc_id):
    if 'user_id' not in session: return redirect(url_for("login"))
    if session.get("role") != "patient": return "Only patients may share", 403
    doctor_username = request.form.get("doctor_username","").strip()
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username=? AND role='doctor'", (doctor_username,))
    r = cur.fetchone()
    if not r:
        flash("Doctor not found"); conn.close(); return redirect(url_for("mydocs"))
    doctor_id = r['id']
    cur.execute("UPDATE documents SET shared_with=? WHERE id=? AND owner_id=?", (doctor_id, doc_id, session['user_id']))
    conn.commit(); conn.close()
    flash("Shared with doctor")
    return redirect(url_for("mydocs"))

@app.route("/doc/<doc_id>", methods=["GET","POST"])
def view_doc(doc_id):
    if 'user_id' not in session: return redirect(url_for("login"))
    conn = get_db(); cur = conn.cursor(); cur.execute("SELECT * FROM documents WHERE id=?", (doc_id,)); d = cur.fetchone(); conn.close()
    if not d: flash("Not found"); return redirect(url_for("mydocs"))
    # permission check
    if session.get("role") == "doctor" and d['shared_with'] != session['user_id']:
        flash("Not shared with you"); return redirect(url_for("mydocs"))
    notes_form = ""
    if session.get("role") == "doctor":
        if request.method == "POST":
            note = request.form.get("note","").strip()
            conn = get_db(); cur = conn.cursor(); existing = d['summary'] or ""
            new = existing + "\n\nDoctor note: " + note
            cur.execute("UPDATE documents SET summary=? WHERE id=?", (new, doc_id)); conn.commit(); conn.close(); flash("Note added"); return redirect(url_for("view_doc", doc_id=doc_id))
        notes_form = "<form method='post'><label>Add note</label><textarea name='note' style='width:100%;padding:8px;margin-top:6px'></textarea><button class='big' type='submit' style='margin-top:8px'>Add Note</button></form>"
    ocr_preview = (d['ocr_text'] or "")[:2000] + ("..." if d['ocr_text'] and len(d['ocr_text'])>2000 else "")
    body = f"<div class='card'><h3>{d['original_name']}</h3><div class='small'>Uploaded: {d['uploaded_at']}</div><div style='margin-top:8px'><a class='button' href='/file/{d['owner_id']}/{d['filename']}'>Open file</a></div><hr/><h4>Summary</h4><div class='card small'>{d['summary'] or 'No summary'}</div><h4>OCR Preview</h4><pre style='white-space:pre-wrap'>{ocr_preview}</pre>{notes_form}</div>"
    return render_template_string(BASE_HTML, body=body)

# Symptom checker
def symptom_rules(text):
    t = (text or "").lower()
    if "fever" in t and "cough" in t:
        return {"condition":"Flu-like illness","specialty":"General Physician"}
    if "chest" in t or "breath" in t:
        return {"condition":"Possible cardiac/respiratory issue","specialty":"Cardiology / Pulmonology"}
    if "headache" in t:
        return {"condition":"Headache / Migraine","specialty":"Neurology"}
    return {"condition":"Unknown - consult a doctor","specialty":"General Physician"}

@app.route("/symptom", methods=["GET","POST"])
def symptom_ui():
    prediction = None; contact = None
    if request.method == "POST":
        s = request.form.get("symptoms","")
        result = symptom_rules(s)
        prediction = result['condition']; contact = f"Suggested specialty: {result['specialty']}"
    body = "<h3>Symptom checker</h3><form method='post'><textarea name='symptoms' rows='4' style='width:100%;padding:8px' placeholder='e.g. fever, cough'></textarea><button class='big' type='submit' style='margin-top:8px'>Check</button></form>"
    if prediction:
        body += f"<div class='card' style='margin-top:8px'><strong>Possible:</strong> {prediction}<br/><strong>Action:</strong> {contact}</div>"
    body += "<div class='small' style='margin-top:8px'><a href='/dashboard'>Back</a></div>"
    return render_template_string(BASE_HTML, body=body)

# API: summarize arbitrary text (frontend could call)
@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    data = request.get_json() or {}
    text = data.get("text","")
    if not text:
        return jsonify({"error":"no text"}), 400
    summary, err = None, None
    if openai is not None and os.getenv("OPENAI_API_KEY"):
        summary, err = openai_summarize(text)
    if summary is None:
        summary = simple_summarize(text)
    return jsonify({"summary": summary, "error": err})

# Seed demo users
@app.route("/seed")
def seed():
    conn = get_db(); cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (id,username,password_hash,role,created_at) VALUES (?,?,?,?,?)",
                    ("patient1","patient1", generate_password_hash("pass"), "patient", datetime.datetime.utcnow()))
        cur.execute("INSERT INTO users (id,username,password_hash,role,created_at) VALUES (?,?,?,?,?)",
                    ("doctor1","doctor1", generate_password_hash("pass"), "doctor", datetime.datetime.utcnow()))
        conn.commit(); conn.close()
        return "Seeded demo accounts: patient1/pass and doctor1/pass"
    except Exception as e:
        return "Seed error or already seeded: " + str(e)

# DB get_db (same as helper)
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
