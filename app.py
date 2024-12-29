from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import PyPDF2
import re
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import joblib

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super secret key'

# Setup directories and file extensions
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')
MODEL_FOLDER = os.path.join(path, 'models')
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

# Load the machine learning model
model_path = os.path.join(MODEL_FOLDER, 'skill_model.pkl')
model = joblib.load(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as pdfFileObj:
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        num_pages = len(pdfReader.pages)
        for count in range(num_pages):
            pageObj = pdfReader.pages[count]
            text += pageObj.extract_text()
    return text

def is_pdf_empty(filepath):
    text = extract_text_from_pdf(filepath)
    return not text.strip()
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Get all uploaded files
        job_role = request.form['job_role']
        threshold = int(request.form['threshold'])  # Get threshold value from form

        if not files:
            flash('No files uploaded', 'warning')
            return redirect(request.url)

        resumes = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                if is_pdf_empty(file_path):
                    flash(f'{filename} is empty or black.', 'warning')
                    continue

                # Extract text from the resume
                text = extract_text_from_pdf(file_path)

                # Clean resume and predict skills
                cleaned_text = clean_resume(text)
                skills = predict_skills(cleaned_text, threshold)  # Pass threshold to skills prediction
                if skills:  # If skills are found
                    # Calculate the highest probability skill
                    max_skill_name = max(skills, key=skills.get)  # Get the skill with the highest probability
                    max_skill_value = skills[max_skill_name]  # Get the value of that highest skill

                    # Simplified probability calculation based on the average of all skills
                    probability = sum(skills.values()) / len(skills)

                        # Only include resumes with a probability >= threshold
                    resumes.append({
                        'filename': filename,
                        'role': job_role,
                        'skills': skills,
                        'probability': probability,
                        'max_skill_name': max_skill_name,  # Add max skill information
                        'max_skill_value': max_skill_value  # Add max skill value
                    })

        if not resumes:
            flash('No resumes met the threshold criteria', 'warning')
            return redirect(request.url)

        return render_template('screening.html', resumes=resumes, threshold=threshold, role=job_role)  # Pass threshold to template

    return render_template('index.html')  # Render index for GET request





@app.route('/screening/<name>', methods=['GET', 'POST'])
def screening(name):
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            text = extract_text_from_pdf(filename)
    else:
        text = extract_text_from_pdf(name)

    text = clean_resume(text)

    if not text.strip():
        flash('No text found in the uploaded file.')
        return redirect(url_for('upload_file'))

    skills = predict_skills(text, threshold=1)
    summary, plot_url = generate_summary_plot(skills)
    bidang, data_all_list, pie_plot_url = existing_screening_functionality(name)

    # Debugging line to check if data_all_list is populated correctly
    print(data_all_list)

    return render_template("screening.html", filename=name, bidang=bidang, data_all_list=data_all_list,
                           summary=summary, plot_url=pie_plot_url, skills_summary=summary, ml_plot_url=plot_url)



def extract_text_from_pdf(filename):
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as pdfFileObj:
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        num_pages = len(pdfReader.pages)
        text = ""
        for count in range(num_pages):
            pageObj = pdfReader.pages[count]
            text += pageObj.extract_text()
    return text    
    

def clean_resume(resumeText):
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub(r'RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub(r'#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub(r'@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub(r'\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText.lower()

def predict_skills(text, threshold=1):
    skills_pred = model.predict([text])[0]
    skills_prob = model.predict_proba([text])[0]

    # Filter out skills with probability below the threshold
    skills = {skill: prob * 100 for skill, prob in zip(model.classes_, skills_prob)}
    return skills

def generate_summary_plot(skills):
    # Create a DataFrame for better visualization
    df = pd.DataFrame(list(skills.items()), columns=['Skill', 'Probability'])

    # Define a list of colors
    colors = ['#FF6347', '#4682B4', '#32CD32', '#8A2BE2']  # Adjusted colors for 4 categories

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 8))
    df.plot(kind='bar', x='Skill', y='Probability', ax=ax, color=colors[:len(df)])

    # Set x-axis labels to be horizontal (no rotation)
    ax.set_xticklabels(df['Skill'], rotation=0, ha='center', fontsize=10)
    ax.set_xlabel('Skills', fontsize=12)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Predicted Skills Probability', fontsize=14)

    plt.tight_layout()

    # Convert plot to PNG image
    png_image = BytesIO()
    plt.savefig(png_image, format='png')
    png_image.seek(0)
    plot_url = base64.b64encode(png_image.getvalue()).decode('utf8')

    return df, 'data:image/png;base64,{}'.format(plot_url)


def existing_screening_functionality(filename):
    pdfFileObj = open('uploads/{}'.format(filename), 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    num_pages = len(pdfReader.pages)
    count = 0
    text = ""
    while count < num_pages:
        pageObj = pdfReader.pages[count]
        count += 1
        text += pageObj.extract_text()

    def cleanResume(resumeText):
        resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
        resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
        resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
        resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
        resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
        resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
        resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
        return resumeText.lower()

    text = cleanResume(text)

    bidang = {
    "Project Management": [
        "administration",
        "agile",
        "feasibility analysis",
        "finance",
        "leader",
        "leadership",
        "management",
        "milestones",
        "planning",
        "project",
        "risk management",
        "schedule",
        "stakeholders",
        "teamwork",
        "communication",
        "organization",
        "research",
        "public speaking",
        "problem solving",
        "negotiation",
        "team management",
        "time management",
        "adaptability",
        "policy knowledge",
        "reporting",
        "technical",
        "motivation"
    ],
    "Backend": [
        "flask",
        "laravel",
        "django",
        "ruby on rails",
        "express.js",
        "codeigniter",
        "golang",
        "mysql",
        "postgres",
        "mongodb",
        "relational database",
        "non relational database",
        "nosql",
        "application programming interface",
        "object oriented programming",
        "python"
    ],
    "Frontend": [
        "react",
        "angular",
        "vue.js",
        "svelte",
        "jquery",
        "backbone.js",
        "ember.js",
        "semantic-ui",
        "html",
        "css",
        "bootstrap",
        "javascript",
        "xml",
        "dom manipulation",
        "json"
    ],
    "Devops": [
        "networking",
        "tcp",
        "udp",
        "microsoft azure",
        "amazon web services",
        "alibaba cloud",
        "google cloud",
        "docker",
        "kubernetes",
        "virtual machine",
        "cloud computing",
        "security",
        "linux",
        "ubuntu",
        "debian",
        "arch linux",
        "kali linux",
        "automation",
        "containers",
        "operations",
        "troubleshooting"
    ]
}


    project = 0
    backend = 0
    frontend = 0
    data = 0
    devops = 0

    project_list = []
    backend_list = []
    frontend_list = []
    data_list = []
    devops_list = []

    # Create an empty list where the scores will be stored
    scores = []

    # Obtain the total number of words in the CV (cleaned)
    words = len(text.split(" "))

    for key in bidang:
        for skill in bidang[key]:
            if skill in text:
                if key == "Project Management":
                    project += 1
                    project_list.append(skill)
                elif key == "Backend":
                    backend += 1
                    backend_list.append(skill)
                elif key == "Frontend":
                    frontend += 1
                    frontend_list.append(skill)
                elif key == "Devops":
                    devops += 1
                    devops_list.append(skill)

    def create_plot(data_all):
        explode = (0.1, 0.1, 0.1, 0.1, 0.1)
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"))
        wedges, texts, autotexts = ax.pie(data_all, explode=explode, colors=colors, autopct='%1.1f%%', startangle=140)
        ax.legend(wedges, ['Project Management', 'Backend', 'Frontend', 'Data Science', 'DevOps'],
                  title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=10, weight="bold")
        ax.set_title("Skills Distribution")
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plot_url = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return plot_url

    data_all = [project, backend, frontend, data, devops]
    data_all_list = [project_list, backend_list, frontend_list, data_list, devops_list]

    pie_plot_url = create_plot(data_all)

    return bidang, data_all_list, pie_plot_url

@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = UPLOAD_FOLDER + '/' + filename
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True, attachment_filename='')

if __name__ == "__main__":
    app.run(debug=True)
