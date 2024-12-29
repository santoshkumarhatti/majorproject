import os
from flask import Flask, flash, request, redirect, url_for, render_template
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

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'warning')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'warning')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File uploaded successfully.', 'success')
            return redirect(url_for('screening', name=filename) + "#hasil")
        else:
            flash('Upload a valid PDF file.', 'warning')
            return redirect(request.url)
    return render_template("index.html")
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

    skills = predict_skills(text, threshold=5)
    summary, plot_url = generate_summary_plot(skills)
    bidang, data_all_list, pie_plot_url = existing_screening_functionality(name)

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

def predict_skills(text, threshold=5):
    skills_pred = model.predict([text])[0]
    skills_prob = model.predict_proba([text])[0]

    # Filter out skills with probability below the threshold
    skills = {skill: prob * 100 for skill, prob in zip(model.classes_, skills_prob) if prob * 100 >= threshold}
    return skills

def generate_summary_plot(skills):
    # Create a DataFrame for better visualization
    df = pd.DataFrame(list(skills.items()), columns=['Skill', 'Probability'])

    # Plot the data
    fig, ax = plt.subplots()
    df.plot(kind='bar', x='Skill', y='Probability', ax=ax)

    # Convert plot to PNG image
    png_image = BytesIO()
    plt.savefig(png_image, format='png')
    png_image.seek(0)
    plot_url = base64.b64encode(png_image.getvalue()).decode('utf8')

    return df, 'data:image/png;base64,{}'.format(plot_url)

def existing_screening_functionality(filename):
    text = extract_text_from_pdf(filename)
    text = clean_resume(text)

    bidang = {
        'Project Management': ['administration', 'agile', 'feasibility analysis', 'finance', 'leader', 'leadership',
                               'management', 'milestones', 'planning', 'project', 'risk management', 'schedule',
                               'stakeholders', 'teamwork', 'communication', 'organization', 'research',
                               'public speaking', 'problem solving', 'negotiation', 'team management',
                               'time management', 'adaptability', 'policy knowledge', 'reporting', 'technical',
                               'motivation'],

        'Backend': ['flask', 'laravel', 'django', 'ruby on rails', 'express.js', 'codeigniter', 'golang', 'mysql',
                    'postgres', 'mongodb', 'relational database', 'non relational database', 'nosql',
                    'application programming interface', 'object oriented programming'],

        'Frontend': ['react', 'angular', 'vue.js', 'svelte', 'jquery', 'backbone.js ', 'ember.js', 'semantic-ui',
                     'html', 'css', 'bootstrap', 'javascript', 'xml', 'dom manipulation', 'json'],

        'Data Science': ['math', 'statistic', 'probability', 'preprocessing', 'machine learning',
                         'data visualization', 'python', 'r programming', 'tableau', 'natural language processing',
                         'data modeling', 'big data', 'deep learning', 'relational database management', 'clustering',
                         'data mining', 'text mining', 'jupyter', 'neural networks', 'deep neural network', 'pandas',
                         'scipy', 'matplotlib', 'numpy', 'tensorflow', 'scikit learn', 'data analysis', 'data privacy',
                         'data', 'enterprise resource planning', 'oracle', 'sybase', 'decision making', 'microsoft excel',
                         'data collection', 'data cleaning', 'pattern recognition', 'google analytics'],

        'DevOps': ['networking', 'tcp', 'udp', 'microsoft azure', 'amazon web services', 'alibaba cloud',
                   'google cloud', 'docker', 'kubernetes', 'virtual machine', 'cloud computing', 'security', 'linux',
                   'ubuntu', 'debian', 'arch linux', 'kali linux', 'automation', 'containers', 'operations', 'security',
                   'testing', 'troubleshooting']
    }

    project_list = []
    backend_list = []
    frontend_list = []
    data_list = []
    devops_list = []

    for key in bidang:
        for skill in bidang[key]:
            if skill in text:
                if key == "Project Management":
                    project_list.append(skill)
                elif key == "Backend":
                    backend_list.append(skill)
                elif key == "Frontend":
                    frontend_list.append(skill)
                elif key == "Data Science":
                    data_list.append(skill)
                elif key == "DevOps":
                    devops_list.append(skill)

    # Create lists for categories with non-zero skills only
    categories = []
    values = []
    data_all_list = []

    if project_list:
        categories.append('Project Management')
        values.append(len(project_list))
        data_all_list.append(project_list)
    if backend_list:
        categories.append('Backend')
        values.append(len(backend_list))
        data_all_list.append(backend_list)
    if frontend_list:
        categories.append('Frontend')
        values.append(len(frontend_list))
        data_all_list.append(frontend_list)
    if data_list:
        categories.append('Data Science')
        values.append(len(data_list))
        data_all_list.append(data_list)
    if devops_list:
        categories.append('DevOps')
        values.append(len(devops_list))
        data_all_list.append(devops_list)

    def create_plot(data_all, labels):
        if not data_all:  # Check if there are no categories to display
            return None

        # Filter out zero values
        filtered_data = [(label, value) for label, value in zip(labels, data_all) if value > 0]
        if not filtered_data:
            return None

        filtered_labels, filtered_values = zip(*filtered_data)
        
        explode = (0.1,) * len(filtered_labels)  # Adjust explosion based on the number of categories
        colors = plt.get_cmap("tab10").colors  # Use a colormap with enough colors
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"))
        wedges, texts, autotexts = ax.pie(filtered_values, explode=explode, colors=colors, autopct='%1.1f%%', startangle=140)
        ax.legend(wedges, filtered_labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=10, weight="bold")
        ax.set_title("Skills Distribution")
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plot_url = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return plot_url

    pie_plot_url = create_plot(values, categories)

    return bidang, data_all_list, pie_plot_url

if __name__ == '__main__':
    app.run(debug=True)
