from flask import *
import os

app = Flask(__name__)

@app.route("/")
def index():
    current_page = "home"
    return render_template('home.html', current_page=current_page)

@app.route('/upload', methods=['GET','POST'])
def upload():
    file = request.files.get('file')
    file_path = 'static/temp/' + file.filename
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the file to the server
    file.save(file_path)
    current_page = "upload"
    return redirect('/process')

@app.route('/process')
def process():
    current_page = "process"
    return render_template('process.html', current_page=current_page)


