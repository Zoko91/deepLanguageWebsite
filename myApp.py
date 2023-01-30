from flask import *
import os

app = Flask(__name__)

@app.route("/")
def index():
    current_page = "home"
    return render_template('home.html', current_page=current_page)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file_path = 'static/temp/' + file.filename
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the file to the server
    file.save(file_path)
    current_page = "process"
    return render_template('process.html', current_page=current_page)
