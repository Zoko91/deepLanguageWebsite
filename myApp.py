from flask import *

app = Flask(__name__)

@app.route("/")
def index():
    current_page = "home"
    return render_template('home.html', current_page=current_page)
