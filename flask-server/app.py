from flask import Flask

UPLOAD_FOLDER = './static/uploads'

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4'}
#UPLOAD_FOLDER = '../vid_sum_3rd/vid_sum/input_video/uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2024 * 2024 * 1024