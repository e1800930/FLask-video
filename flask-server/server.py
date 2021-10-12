import os
from app import ALLOWED_EXTENSIONS, app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
import pafy
from werkzeug.utils import secure_filename
import pafy

## code imported from vid_sum_3rd/vid_sum/evaluate.py
#from ..vid_sum_3rd.vid_sum.video_generation.generator import generate_video
#from ..vid_sum_3rd.vid_sum.video_summary.solver import Solver
#from ..vid_sum_3rd.vid_sum.video_summary.data_loader import get_loader
#from ..vid_sum_3rd.vid_sum.video_summary.configs import get_config
#from ..vid_sum_3rd.vid_sum.feature_extraction.generate_dataset import GenerateDataset
#import sys

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
		# check if the post request has the file part
	    if 'file' not in request.files:
		    flash('No file part')
		    return redirect(request.url)
	    file = request.files['file']
	    if file.filename == '':
		    flash('No image selected for uploading')
		    return redirect(request.url)
	    else:
		    filename = secure_filename(file.filename)
		    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			#print('upload_video filename: ' + filename)
		    flash('Video successfully uploaded and displayed below')
		    return render_template('upload.html', filename=filename)

def Process_video():
    if request.method == 'POST':
		# check if the post request has the file part
	    if 'file' not in request.files:
		    flash('No file part exist')
		    return redirect(request.url)
	    file = request.files['file']
	    if file.filename == '':
		    flash('No file to process, please try again')
		    return redirect(request.url)
	    else:
		    filename = secure_filename(file.filename)
		    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			#print('upload_video filename: ' + filename)
		    flash('Video successfully summarized and displayed below')
		    return render_template('upload.html', filename=filename)

		 #if len(sys.argv) > 1:
           # video_path = sys.argv[1].strip()
       # save_path = 'output_feature/output_feature.h5'

        # feature extraction
       # gen_data = GenerateDataset(video_path, save_path)
       # gen_data.generate_dataset()

        # init test config
      #  config = get_config(mode='test', video_type='custom_video')
       # print(config)

        # init data loader
       #train_loader = None
       # test_loader = get_loader(config.mode, save_path, config.action_state_size)

        # evaluation
       # solver = Solver(config, train_loader, test_loader)
       # solver.build()
       # solver.load_model('models/epoch-84.pkl')
       # solver.evaluate(-1)

        # generate video
       # score_path = 'output_feature/custom_video/scores/split' + str(config.split_index) + '/custom_video_-1.json'
       # generate_video(score_path, save_path, video_path)
	  

@app.route('/display/<filename>')
def display_video(filename):
	#print('display_video filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)
	

@app.route('/download_vid', methods=['POST', 'GET'])
def download():
	return render_template('download.html')

def download_vid(filename):
    url = request.form['download_path']
    v = pafy.new(url)
    s = v.allstreams[len(v.allstreams)-1]
    filename = s.download("static/uploads/*.mp4")
    return redirect(url_for('static', filename='download/' + filename), code=301) #orig: return redirect(url_for('done'))
	 

if __name__ == "__main__":
	app.run(debug=True,port= 5000, host='0.0.0.0')