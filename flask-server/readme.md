# Project: Video uploading/downloading and summarizing

## The backend for this project will be writen in Flask

## Usage local storegate

BACKEND design implemented by Thuy Hien

# How to install FLask project?

# Step 1: Install Virtual Environment

# Install virtualenv on Linux

The package managers on Linux provides virtualenv.

## For Debian/Ubuntu:

1. Start by opening the Linux terminal.

2. Use apt to install virtualenv on Debian, Ubuntu and other related distributions:

$ sudo apt install python-virtualenv

## For CentOS/Fedora/Red Hat:

1. Open the Linux terminal.

2. Use yum to install virtualenv on CentOS, Red Hat, Fedora and related distributions:

$ sudo yum install python-virtualenv

## Install virtualenv on MacOS

1. Open the terminal.

2. Install virtualenv on Mac using pip:

$ sudo python2 -m pip install virtualenv

# Step 2: Create an Environment

1. Make a separate directory for your project:

$ mkdir <project name>

2. Navigate into the directory by using " cd " command

$ cd <project name>

## Create an Environment in Linux and MacOS

### For Python 3:

To create a virtual environment for Python 3, use the venv module and give it a name:

$ python3 -m venv <name of environment>

### For Python 2:

For Python 2, use the virtualenv module to create a virtual environment and name it:

python -m virtualenv <name of environment>

## Create an Environment in Windows

### For Python 3:

Create and name a virtual environment in Python 3 with:

$ py -3 -m venv <name of environment>

### For Python 2:

For Python 2, create the virtual environment with the virtualenv module:

$ py -2 -m virtualenv <name of environment>

# Step 3: Activate the Environment

## Activate the Environment on Linux and MacOS

Activate the virtual environment in Linux and MacOS with:

$ source <name of environment>/bin/activate

## Activate the Environment on Windows

For Windows, activate the virtual environment with:

$ <name of environment>Scriptsactivate

# Step 4: Install Flask

## Activate the Environment on Linux and MacOS

$ pip3 install Flask

## Activate the Environment on Windows

$ pip install Flask

## Install needed Dependencies

Run this command to patch all require dependencies

<pre>
pip install -r requirements.txt
</pre>

## Step 5: Test the Development Environment - Uploading video

The main server.py deploy the Flask BackEnd needed to run this application
The app.py is a stoing file for important information adjust and management
To upload videos stored in a directory, run:

<pre>
python server.py [path_to_video_dataset_folder]
</pre>

This application is ruunung on your localhost machine, at port 5000.
Try out the backend to update and play your video at first.
