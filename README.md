# fastai
my fast.ai v3 and v4 repo
 
Note: The video lectures given in youtube and the site are very different. The best place is the site as the latest books are present.

for v4 I will follow fastbook.

## installation

bookmarked links: 



0. The link
https://course.fast.ai/gpu_tutorial.html

1. jupyter kernal not showing issue
https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook

2. useful for setting up local fast ai machine
https://stackoverflow.com/questions/57813777/how-to-install-fastai-on-windows-10

3. setting up fast ai on google collab
https://course.fast.ai/start_colab.html

4. colab hints 
https://medium.com/@robertbracco1/configuring-google-colab-like-a-pro-d61c253f7573

    %%javascript
    function ClickConnect(){
    console.log("Working");
    document.querySelector("colab-toolbar-button#connect").click()
    }setInterval(ClickConnect,60000)

    This needs to be added before a training loop

    ## Command Line
    # note, your file_id can be found in the shareable link of the file
    ! pip install gdown -q
    ! gdown — id <file_id>
    ## In Python
    import gdown
    url = https://drive.google.com/uc?id=<file_id>
    output = 'my_archive.tar'
    gdown.download(url, output, quiet=False)

    gdown to grab publically available lib

    import os
    from getpass import getpass
    import urllib
    user = 'rbracco'
    password = getpass('Password: ')
    repo_name = 'fastai2_audio'
    # your password is converted into url format
    password = urllib.parse.quote(password)
    cmd_string = 'git clone https://{0}:{1}@github.com/{0}/{2}.git'.format(user, password, repo_name)
    os.system(cmd_string)
    cmd_string, password = "", "" # removing the password from the variable
    # Bad password fails silently so make sure the repo was copied
    assert os.path.exists(f"/content/{repo_name}"), "Incorrect Password or Repo Not Found, please try again"
    best wayto connect to github

    !git config --global user.email <YOUR EMAIL>
    !git config --global user.name <YOUR NAME>

    add these two at start to let git know who you are

    
Here is definitive proof that "gpu is what you need for deep learning".
Post setup checkup locally.
![](cpu_vs_gpu.png)

However it is recommended that we take a cloud provider for running the state of the art models, my gtx 1050 can handle only so much.

# Documentation

inorder to know what a particular function does simply append ?? in front of the function
for ex-

`learn.predict??`

for full documentation

`doc(accuracy)`

# Setting up FASTAI in colab

![](colab_gpu.png)

How to open google collab for fast ai
1. open from github
    The github url fo collab
    fastai/course-v3

2. change runtime type to gpu

3. install necessary packages
    !curl -s https://course.fast.ai/setup/colab | bash

   for fastai v2:
   !pip install -Uqq fastbook
    import fastbook
    fastbook.setup_book()

4. save a copy in drive
youll be automatically promted to run this

from google.colab import drive
drive.mount('/content/drive')

5. Add the following lines in
    root_dir = "drive/My Drive/" 
    base_dir = root_dir + 'fastai-v3/' 

# using render to deploy the models

https://course.fast.ai/deployment_render.html

# Details notes
https://github.com/hiromis/notes/blob/master/Lesson2.md
