# fastai
my fast.ai v3 repo
 
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

Here is definitive proof that "gpu is what you need for deep learning".
Post setup checkup locally.
![](cpu_vs_gpu.png)

However it is recommended that we take a cloud provider for running the state of the art models, my gtx 1050 can handle only so much.

# Setting up FASTAI in colab

![](colab_gpu.png)

How to open google collab for fast ai
1. open from github
    The github url fo collab
    fastai/course-v3

2. change runtime type to gpu

3. install necessary packages
      !curl -s https://course.fast.ai/setup/colab | bash

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
