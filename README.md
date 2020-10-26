# fastai

my fast.ai v3 and v4 repo :smile_cat:

fastai makes me happy

Note: The video lectures given in youtube and the site are very different. The best place is the site as the latest books are present.

for v4 I will follow www.github.com/fastai/fastbook.

check out MicPie's guide : <https://github.com/MicPie/fastai-pytorch-course-vienna/blob/master/README.md>

* Datasets ideas:

  * Tabular data:

    * Heart disease classification: <https://www.kaggle.com/ronitf/heart-disease-uci>

    * Blue book for bulldozers: <https://www.kaggle.com/c/bluebook-for-bulldozers>

  * Recommendation System:

    * Create an Artificial Sommelier: <https://www.kaggle.com/zynicide/wine-reviews>

    * Movie lens data: <https://www.kaggle.com/prajitdatta/movielens-100k-dataset/notebooks?sortBy=relevance&group=everyone&search=Fast&page=1&pageSize=20&datasetId=626>

  * Time series:

    * Suicide prediction: <https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016>

  * Computer Vision:

    * Histopathologic cancer : <https://www.kaggle.com/mentalwanderer/image-classification-workflow-with-fast-ai>

    * Chest X-tay Images : <https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>

    * Plant Pathology: <https://www.kaggle.com/c/plant-pathology-2020-fgvc7>

    * Malaria infection classification: <https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria>

    * Dog breed classification: <https://www.kaggle.com/jessicali9530/stanford-dogs-dataset>

    * Image segmentation - lunar - <https://www.kaggle.com/romainpessia/artificial-lunar-rocky-landscape-dataset>

    * Fruits classification - <https://www.kaggle.com/moltean/fruits>

    * Detect the artist, based on the image: <https://www.kaggle.com/ikarus777/best-artworks-of-all-time>

  * NLP:

    * Toxic classification: <https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data>

    * Sarcasm detection: <https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection>

    * Open analysis: Predict startup success by media mentions & comments - <https://www.kaggle.com/hacker-news/hacker-news>

  * GAN:
  
    * Create realistic images of the moon: <https://www.kaggle.com/romainpessia/artificial-lunar-rocky-landscape-dataset>

    * Image restoration model (using GAN and fastai 'crapify' idea)

    * Style transfer

* Open lesson for going through practical applications with the fastai library.
* What next, how to keep going, and [keep learning](https://github.com/MicPie/fastai-pytorch-course-vienna#learning-tips)!

## installation

bookmarked links:

0. The link
<https://course.fast.ai/gpu_tutorial.html>

1. jupyter kernal not showing issue
<https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook>

2. useful for setting up local fast ai machine
<https://stackoverflow.com/questions/57813777/how-to-install-fastai-on-windows-10>

3. setting up fast ai on google collab
<https://course.fast.ai/start_colab.html>

4. colab hints
<https://medium.com/@robertbracco1/configuring-google-colab-like-a-pro-d61c253f7573>

``` python
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

```

add these two at start to let git know who you are

Here is definitive proof that "gpu is what you need for deep learning".
Post setup checkup locally.
![cpugpu](cpu_vs_gpu.png)

However it is recommended that we take a cloud provider for running the state of the art models, my gtx 1050 can handle only so much.

## Documentation

inorder to know what a particular function does simply append ?? in front of the function
for ex-

`learn.predict??`

for full documentation

`doc(accuracy)`

### Setting up FASTAI in colab

![colabgpu](colab_gpu.png)

How to open google collab for fast ai

1. open from github
    The github url fo collab
    fastai/course-v3

2. change runtime type to gpu

3. install necessary packages
    `!curl -s https://course.fast.ai/setup/colab | bash`

   for fastai v2:

   ```python
   !pip install -Uqq fastbook
    import fastbook
    fastbook.setup_book()
    ```

4. save a copy in drive
youll be automatically promted to run this

from google.colab import drive
drive.mount('/content/drive')

### using render to deploy the models

<https://course.fast.ai/deployment_render.html>

### Detailed notes

<https://github.com/hiromis/notes/blob/master/Lesson2.md>

## Resources

from : <https://raw.githubusercontent.com/MicPie/fastai-pytorch-course-vienna/master/README.md>

## General PyTorch Deep Learning resources

* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [PyTorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html)
* [PyTorch Docs](https://pytorch.org/docs)
* [Udacity Deep Learning PyTorch Notebooks](https://github.com/udacity/deep-learning-v2-pytorch)
* [CMU Deep Learning Course](http://deeplearning.cs.cmu.edu)
* [CMU Deep Learning Course Recitation Repo](https://github.com/cmudeeplearning11785/Spring2019_Tutorials)
* [Deep Lizard PyTorch Tutorials](http://deeplizard.com/learn/video/v5cngxo4mIg)
* [EE-559 – EPFL – Deep Learning](https://fleuret.org/ee559/)
* [Pytorch torch.einsum](https://rockt.github.io/2018/04/30/einsum) (= the best way to get familiar with matrix calculus and einsum)
* [PyTorch under the hood](https://speakerdeck.com/perone/pytorch-under-the-hood)
* [Advanced PyTorch concepts with code](https://github.com/graykode/projects)

## Deep Learning

* [The deep learning book (Ian Goodfellow and Yoshua Bengio and Aaron Courville)](https://www.deeplearningbook.org)
* [Neural Networks and Deep Learning (Michael Nielson)](http://neuralnetworksanddeeplearning.com)
* [ML yearning (Andrew Ng)](https://www.mlyearning.org) (About how to structure Machine Learning projects.)
* CS 230 Deep Learning Cheatsheets:
  * [Convolutional Neural Networks cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
  * [Recurrent Neural Networks cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)
  * [Deep Learning Tips and Tricks cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks)
* [AI Transformation Playbook (Andrew Ng)](https://landing.ai/ai-transformation-playbook/) (A playbook to become a strong AI company.)

## Mathematics

* [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/index.html)
* [Computational Linear Algebra for Coders](https://github.com/fastai/numerical-linear-algebra)
* <https://www.3blue1brown.com>

## Selected publications

* [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG network)](https://arxiv.org/abs/1409.1556)
* [Deep Residual Learning for Image Recognition (ResNet network)](https://arxiv.org/abs/1512.03385)
* [Network In Network (1x1 convolutions)](https://arxiv.org/abs/1312.4400)
* [Going deeper with convolutions (Inception network)](https://arxiv.org/abs/1409.4842)
* Everything on <https://distill.pub> and <https://colah.github.io>.

## Selected newsletters

* [Import AI](https://jack-clark.net)

## Possible presentation topics

Present one topic with a general introduction and PyTorch code in a Jupyter notebook in approx. 10-20 min. Feel free to add the notebooks to this repo.

* Weight decay, L1-, and L2 regularization ([see weight decay vs. L2 regularization](https://bbabenko.github.io/weight-decay/))
* Drop out (see [chapter 7.12](https://www.deeplearningbook.org/contents/regularization.html))
* [(fastai) Data augmentation](https://github.com/kechan/FastaiPlayground/blob/master/Quick%20Tour%20of%20Data%20Augmentation.ipynb)
* CNN ([Conv Nets: A Modular Perspective](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/) and [Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/), [Convolution arithmetic animations](https://github.com/vdumoulin/conv_arithmetic), and [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)), or [Advanced CNN Architectures](https://dvl.in.tum.de/teaching/adl4cv-ws18/) (Advanced Deep Learning for Computer vision - Munich University)
* ResNets & DenseNets (network architecture, [PyTorch model code](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py), [loss function shape](https://arxiv.org/pdf/1712.09913), etc.)
* 1x1 convolutions ([Network In Network](https://arxiv.org/abs/1312.4400))
* Batch norm ([Udacity Notebook](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/batch-norm/Batch_Normalization.ipynb), [Batch Normalization](https://arxiv.org/abs/1502.03167), [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604), and [Group Normalization](https://arxiv.org/abs/1803.08494))
* LSTM unit ([Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))
* Attention ([Notebook](https://github.com/MicPie/pytorch/blob/master/attention.ipynb))
* Cross entropy loss (based on this [introduction](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/) and [information theory](https://colah.github.io/posts/2015-09-Visual-Information/)).
* [Mixed precision training](https://forums.fast.ai/t/mixed-precision-training/20720), [floating point arithmetics](https://en.wikipedia.org/wiki/Floating-point_arithmetic), and the [fastai callback](https://docs.fast.ai/callbacks.fp16.html).

## Possible projects

* Tensorboard visualisation with fastai callback using [TensorboardX](https://github.com/lanpa/tensorboardX), including 2D visualisations for CNNs ([see starter notebook](https://github.com/MicPie/fastai_course_v3/blob/master/TBLogger_v2.ipynb) and [fastai forum thread](https://forums.fast.ai/t/tensorboard-integration/38023/))
* [Histopathologic Cancer Detection on Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection)

## Deployment

* [fast.ai deployment options](https://course.fast.ai/deployment_render.html)
* [Pythonanywhere](https://www.pythonanywhere.com/) - free to start
* [Render](https://render.com/) - free for static (HTML) sites, 5$ for python hosting
* [Heroku](https://www.heroku.com/) - free basic cloud account for up to 3 projects

## Learning tips

* [10 Top Ideas to Help Your Learning & 10 Pitfalls to Avoid in Your Learning](https://barbaraoakley.com/wp-content/uploads/2018/02/10-Top-Ideas-to-Help-Your-Learning-and-10-Pitfalls-1.pdf) (from the [Learning how to learn](https://www.coursera.org/learn/learning-how-to-learn) course)
* Use spaced repetition to memorize important concepts, APIs, and everything else:
  * [Short intro to the spacing effect](https://fs.blog/2018/12/spacing-effect/)
  * [(More detailed) Introduction to augmenting Long-term Memory](http://augmentingcognition.com/ltm.html) ([concept outlined for a mathematic example](http://cognitivemedium.com/srs-mathematics))
  * [Spaced repitition in detail](https://www.gwern.net/Spaced-repetition)
  * [Anki spaced repitition flashcard app](https://apps.ankiweb.net)
* [Remember, a little bit everyday makes the difference!](https://i1.wp.com/www.samuelthomasdavies.com/wp-content/uploads/2014/08/1-Percent-A-Day.png?w=458&ssl=1)
