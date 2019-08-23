# Simple Inception Model

Simple Inception Model is a simple inception model written in Python and uses tensorflow libraries.

## Installation

Install first tensorflow. For Windows 10 and Python 3.7 (64-bit), use the `tensorflow-1.14.0-cp37-cp37m-win_amd64.whl` instead of downloading tensorflow by yourself.

```bash
pip install tensorflow-1.14.0-cp37-cp37m-win_amd64.whl
```

If this fails, try to install just the regular tensorflow:

```bash
pip install tensorflow
```

After installing tensorflow, install the other libraries inside `requirements.txt`

```bash
pip install -r requirements.txt
```

## Preparing Traning Data

Inside the `\train` folder are folders for each classes. By default, the folders inside `\train` are four fruit folders containing `JPG` images. Only .JPG images works for now.

## Modifying the Hyperparameters of the Model (Optional)

Inside the script `settings.py` are adjustable parameters of the Inception model. If you want to train using larger images, just change the `IMG_SIZE` (in pixels and length x width are the same).

## Training the Inception Model

After modifying the contents of the `\train` folder, training can be started by using the command:

```bash
python train.py
```

## Detect/Classify Images

After the training, a model is saved using the parameter `MODEL_NAME`. The `test.py` uses the same model to detect/classify images. To use the `test.py` script to detect/classify images, just run the command:

```bash
python test.py "C:\Users\my_user\images\applefruit.jpg"
```

Contents of the `applefruit.jpg`:

![alt text](https://cdn.hswstatic.com/gif/johnny-appleseed-america-1.jpg)

Sample output of the `test.py` script using the image above is:

```bash
RESULTS:

apples 77.7888 %
dragon fruit 9.9005 %
grapes 8.1165 %
oranges 4.1942 %

Image is class apples
```

## License
[MIT](https://choosealicense.com/licenses/mit/)