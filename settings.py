#dimensions of images to be processed
IMG_SIZE = 128
#1 for grayscale image, 3 for colored
IMAGE_CHANNELS = 3
LR = 1e-4
NUM_OUTPUT = 10
NUM_EPOCHS = 50
FIRST_NUM_CHANNEL = 8

TRAIN_DIR = 'train'
MODEL_NAME = 'inception{}-{}.model'.format(LR, '4inceptionlayers')