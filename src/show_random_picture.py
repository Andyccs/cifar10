# Open 10 random picture from traindata/ folder
# The original images will be converted to BMP files temporary, 
# and open using default image viewer (typically Paint in Windows and Preview in MAC OS X)

from PIL import Image
import os
import random

train_dir = 'traindata/'
images = os.listdir(train_dir)

print len(images)

for _ in range(10):
  index = random.randint(0, 49999)
  # print index
  image_name = images[index]
  image = Image.open(train_dir + image_name)
  image.show()
  image.close()
