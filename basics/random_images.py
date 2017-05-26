import numpy as np

import PIL.Image 
import matplotlib.pyplot as plt
from subprocess import call
w = 560
h = 800
input_array = np.random.randint(2,size=(w,h))
im = PIL.Image.fromarray(input_array.astype('uint8')*255)
filePath = "grey.jpg"
with open(filePath, "w") as f1:
	im.save(f1, "jpeg")
call(["ristretto", filePath])
