from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np 

model = load_model('/Users/matrikasubedi/Documents/Deep_Learning/KvasirDataset/kvasir.h5')

img_rows,img_cols = 224,224


class_labels = [
	'Dyed and Lifted Polyps', 
	'Dyed Resection Margins', 
	'Esophagitis', 
	'Cecum', 
	'Pylorus', 
	'Z-line',
	'Polyps',
	'Ulcerative Colitis'
	]
def check(path):
    
    
    # prediction
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32')/255
    pred = np.argmax(model.predict(x))
   
    print("It's a {}.".format(class_labels[pred])) 
  
check('/Users/matrikasubedi/Documents/Deep_Learning/KvasirDataset/dyed.jpg')