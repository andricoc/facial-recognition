# Facial recognition
School project to do facial recognition using deep learning

Facial expression data is modified from https://www.kaggle.com/c/3364

The code is inside [facial_recognition.ipynb](https://github.com/andricoc/facial-recognition/blob/13d72c784ecb8812809ea8b19888eb1def6dbf9f/facial_recognition.ipynb)

# Overview
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

The data contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order.

This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of a research project. This data is further modified by the lecturer for the school project.
The dataset are not uploaded due to github limit, thus feel free to use the kaggle dataset although result might not be the same.

# Process
The data are augmented by flipping it, thus producing more data to be trained (might not be necessary on the main dataset from kaggle)

![image](https://user-images.githubusercontent.com/63791918/228765900-dbfdf958-5af0-4c0a-aa11-e0c3108e41aa.png)

After that, there are multiple model we can use, Resnet50 being the state of art model. However to use resnet, the data need to be RGB scaled first

```python
rgb_batch = np.repeat(faces[...], 3, -1)    
print(rgb_batch.shape)

x_train, x_val, y_train, y_val = train_test_split(rgb_batch, emotions, test_size=0.2, random_state=2020)
x_train = x_train.astype('float32') / 255
x_val = x_val.astype('float32') / 255
print(faces.shape)
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape, "y_val shape:", y_val.shape)
```

However for me, I am creating my own model.
There are a few other models you can use too. 
It is all inside the [facial_recognition.ipynb](https://github.com/andricoc/facial-recognition/blob/13d72c784ecb8812809ea8b19888eb1def6dbf9f/facial_recognition.ipynb)

# Result
![image](https://user-images.githubusercontent.com/63791918/228769085-20d0e2ff-714b-46ed-9ec6-dc523f936b91.png)

