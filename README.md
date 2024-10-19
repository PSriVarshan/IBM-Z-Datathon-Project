# IBM-Z-Datathon-Project
# 1. Alzheimer's Detection Using Biomarkers

## Dependencies

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

## Data Preprocessing

```python
# Load the dataset
file_path = 'BIOM.csv'
data = pd.read_csv(file_path)

# Select relevant columns
selected_columns = ["AGE", "PTGENDER", "FDG", "PIB", "MMSE", "PTMARRY", "APOE4", "DX"]
data_selected = data[selected_columns]

# Clean and preprocess data
data_cleaned = data_selected.replace("NA", pd.NA)

# Fill missing values
data_cleaned['AGE'] = data_cleaned['AGE'].fillna(data_cleaned['AGE'].mean())
data_cleaned['FDG'] = data_cleaned['FDG'].fillna(data_cleaned['FDG'].mean())
data_cleaned['PIB'] = data_cleaned['PIB'].fillna(data_cleaned['PIB'].mean())
data_cleaned['MMSE'] = data_cleaned['MMSE'].fillna(data_cleaned['MMSE'].mean())
data_cleaned['APOE4'] = data_cleaned['APOE4'].fillna(data_cleaned['APOE4'].mode()[0])
data_cleaned['PTGENDER'] = data_cleaned['PTGENDER'].fillna(data_cleaned['PTGENDER'].mode()[0])
data_cleaned['PTMARRY'] = data_cleaned['PTMARRY'].fillna(data_cleaned['PTMARRY'].mode()[0])

# Convert categorical variables to numerical
data_cleaned['PTGENDER'] = data_cleaned['PTGENDER'].map({'Male': 0, 'Female': 1})
data_cleaned['PTMARRY'] = data_cleaned['PTMARRY'].map({
    'Married': 0, 'Divorced': 1, 'Widowed': 2, 'Never married': 3, 'Unknown': 4
})

# Map DX column to binary classification
dx_mapping = {
    'NL': 0, 'NL to MCI': 0, 'MCI to NL': 0, 'MCI': 0,
    'Dementia': 1, 'MCI to Dementia': 1, 'NL to Dementia': 1
}
data_cleaned['DX'] = data_cleaned['DX'].map(dx_mapping).dropna()

# Separate features and target variable
X = data_cleaned.drop(columns=['DX'])
y = data_cleaned['DX']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
```

## Model Training

```python
# Define base learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(kernel='linear', probability=True))
]

# Create the stacking ensemble
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())

# Train the stacking model
stacking_model.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/4ad4ff2b-49f4-41b7-9ba2-da26c4ea156d)

## Model Evaluation

```python
# Make predictions
y_pred = stacking_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the stacking model: {accuracy * 100:.2f}%")
```
![image](https://github.com/user-attachments/assets/42fe470d-87bc-481a-90d5-145aeae689fd)


## Making Predictions

```python
def get_user_input():
    age = float(input("Enter Age: "))
    ptgender = int(input("Enter Gender (0 = Male, 1 = Female): "))
    fdg = float(input("Enter FDG value: "))
    pib = float(input("Enter PIB value: "))
    mmse = float(input("Enter MMSE score: "))
    ptmarry = int(input("Enter Marital Status (0=Married, 1=Divorced, 2=Widowed, 3=Never married, 4=Unknown): "))
    apoe4 = int(input("Enter APOE4 allele count (0, 1, or 2): "))

    input_data = np.array([[age, ptgender, fdg, pib, mmse, ptmarry, apoe4]])
    return input_data

# Get user input
user_input = get_user_input()

# Make prediction
prediction = stacking_model.predict(user_input)
probabilities = stacking_model.predict_proba(user_input)

# Display results
print(f"Predicted class (0 = Normal, 1 = Dementia): {prediction[0]}")
print(f"Probability for each class: {probabilities[0]}")
```

# 2. Alzheimer's Detection Using MRI Scans

## Dependencies

```python
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
```

## Reading the Dataset

```python
import os
import pandas as pd

# Directory containing the images
base_dir = 'ad'

# Initialize lists to store image paths and their corresponding labels
images = []
labels = []

# Iterate over each category folder
for label in os.listdir(base_dir):
    label_dir = os.path.join(base_dir, label)
    
    # Ensure it's a directory and not a file
    if os.path.isdir(label_dir):
        # Iterate over each image in the folder
        for image_filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_filename)
            images.append(image_path)
            labels.append(label)

# Create a DataFrame with the image paths and corresponding labels
df = pd.DataFrame({'image': images, 'label': labels})

# Display the DataFrame
df
```
## Displaying the Dataset

```py
plt.figure(figsize=(50,50))
for n,i in enumerate(np.random.randint(0,len(df),50)):
    plt.subplot(10,5,n+1)
    img=cv2.imread(df.image[i])
    img=cv2.resize(img,(224,224))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title(df.label[i],fontsize=25)
```
![image](https://github.com/user-attachments/assets/2aa7c0d8-0f12-48c6-86a4-61cf41b7707d)

## Data Augmentation 

```py
Size=(176,176)
work_dr = ImageDataGenerator(
    rescale = 1./255
)
train_data_gen = work_dr.flow_from_dataframe(df,x_col='image',y_col='label', target_size=Size, batch_size=6500, shuffle=False)

train_data, train_labels = train_data_gen.next()

class_num=list(train_data_gen.class_indices.keys())
class_num

sm = SMOTE(random_state=42)
train_data, train_labels = sm.fit_resample(train_data.reshape(-1, 176 * 176 * 3), train_labels)
train_data = train_data.reshape(-1, 176,176, 3)
print(train_data.shape, train_labels.shape)

labels=[class_num[i] for i in np.argmax(train_labels,axis=1) ]
plt.figure(figsize=(15,8))
ax = sns.countplot(x=labels,palette='Set1')
ax.set_xlabel("Class",fontsize=20)
ax.set_ylabel("Count",fontsize=20)
plt.title('The Number Of Samples For Each Class',fontsize=20)
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
```
![image](https://github.com/user-attachments/assets/4fd386a2-616d-4e6b-b72d-7abbbcb8a5b0)

## Data Splitting for Training, Validation, and Testing

```py
X_train, X_test1, y_train, y_test1 = train_test_split(train_data,train_labels, test_size=0.3, random_state=42,shuffle=True,stratify=train_labels)
X_val, X_test, y_val, y_test = train_test_split(X_test1,y_test1, test_size=0.5, random_state=42,shuffle=True,stratify=y_test1)
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('X_val shape is ' , X_val.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)
print('y_val shape is ' , y_val.shape)
```


## Model Training

```python
model=keras.models.Sequential()
model.add(keras.layers.Conv2D(32,kernel_size=(3,3),strides=2,padding='same',activation='relu',input_shape=(176,176,3)))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same'))
model.add(keras.layers.Conv2D(64,kernel_size=(3,3),strides=2,activation='relu',padding='same'))
model.add(keras.layers.MaxPool2D((2,2),2,padding='same'))
model.add(keras.layers.Conv2D(128,kernel_size=(3,3),strides=2,activation='relu',padding='same'))
model.add(keras.layers.MaxPool2D((2,2),2,padding='same'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024,activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(4,activation='softmax'))
model.summary()
```
![image](https://github.com/user-attachments/assets/2bc060ac-414f-4325-8977-e38340dee6e4)

## Model Architecture

```py
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True,show_dtype=True,dpi=120)
```
![model](https://github.com/user-attachments/assets/8af62913-6974-4484-91eb-24c1d5ed49ce)


## Model Evaluation

```python
checkpoint_cb =ModelCheckpoint("CNN_model.h5", save_best_only=True)
early_stopping_cb =EarlyStopping(patience=10, restore_best_weights=True)
model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train,y_train, epochs=50, validation_data=(X_val,y_val), callbacks=[checkpoint_cb, early_stopping_cb])

hist_=pd.DataFrame(hist.history)
hist_

plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.plot(hist_['loss'],label='Train_Loss')
plt.plot(hist_['val_loss'],label='Validation_Loss')
plt.title('Train_Loss & Validation_Loss',fontsize=20)
plt.legend()
plt.subplot(1,2,2)
plt.plot(hist_['accuracy'],label='Train_Accuracy')
plt.plot(hist_['val_accuracy'],label='Validation_Accuracy')
plt.title('Train_Accuracy & Validation_Accuracy',fontsize=20)
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/4b938ae3-6c46-4d09-8dd9-47a97ca1980b)

## Making Predictions

```python
score, acc= model.evaluate(X_test,y_test)
print('Test Loss =', score)
print('Test Accuracy =', acc)

predictions = model.predict(X_test)
y_pred = np.argmax(predictions,axis=1)
y_test_ = np.argmax(y_test,axis=1)
df = pd.DataFrame({'Actual': y_test_, 'Prediction': y_pred})
df

plt.figure(figsize=(30,70))
for n,i in enumerate(np.random.randint(0,len(X_test),50)):
    plt.subplot(10,5,n+1)
    plt.imshow(X_test[i])
    plt.axis('off')
    plt.title(f"Actual: {class_num[y_test_[i]]}, \n Predicted: {class_num[y_pred[i]]}.\n Confidence: {round(predictions[i][np.argmax(predictions[i])],0)}%",fontsize=20)

```
![image](https://github.com/user-attachments/assets/89783c6a-97b8-4708-87f8-e976b3212e1a)

# 3.Readmission

##
