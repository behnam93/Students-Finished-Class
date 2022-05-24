import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

data = pd.read_csv('/content/drive/MyDrive/Data/OtherProjects/FeatureVectorWithLabel.csv')

data.head()

y = data['label']

X = data.iloc[:,2:]

rus = RandomUnderSampler()

X_rus,y_rus = rus.fit_resample(X,y)

np.unique(y_rus, return_counts=True)

# PreProcessing: drop columns iwthout any information
X_fin = X_rus.drop(['server_discussion_percent','browser_html_percent','browser_dictation','act_cnt_day_00','server_course_percent','browser_course_info_percent','browser_course','browser_vertical_percent','browser_about','server_outlink_percent'],axis=1)

# PreProcessing: Scale dataset
le = MinMaxScaler()
X_fin = le.fit_transform(X_fin)

X_fin.shape

# Convert categorical variable into dummy/indicator variables.
y_rus = pd.get_dummies(y_rus)

# Split arrays or matrices into random train and test subsets.
X_train,X_test,y_train,y_test = train_test_split(X_fin,y_rus,random_state=100)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(129,activation='relu', input_shape=(129,)),
  tf.keras.layers.Dense(258,activation='relu'),
  tf.keras.layers.Dense(100,activation='relu'),
  tf.keras.layers.Dense(2,activation='softmax'),
])

model.compile('Adam',loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=200, validation_data=(X_test,y_test), batch_size=200)

plt.plot(history.history['accuracy'],c='red',label='Accuracy')

