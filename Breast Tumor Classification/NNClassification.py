import pandas as pd
import numpy as np

data = pd.read_table('cancer.txt',delimiter=',')

import seaborn as sns
plt.figure(figsize=(16,16))
sns.heatmap(data.corr(), annot=True, cmap = 'Blues')

cor_matrix = data.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(to_drop)
df = data.drop(to_drop, axis=1)

sns.set_palette('Set2')
sns.countplot(y)

from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X, y = oversample.fit_resample(df.iloc[:,2:],df.iloc[:,1])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(23, activation = 'relu', input_shape = (23,)))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(
    optimizer = Adam(learning_rate=0.0001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

ann = model.fit(X_train, y_train,
          epochs = 100,
          verbose = 1,
          validation_data = (X_test,y_test))
          
import matplotlib.pyplot as plt


loss_train = ann.history['accuracy']
loss_val = ann.history['val_accuracy']
epochs = range(1,101)
plt.plot(epochs,loss_train, label='Training accuracy')
plt.plot(epochs,loss_val, label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


loss_train = ann.history['loss']
loss_val = ann.history['val_loss']
epochs = range(1,101)
plt.plot(epochs,loss_train, label='Training Loss')
plt.plot(epochs,loss_val, label='validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

from sklearn.metrics import roc_curve, auc
y_prob = model.predict_proba(X_test)
          
fpr, tpr, threshold = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'royalblue', label = 'AUC = %0.7f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'lightcoral')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))






































