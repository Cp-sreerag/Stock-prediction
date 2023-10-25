#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data


# In[2]:


import yfinance as yf

start = "2010-01-01"
end = "2023-01-01"

# Fetch AAPL stock data
krt = yf.Ticker("KRETTOSYS.BO")
df = krt.history(start=start, end=end)

# Display the first 20 rows
df.head(20)
df=df.reset_index()


# In[3]:


print (df.head())


# In[4]:


df.columns


# In[5]:


df = df.drop(['Dividends','Stock Splits'],axis=1)
df.head()


# In[6]:


df.head()


# In[7]:



 plt.plot(df['Date'],df['Close'])


# # Creating moving average

# In[8]:


ma100 = df.Close.rolling(100).mean()
print(ma100)


# In[9]:


plt.figure(figsize=(12,6))
plt.plot(df['Date'],df['Close'])
plt.plot(df['Date'],ma100,'r')


# In[10]:


ma200 = df.Close.rolling(200).mean()
print(ma200)


# In[11]:


plt.figure(figsize=(12,6))
plt.plot(df['Date'],df['Close'])
plt.plot(df['Date'],ma100,'r')
plt.plot(df['Date'],ma200,'g')


# In[12]:


df.shape


# # Splitting Data into testing and training

# In[13]:


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(data_training.shape)
print(data_testing.shape)


# # Scaling of Data

# In[14]:


from sklearn.preprocessing import MinMaxScaler


# In[15]:


scaler = MinMaxScaler(feature_range=(0,1))


# In[16]:


data_training_array = scaler.fit_transform(data_training)


# In[17]:


data_training_array.shape


# # Creating x_train and y_train for actual and predicted value

# In[18]:


x_train = []
y_train = []
for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    


# In[ ]:





# # Converting x_train and y_train to numpy arrays

# In[19]:


x_train,y_train = np.array(x_train),np.array(y_train)


# In[92]:


print(x_train)
print(y_train)


# # Creating ML model

# In[93]:


get_ipython().system('pip install keras tensorflow')


# In[94]:


import keras
from keras.layers import Dense,Dropout,LSTM



# In[95]:


from keras.models import Sequential 


# In[96]:


model = Sequential()


# In[97]:


model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))


# In[98]:


model.summary()


# In[99]:


model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)


# In[103]:


model.save('keras_models1.h5')


# # Testing

# In[104]:


data_testing.head()


# In[105]:


data_training.tail()


# # Appending

# In[106]:


past_100_data = data_training.tail()
final_df = past_100_data.append(data_testing,ignore_index = True)


# In[107]:


final_df.head


# # Saving final df

# In[108]:


input_data = scaler.fit_transform(final_df)
input_data.shape


# # Creating x_test,y_test

# In[109]:


x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    


# # Converting to numpy arrays

# In[110]:


x_test,y_test =np.array(x_test),np.array(y_test)


# In[111]:


print(x_test.shape)
print(y_test.shape) 


# # Making predictions

# In[112]:


y_predicted = model.predict(x_test)


# In[114]:


y_predicted.shape


# In[115]:


y_test


# In[116]:


y_predicted


# # Scaling up

# ## Finding scaling factor

# In[117]:


scaler.scale_


# In[118]:


scale_factor= 1/scaler.scale_


# In[119]:


y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


# # Ploting predicted and original value

# In[120]:


plt.figure(figsize = (12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




