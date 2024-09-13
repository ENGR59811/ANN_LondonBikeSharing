# The City College of New York, City University of New York
# Written by Ricardo Valdez and Jian Wen Choong                                                                          
# August, 2020
# Data for this example was taken from: 
# https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset
import numpy as np
import pandas as pd
import time
from datetime import datetime
from tensorflow import keras

# import print_weights as pw
# call: pw.print_weights()

def print_weights(weights):
    # weights = model.get_weights();
    print('\n******* WEIGHTS OF ANN *******\n') 
    for i in range(int(len(weights)/2)):
        print('Weights W%d:\n' %(i), weights[i*2])
        print('Bias b%d:\n' %(i), weights[(i*2)+1])
#END print_weights()

#% ANN TRAINING
print('\n')
print('********************************************************************')     
print('****  WELCOME TO BIKE SHARING USING ARTIFICIAL NEURAL NETWORKS  ****')
print('********************************************************************')

# prompt user to train or load an ANN model
option_list = ['1','2']
option = ''
while option not in option_list:  
    print('\nOPTIONS:')
    print('1 - Train a new ANN model')
    print('2 - Load an existing model')
    
    option = input('\nSelect an option by entering a number: \n')
    if option not in option_list:
        message = 'Invalid input: Input must be one of the following - '
        print(message, option_list)
        time.sleep(2)
        
if option == '1':
    ## OPTION 1: TRAIN A NEW ANN MODEL
    train_data_file = 'london_bike_sharing_data.csv'
    
    print('\n********* NOW TRAINING ANN USING', train_data_file,'*********')
    time.sleep(3)
    
    ## load the training data
    df = pd.read_csv(train_data_file)
    
    ## the training data contains 7 columns:
    ##
    ##      timestamp - the date and time the sample was recorded
    ##      new_bikes_shared - number of new bikes shared over the last hour
    ##      is_weekend - boolean that is 1 (true) if the day is a weekend
    ##      temp_c - the temperature in Celcius
    ##      wind_speed - wind speed in km/h
    ##      weather_code - category of weather: 1 = clear
    ##                                          2 = scattered clouds
    ##                                          3 = broken clouds
    ##                                          4 = cloudy
    ##                                          7 = rain
    ##                                         10 = thunderstorm
    ##                                         26 = snow
    ##                                         94 = freezing fog
    
    ## the timestamp column of df are stored as strings. We want to
    ## convert each timestamp string into a datetime objects using the 
    ## function datetime.strptime(). The first input of datetime.strptime() 
    ## is the string you want to convert, and the second input is the 
    ## format of the string, where
    ## %m = month, %d = day, %Y = year, %H = hour, %M = minute.
    ##
    ## create lambda function to perform conversion and return the hour.
    get_hour = lambda timestamp: datetime.strptime(timestamp,
                                                   '%m/%d/%Y %H:%M').hour
    ## apply the lambda function to every timestamp in column df['timestamp']
    df['time_hour'] = df['timestamp'].apply(get_hour)
    
    ## define input matrix X (get rid of columns called timestamp and
    ## new_bikes_shared)
    X = np.array(df.drop(['timestamp','new_bikes_shared'], axis=1))
    ## define expected output matrix Y
    Y = np.array(df['new_bikes_shared'])
    
    ## create a model for the ANN
    model = keras.Sequential()
    ## add a hidden layer that accepts 5 input features (time_hour, temp_c
    ## wind_speed, weather_code, is_weekend)
    ## the hidden layer has 5 neurons.
    ## Dense means every neuron in the layer connects to every neuron in the
    ## previous layer.
    model.add(keras.layers.Dense(5, activation='relu', input_shape=(5,)))
    ## add another hidden layer with 6 neurons to the ANN
    model.add(keras.layers.Dense(6, activation='relu'))
    ## add an output layer with a single output (new_bikes_shared)
    model.add(keras.layers.Dense(1, activation='linear'))
    
    ## set the optimization algorithm used for minimizing loss function
    ## use gradient descent (adam) to minimize error (loss)
    model.compile(optimizer='adam', loss='mean_squared_error')
    ## train the ANN model using 2000 iterations
    model.fit(X, Y, epochs=2000)
    
    print('\n\n********** ANN training complete **********\n\n')    
elif option == '2':
    ## OPTION 2: LOAD ANN MODEL FROM FILE
    
    message = 'Enter the file name of the ANN Model you want to load: \n'
    load_file = input(message)
    #load_file = input('It must be a .h5 file')
    
    ## if file name does not end with '.h5', add '.h5' to the file name
    if load_file[-3:] != '.h5':
        load_file += '.h5'
    ## load the ANN model from load_file
    model = keras.models.load_model(load_file)
    
    print('\n\n****** SUCCESSFULLY LOADED ANN MODEL FROM', load_file,'******')   
else:
    print('ERROR: INVALID OPTION SELECTED')
    ## raise an exception to terminate the program
    raise ValueError()

weights = model.get_weights();
print_weights(weights)

#% BIKE SHARING PREDICTION USING ANN
input('\n\n********** Press ENTER to start using the ANN **********\n\n')
finished = False
while not finished:
    ## prompt user for inputs
    temp_c = float(input('\n\nEnter temperature in Celcius: \n'))
    hour = float(input('Enter hour of the day (military): (0-23) \n'))
    is_weekend = input('Is it the weekend? (y/n): \n')
    if is_weekend == 'y':
        is_weekend = 1
    else:
        is_weekend = 0
    wind_speed = float(input('Enter wind speed: (km/h) \n'))
    weather_code = int(input('Enter weather code: (1 = clear, 2 = few clouds, '
                   + '3 = broken clouds, 4 = cloudy, 7 = rain, '
                   + '10 = thunderstorm, 26 = snow, 94 = freezing fog) \n'))
    
    user_input=np.array([[is_weekend, temp_c, wind_speed, weather_code,hour]])
    prediction = model.predict(user_input)
    
    ## restrict prediction to non-negative values
    if prediction < 0:
        prediction = 0;
    else:
        pass

    ## display prediction
    print('\n*****************************************')
    print('ANN Predicted number of shared bikes: ', int (prediction))
    print('*****************************************')
    ## ask user if they would like to continue
    choice = ''
    while choice not in ['y','n']:
        choice = input('\n\nWould you like to continue? (y/n): \n')
        if choice == 'y':
            pass
        elif choice == 'n':
            finished = True
        else:
            print("Invalid input: Input must be 'y' or 'n'")
    #END WHILE
#END WHILE

## ask user if they would like to save the ANN model
choice = ''
while choice not in ['y','n']:
    choice = input('\n\nWould you like to save the ANN model? (y/n): \n')
    if choice == 'y':
        save_name = input('\n\nEnter a name for the save file: \n')
        ## if file name does not end with '.h5', add '.h5' to the file name
        if save_name[-3:] != '.h5':
            save_name += '.h5'
        model.save(save_name)
        print('\n\n')
        print('***** ANN MODEL SUCCESSFULLY SAVED AS '+save_name+' *****')
    elif choice == 'n':
        pass
    else:
        print("Invalid input: Input must be 'y' or 'n'")
#END WHILE
