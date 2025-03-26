import numpy as np
import pandas as pd
import ast

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.models import load_model

from keras.layers import Dense, LSTM, Dropout

from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

def haversine(pred, gt):
    """
    Havarsine distance between two points on the Earth surface.

    Parameters
    -----
    pred: numpy array of shape (N, 2)
        Contains predicted (LATITUDE, LONGITUDE).
    gt: numpy array of shape (N, 2)
        Contains ground-truth (LATITUDE, LONGITUDE).

    Returns
    ------
    numpy array of shape (N,)
        Contains haversine distance between predictions
        and ground truth.
    """
    pred_lat = np.radians(pred[:, 0])
    pred_long = np.radians(pred[:, 1])
    gt_lat = np.radians(gt[:, 0])
    gt_long = np.radians(gt[:, 1])

    dlat = gt_lat - pred_lat
    dlon = gt_long - pred_long

    a = np.sin(dlat/2)**2 + np.cos(pred_lat) * np.cos(gt_lat) * np.sin(dlon/2)**2

    d = 2 * 6371 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return d

def write_submission(trip_ids, destinations, file_name="submission"):
    """
    This function writes a submission csv file given the trip ids, 
    and the predicted destinations.

    Parameters
    ----------
    trip_id : List of Strings
        List of trip ids (e.g., "T1").
    destinations : NumPy Array of Shape (n_samples, 2) with float values
        Array of destinations (latitude and longitude) for each trip.
    file_name : String
        Name of the submission file to be saved.
        Default: "submission".
    """
    n_samples = len(trip_ids)
    assert destinations.shape == (n_samples, 2)

    submission = pd.DataFrame(
        data={
            'LATITUDE': destinations[:, 0],
            'LONGITUDE': destinations[:, 1],
        },
        columns=["LATITUDE", "LONGITUDE"],
        index=trip_ids,
    )

    # Write file
    submission.to_csv(file_name + ".csv", index_label="TRIP_ID")

def load_data(csv_path):
    """
    Reads a CSV file (train or test) and returns the data contained.

    Parameters
    ----------
    csv_path : String
        Path to the CSV file to be read.
        e.g., "train.csv"

    Returns
    -------
    data : Pandas DataFrame 
        Data read from CSV file.
    n_samples : Integer
        Number of rows (samples) in the dataset.
    """
    data = pd.read_csv(csv_path, index_col="TRIP_ID")

    return data, len(data)

def polyline_str_to_sequence(polyline_str):
    """
    Converts a polyline string to a sequence of latitude and longitude coordinates.
    
    Parameters
    ----------
    polyline_str : String
        String representation of a polyline.
        e.g., "[[38.5, -120.2], [40.7, -120.95], [43.252, -126.453]]"
    
    Returns
    -------
    sequence : List of Tuples
        Sequence of latitude and longitude coordinates.
        e.g., [(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)]   
    """    
    
    # Convert the string representation of a list to an actual list
    coordinates_list = ast.literal_eval(polyline_str)
    
    # Extract latitude and longitude from each coordinate pair
    sequence = [(coord[1], coord[0]) for coord in coordinates_list]
    
    return sequence

def polylines_to_padded_sequences(polylines, sequence_length):
    """
    Converts a list of polylines to a padded sequence of latitude and longitude coordinates.
    
    Parameters
    ----------
    polylines : List of Strings
        List of string representations of polylines.
        e.g., ["[[38.5, -120.2], [40.7, -120.95]]", "[[43.252, -126.453]]"]
    
    Returns
    -------
    sequence : NumPy Array of Shape (n_samples, sequence_length, 2)
        Padded sequence of latitude and longitude coordinates.
        e.g., [
            [[38.5, -120.2], [40.7, -120.95]],
            [[43.252, -126.453], [0, 0]],
        ]   
    """

    sequence = [polyline_str_to_sequence(polyline) for polyline in polylines]
    
    sequence = pad_sequences(sequence, maxlen=sequence_length, padding='pre', dtype='float32')
    
    return sequence
    
def preprocessing(data):
    """
    Preprocesses the data for training or testing.

    Parameters
    ----------
    data : Pandas DataFrame
        Data read from CSV file.

    Returns
    -------
    X : NumPy Array of Shape (n_samples, n_features)
        Array of features for each sample.
    y : NumPy Array of Shape (n_samples, 2)
        Array of destinations (latitude and longitude) for each sample.
    """
    
    data = data[data['MISSING_DATA'] == False]
    data = data.drop('MISSING_DATA', axis=1)
    
    data = data[data['POLYLINE'] != '[]']
    
    data = data.drop('DAY_TYPE', axis=1)
    
    data = data[~data.index.duplicated(keep='first')]
    
    data['CALL_TYPE'] = data['CALL_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2])

    data = data.fillna(0)
    
    # Convert timestamp to weekday and hour
    data['WEEKDAY'] = [date.weekday() for date in pd.to_datetime(data['TIMESTAMP'], unit='s')]
    data['HOUR'] = [date.hour for date in pd.to_datetime(data['TIMESTAMP'], unit='s')]
    data = data.drop('TIMESTAMP', axis=1)

    # Get the target values
    data['END_Long'] = [eval(polyline)[-1][0] for polyline in data['POLYLINE']]
    data['END_Lat'] = [eval(polyline)[-1][1] for polyline in data['POLYLINE']]
    
    # Separate features and target values
    X, y = data.drop(['END_Long', 'END_Lat'], axis=1), data[['END_Long', 'END_Lat']]
    
    return X, y

def sequence_model(sequence_length, input_features):
    """
    Creates a sequence model.
    
    Parameters
    ----------
    sequence_length : Integer
        Length of the sequence.
    input_features : Integer
        Number of features in the input.
        
    Returns
    -------
    model : Keras Sequential Model
        Sequence model.
    """
    
    # Initialising the model
    model = Sequential()

    # Adding the LSTM layer
    model.add(LSTM(units=50, activation='relu', return_sequences=False, input_shape=(sequence_length, input_features)))

    #  Adding Dropout regularisation between the LSTM layer and the output layer 
    model.add(Dropout(0.1))
        
    # Adding the output layer
    model.add(Dense(units=2)) # 2 units for latitude and longitude coordinates
    
    # Compiling the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_model(model, X_train, y_train, pathname, batch_size=32, epochs=15):
    """
    Trains the model.
    
    Parameters
    ----------
    model : Keras Sequential Model
        Sequence model.
    
    X_train : NumPy Array of Shape (n_samples, sequence_length, n_features)
        Training data.
    
    y_train : NumPy Array of Shape (n_samples, 2)
        Training targets.
    
    pathname : String
        Pathname to save the best model.
    
    batch_size : Integer
        Batch size.
        Default: 32.
    
    epochs : Integer
        Number of epochs.
        Default: 15.
    
    Returns
    -------
    model : Keras Sequential Model
        Trained sequence model.
    """
    # Define learning rate schedule
    def lr_schedule(epoch, lr):
        if epoch < 5:
            return lr
        else:
            return lr * 0.1
        
    # Define callbacks
    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(pathname, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    
    # Train the model 
    model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=batch_size, 
        validation_split=0.1,
        callbacks=[checkpoint, lr_scheduler, early_stopping],
        verbose=1
    )
        
if __name__ == "__main__":
    
    sequence_length = 5
    
    # Load Train set
    train_data, n_trip_train = load_data("train.csv")
    print(f"Train data shape: {train_data.shape}")
    
    X, y = preprocessing(train_data) 
    print(f"Train data X shape: {X.shape}")
    print(f"Train data y shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=9)
    
    # Get padded sequence of coordinates
    train_sequences = polylines_to_padded_sequences(X_train['POLYLINE'], sequence_length)
    test_sequences = polylines_to_padded_sequences(X_test['POLYLINE'], sequence_length)

    # Scale numerical coordinates
    scaler = StandardScaler()
    scaler.fit(train_sequences[0])

    scaled_train_sequences = [scaler.transform(sequence) for sequence in train_sequences]
    scaled_train_sequences = np.array(scaled_train_sequences)

    scaled_test_sequences = [scaler.transform(sequence) for sequence in test_sequences]
    scaled_test_sequences = np.array(scaled_test_sequences)
    
    model = sequence_model(sequence_length, 2)
    model.summary()
        
    train_model(model, scaled_train_sequences, y_train.to_numpy(), "best_model.keras")
    
    # Load best model
    saved_model = load_model("best_model.keras")
    
    # Predict on testing set
    saved_predictions = saved_model.predict(scaled_test_sequences)
    print(np.mean(haversine(saved_predictions, y_test.to_numpy())))
    
    # Load submission set
    test_data, n_trip_test = load_data("test.csv")
    print(f"Test data shape: {test_data.shape}")
    
    # Prepare submission set
    X_submission, _ = preprocessing(test_data)
    submission_sequences = polylines_to_padded_sequences(X_submission['POLYLINE'], sequence_length)
    scaled_submission_sequences = [scaler.transform(sequence) for sequence in submission_sequences]
    scaled_submission_sequences = np.array(scaled_submission_sequences)

    # Predict on submission set
    destinations = saved_model.predict(scaled_submission_sequences)
    destinations = np.stack((destinations[:, 1], destinations[:, 0]), axis=-1) # Swap columns (longitude, latitude) -> (latitude, longitude)

    # Write submission
    test_trips_ids = list(test_data.index)
    write_submission(
        trip_ids=test_trips_ids, 
        destinations=destinations,
        file_name="submission_rnn"
    )
    