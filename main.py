import numpy as np
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def get_weather_data():
    weather_dic = {}
    fi = csv.reader(open("input/weather.csv"))
    weather_head = fi.__next__()
    for line in fi:
        if line[0] == '1':
            continue
        weather_dic[line[1]] = line
    weather_indexes = dict([(weather_head[i], i) for i in range(len(weather_head))])
    return weather_dic, weather_indexes

def process_line(line, indexes, weather_dic, weather_indexes):
    def get(name):
        return line[indexes[name]]
    species_map = {'CULEX RESTUANS' : 0,
              'CULEX TERRITANS' :0, 
              'CULEX PIPIENS'   : 0, 
              'CULEX PIPIENS/RESTUANS' : 0, 
              'CULEX ERRATICUS' : 0, 
              'CULEX SALINARIUS': 0, 
              'CULEX TARSALIS' :  0,
              'UNSPECIFIED CULEX': 0} 
    date = get("Date")
    month = float(date.split('-')[1])
    week = int(date.split('-')[1]) * 4 + int(date.split('-')[2]) / 7
    latitude = float(get("Latitude"))
    longitude = float(get("Longitude"))
    tmax = float(weather_dic[date][weather_indexes["Tmax"]])
    tmin = float(weather_dic[date][weather_indexes["Tmin"]])
    tavg = float(weather_dic[date][weather_indexes["Tavg"]])
    dewpoint = float(weather_dic[date][weather_indexes["DewPoint"]])
    wetbulb = float(weather_dic[date][weather_indexes["WetBulb"]])
    pressure = float(weather_dic[date][weather_indexes["StnPressure"]])
    species_map[str(get("Species"))] = 1
    restuans = species_map['CULEX RESTUANS']
    territans = species_map['CULEX TERRITANS']
    pipiens = max(species_map['CULEX PIPIENS'], species_map['UNSPECIFIED CULEX'])# Treating unspecified as PIPIENS (http://www.ajtmh.org/content/80/2/268.full)
    pip_rest = species_map['CULEX PIPIENS/RESTUANS']
    errat = species_map['CULEX ERRATICUS']
    salin = species_map['CULEX SALINARIUS']
    tarsalis = species_map['CULEX TARSALIS']
    #adicinar cada espécie como uma variável boolean?
    return [month, week, latitude, longitude, tmax, tmin, tavg, dewpoint, wetbulb, restuans, territans, pipiens, pip_rest,errat,salin,tarsalis]#, species]#, dewpoint, avgspeed]

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def shuffle(X, y, seed=1337):
    np.random.seed(seed)
    shuffle = np.arange(len(y))
    np.random.shuffle(shuffle)
    X = X[shuffle]
    y = y[shuffle]
    return X, y

def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(400))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam")
    return model

def scaled_count(record):
    SCALE = 9.0
    num_mosquitos = int(record[10])
    return int(np.ceil(num_mosquitos / SCALE))
# now the actual script

print("Processing training data...")

rows = []
labels = []
fi = csv.reader(open("input/train.csv"))
head = fi.__next__()
indexes = dict([(head[i], i) for i in range(len(head))])
weather_dic, weather_indexes = get_weather_data()

for line in fi:
    for repeat in range(scaled_count(line)):
        rows.append(process_line(line, indexes, weather_dic, weather_indexes))
        labels.append(float(line[indexes["WnvPresent"]]))
X = np.array(rows)
y = np.array(labels)
X, y = shuffle(X, y)
X, scaler = preprocess_data(X)
Y = np_utils.to_categorical(y)
input_dim = X.shape[1]
output_dim = 2

print("Validation...")

nb_folds = 4
kfolds = KFold(nb_folds)
av_roc = 0.
f = 0
for train, valid in kfolds.split(X):
    print('---'*20)
    print('Fold', f)
    print('---'*20)
    f += 1
    X_train = X[train]
    X_valid = X[valid]
    Y_train = Y[train]
    Y_valid = Y[valid]
    y_valid = y[valid]

    print("Building model...")
    model = build_model(input_dim, output_dim)

    print("Training model...")

    model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_data=(X_valid, Y_valid), verbose=0)
    valid_preds = model.predict(X_valid, verbose=0)
    # import ipdb
    # ipdb.set_trace()
    valid_preds = valid_preds[:, 1] # 1 is the probability to be infected
    roc = metrics.roc_auc_score(y_valid, valid_preds)
    print("ROC:", roc)
    av_roc += roc

print('Average ROC:', av_roc/nb_folds)

print("Generating submission...")

model = build_model(input_dim, output_dim)
model.fit(X, Y, epochs=100, batch_size=16, verbose=0)

fi = csv.reader(open("input/test.csv"))
head = fi.__next__()
indexes = dict([(head[i], i) for i in range(len(head))])
rows = []
ids = []
for line in fi:
    rows.append(process_line(line, indexes, weather_dic, weather_indexes))
    ids.append(line[0])
X_test = np.array(rows)
X_test, _ = preprocess_data(X_test, scaler)

preds = model.predict(X_test, verbose=0)
preds = preds[:,1]
fo = csv.writer(open("output.csv", "w"), lineterminator="\n")
fo.writerow(["Id","WnvPresent"])

for i, item in enumerate(ids):
    fo.writerow([ids[i], preds[i]])