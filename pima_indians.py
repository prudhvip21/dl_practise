from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("/home/prudhvi/dev/dl_practise/pima-indians-diabetes.txt", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer= 'uniform' , activation= 'relu' ))
model.add(Dense(8, kernel_initializer= 'uniform' , activation= 'relu' ))
model.add(Dense(1, kernel_initializer= 'uniform' , activation= 'sigmoid' ))

model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=['accuracy' ])

"""
for train, test in kfold.split(X, Y):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, init= uniform , activation= relu ))
    model.add(Dense(8, init= uniform , activation= relu ))
    model.add(Dense(1, init= uniform , activation= sigmoid ))
    # Compile model
    model.compile(loss= binary_crossentropy , optimizer= adam , metrics=[ accuracy ])
    # Fit the model
    model.fit(X[train], Y[train], nb_epoch=150, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100
"""

model.fit(X, Y, nb_epoch=150, batch_size=10)

""" splitting data using fit function """

#model.fit(X, Y, nb_epoch=150, batch_size=10 , validation_split= 0.33)

"""manual splitting data """
#model.fit(X_train, y_train, validation_data=(X_test,y_test), nb_epoch=150, batch_size=10)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



""" using mlp classifier from keras """


# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
# Function to create model, required for KerasClassifier
def create_model():
# create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, init= uniform , activation= relu ))
    model.add(Dense(8, init= uniform , activation= relu ))
    model.add(Dense(1, init= uniform , activation= sigmoid ))
    # Compile model
    model.compile(loss= binary_crossentropy , optimizer= adam , metrics=[ accuracy ])
    return model
# fix random seed for reproducibility

seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("/home/prudhvi/dev/dl_practise/pima-indians-diabetes.txt", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

""" hyperparameter tuning"""

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
optimizers = [ rmsprop , adam ]
init = [ glorot_uniform , normal , uniform ]
epochs = numpy.array([50, 100, 150])
batches = numpy.array([5, 10, 20])
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))