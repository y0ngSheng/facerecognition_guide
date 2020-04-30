import sys
# append tinyfacerec to module search path
sys.path.append("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacerec modules
from tinyfacerec.util import read_images
from tinyfacerec.model import EigenfacesModel

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print ("USAGE: example_model_eigenfaces.py </path/to/images>")
        sys.exit()
    
    # read images
    [X,y,X_test,y_test] = read_images(sys.argv[1])
    # compute the eigenfaces model
    model = EigenfacesModel(X[0:], y[0:])
    # get a prediction for the first observation
    #print "expected =", y[0], "/", "predicted =", model.predict(X[0])
    k=0
    for i in range(len(X_test)):
        predicted=model.predict(X_test[i])
        if(y_test[i]==predicted):
            k=k+1
    print("accuracy is ",k/len(X_test))
