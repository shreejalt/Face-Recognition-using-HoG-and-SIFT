import cv2
import numpy as np
from sklearn.decomposition import PCA 
from skimage import feature
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def compute_HoG(img):
    h = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))
    return h

def load_data():
    X_train=np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    unique_labels = np.load('un_label.npy')

    return X_train, X_test, y_train, y_test, unique_labels

def compute_HoG_dataset(X_train):
    X_train_hog=[]
    for i in range(X_train.shape[0]):
        X_train_hog.append(np.reshape(compute_HoG(np.reshape(X_train[i],(256,256))),-1))
    return np.array(X_train_hog)


    

def return_PCA(X_train_hog,X_test_hog):
    PCA_fit = PCA(0.95)
    X_train_hog_pca = PCA_fit.fit_transform(X_train_hog)
    X_test_hog_pca = PCA_fit.transform(X_test_hog)
    
    return PCA_fit, X_train_hog_pca,X_test_hog_pca

def return_labels(y_train, y_test):
    y_train_new = []
    y_test_new = []
    for i in range(y_train.shape[0]):
        idx = (y_train[i]==1).nonzero()[0]
        y_train_new.append(idx[0])  
    
    for i in range(y_test.shape[0]):
        idx = (y_test[i]==1).nonzero()[0]
        y_test_new.append(idx[0])  
        
    return np.array(y_train_new), np.array(y_test_new)

def svm_classifier(X,y):
    model = SVC(kernel="poly", C=30, gamma = "auto")
    model.fit(X, y)
    return model

def predict(unique_labels, X_test, y_test_new,model):
    pred = model.predict(X_test)   
    return (unique_labels[pred[0]], unique_labels[y_test_new])

def main():
    X_train, X_test, y_train, y_test, unique_labels = load_data()
    y_train_new, y_test_new = return_labels(y_train, y_test)
    X_train_hog = compute_HoG_dataset(X_train)
    X_test_hog = compute_HoG_dataset(X_test)
    
    PCA_fit, X_train_hog_pca, X_test_hog_pca = return_PCA(X_train_hog, X_test_hog)
    print ("Before PCA shape = :"+str(X_train_hog.shape)+" After PCA shape(95% variance) = "+str(X_train_hog_pca.shape))
    model = svm_classifier(X_train_hog_pca, y_train_new)
    print ("Score of model(SVM - Polynomial): "+str(model.score(X_train_hog_pca, y_train_new)))
    true_cnt = 0
    true_cnt_val = []
    predic = []
    for i in range(X_test.shape[0]):
        pred,true_val = predict(unique_labels, np.reshape(X_test_hog_pca[i],(1,-1)), y_test_new[i], model) 
        #print("Predicted: "+ str(pred)+" : True: "+str(true_val))
        predic.append(pred)
        true_cnt_val.append(true_val)
        if pred == true_val:
            true_cnt+=1
    print ("F1 score of Polynomial kernel"+str(f1_score(true_cnt_val, predic, average = "weighted")))
    print ("Total true count: "+str(true_cnt)+" out of total size :"+str(X_test.shape[0])) 

if "__main()__" == main():
    
    main()  