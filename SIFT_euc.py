import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sift_des(img):
    sift = cv2.xfeatures2d.SIFT_create()
    _, des = sift.detectAndCompute(img,None)
    return  des
    
def get_sift_features(X_train):
    
    X_train_des = []
    for i in range(X_train.shape[0]):
        des = sift_des(np.reshape(X_train[i],(256,256)))
        X_train_des.append(des)
    return X_train_des
    
def load_data():
    X_train=np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    un_labels = np.load('unique_labels.npy')
    return X_train, X_test, y_train_y_test, un_labels

def return_labels(y_train, y_test):
    y_train_new = []
    y_test_new = []
    for i in range(y_train.shape[0]):
        idx = (y_train[i]==1).nonzero()[0]
        y_train_new.append(idx[0])  
    
    for i in range(y_test.shape[0]):
        idx = (y_test[i]==1).nonzero()[0]
        y_test_new.append(idx[0])  
        
    return y_train_new, y_test_new

def return_best_match(X_train_des, test_img, un_labels, y_train_new, y_test_new):
    test_img_des = sift_des(test_img)
    print (test_img_des.shape)
    bf = cv2.BFMatcher()
    cnt = 0
    good=[]
    match_array=dict()
    for i in range(np.array(X_train_des).shape[0]):
        print (X_train_des[i].shape)
        matches = bf.knnMatch(X_train_des[i], test_img_des, k=2)
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append([m])
                cnt+=1
        match_array[i] = cnt
    best_match = max(match_array,key = lambda x: match_array[x])
    print(best_match)
    return un_labels[y_train_new[best_match]]
            
    
    
split_data()
X_train, X_test, y_train, y_test, un_labels = load_data()
y_train_new, y_test_new = return_labels(y_train, y_test)

X_train_des = get_sift_features(X_train)

best_match = return_best_match(X_train_des, np.reshape(X_test[5],(256,256)), un_labels,y_train_new, y_test_new)
print (best_match)
print (un_labels[y_test_new[0]])

plt.imshow(np.resize(X_train[458],(256,256)))