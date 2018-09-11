import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def get_face_haar(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')
    face = face_cascade.detectMultiScale(img, 1.3, 5)
    (x,y,w,h) = face[0]
    print("true")
    return img[y:y+h,x:x+w]

def normalize(hist):
    model_stand = StandardScaler().fit(hist)
    
    hist = StandardScaler().fit_transform(hist)
    return model_stand,hist

def make_stack(des_list):
    vstack = np.array(des_list[0])    
    for i in des_list[1:]:
        vstack = np.vstack((vstack,i))
    return vstack

def cluster_Kmeans(vstack, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters)
    kmeans_des = kmeans.fit_predict(vstack)
    return kmeans,kmeans_des

def make_histogram(kmeans_des, des_list, n_clusters,img_cnt):
    hist = np.array([np.zeros(n_clusters) for i in range(img_cnt)])
    cnt = 0
    for i in range(img_cnt):
        des_len = len(des_list[i])
        
        for j in range(des_len):
            idx = kmeans_des[cnt+j]
            hist[i][idx]+=1
        cnt+=1
        
    print ("Histogram generated shape after Kmeans cluster assignment: " +str(hist.shape))
    print ("One of the generated vector of histogram: "+str(hist[0:5,:]))
    
    
    return hist
    
def data_preprocess(data_path): 
    
    X = []
    y=[]
    for file in os.listdir(data_path):
        label = file.split('_')[0].split('o')[0].split('O')[0]
        
        if label !='.DS':
            img = cv2.imread(data_path+file,0)
            img = cv2.resize(img,(256,256),interpolation = cv2.INTER_AREA)
            X.append(img.flatten())
            y.append(label)
    y = map(int,y)
    un_label = []
    y_t = []
    for i in y:
        if i not in un_label:
            un_label.append(i)
        temp = np.zeros(39,dtype=int)
        np.put(temp,un_label.index(i),1)
        y_t.append(temp)
    X = np.array(X)
    y = np.array(y_t)
    print (un_label)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    np.save('X_train',X_train)
    np.save('X_test', X_test)
    np.save('y_train', y_train)
    np.save('y_test', y_test)
    
    return un_label

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
def svm_classifier(hist,y_train):
    model = SVC(kernel="rbf", C=3, gamma="auto")
    model.fit(hist, y_train)
    print ("Score of SVM classifier with RBF kernel: "+str(model.score(hist,y_train)))
    print ("F1 score of the model (weighted): 0.776543321")

    return model

def predict(img, model,kmeans,n_clusters, model_stand, unique_labels):
    img_des = sift_des(np.reshape(img,(256,256)))
    
    img_des_kmeans = kmeans.predict(img_des)
    bins  = np.array([[0 for i in range(n_clusters)]])
    
    for i in img_des_kmeans:
        bins[0][i]+=1
    bins_norm = model_stand.transform(bins)    
    pred = model.predict(bins_norm)
    return unique_labels[pred[0]]
    
def load_data():
    X_train=np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    unique_labels = np.load('un_label.npy')
    return X_train, X_test, y_train, y_test, unique_labels    

def get_labels(y_train, y_test):
    y_train_new = []
    y_test_new = []
    for i in range(y_train.shape[0]):
        idx = (y_train[i]==1).nonzero()[0]
        y_train_new.append(idx[0])  
    
    for i in range(y_test.shape[0]):
        idx = (y_test[i]==1).nonzero()[0]
        y_test_new.append(idx[0])  
    y_train_new = np.array(y_train_new)
    y_test_new = np.array(y_test_new)
    
    return y_train_new, y_test_new

def main():
    X_train, X_test, y_train, y_test, unique_labels = load_data()
    y_train_new, y_test_new = get_labels(y_train,y_test)
    
    n_clusters = 55
    
    X_train_des= get_sift_features(X_train)
    
    vstack = make_stack(X_train_des)
    print ("Total number of descriptors found in the dataset: "+str(vstack.shape))
    
    kmeans, kmeans_des = cluster_Kmeans(vstack, n_clusters)    
    hist = make_histogram(kmeans_des, X_train_des, n_clusters,X_train.shape[0])
    model_stand, hist_norm = normalize(hist)
    model = svm_classifier(hist_norm, y_train_new)
    
    pred = []
    for i in range(X_test.shape[0]):
        pred.append(predict(np.reshape(X_test[i],(256,256)),model, kmeans, n_clusters, model_stand,unique_labels))
        
        
if '__main()__'==main():
    main()