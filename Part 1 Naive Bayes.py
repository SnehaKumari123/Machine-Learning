#!/usr/bin/env python
# coding: utf-8

# In[3]:


from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
# File Root
FileRoot = '/Users/sneha/Desktop/ML/phase1/'
#This is for the users local sytem. Please change this accordingly.

#####
# Train Image Data 0
train0image = loadmat(FileRoot + 'train_0_img.mat')
data0train  = np.array(train0image['target_img'], dtype='float')

# Train Image Data 1
train1image = loadmat(FileRoot + 'train_1_img.mat')
data1train  = np.array(train1image['target_img'], dtype='float')

#####
# Test Image Data 0
test0image = loadmat(FileRoot + 'test_0_img.mat')
data0test  = np.array(test0image['target_img'], dtype='float')

# Test Image Data 1
test1image = loadmat(FileRoot + 'test_1_img.mat')
data1test  = np.array(test1image['target_img'], dtype='float')

# Train Label Data 0
train0Label = loadmat(FileRoot + 'train_0_label.mat')
data0trainLabel  = np.array(train0Label['target_label'], dtype='float')

# Train Label Data 1
train1Label = loadmat(FileRoot + 'train_1_label.mat')
data1trainLabel  = np.array(train1Label['target_label'], dtype='float')

#####
# Test Label Data 0
test0Label = loadmat(FileRoot + 'test_0_label.mat')
data0testLabel  = np.array(test0Label['target_label'], dtype='float')

# Test Label Data 1
test1Label = loadmat(FileRoot + 'test_1_label.mat')
data1testLabel  = np.array(test1Label['target_label'], dtype='float')


#Fumction to calculate the average brightness of the image
def Mean (Image,Size):
    total = 0
    for row in Image:
        for pixel in row:
            total += pixel
    return total/Size

#Function to calculate the average of the variances of each rows of the image.
def calculateAvgRowVar(image):
    img_array= []
    for i in range(28):
        img_array.append(image [i].var())
    return(np.array(img_array).mean())
    


# Compute the Average and Variance for Each Image in Train Data
def ComputeDataTrain (datasize,datatrain,averagebright,averagevariance):
     for idx in range (0,datasize):
        image = (np.array(datatrain,dtype='float')[:,:,idx]).reshape  (28,28)

        # Compute the Mean
        meanimage = Mean (image,28*28)
 
        # Compute the Variance
        variance = calculateAvgRowVar (image)
        # Add the Mean to the Array
        averagebright.append (meanimage)
        # Add the Variance to the Array
        averagevariance.append (variance)

        #Function to compute the normal Distribution         
def ComputeNormaldist(mean, var , x):
    sd=math.sqrt(var)
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

#Function to plot and display the normal Distribution with all estimated parameters.      
def DisplayNormalDistribution (title,xtitle,data,display):
    # Calculated the mean for the Distribution
    mean = np.mean (data)
    variance = np.var (data)
    sd = np.std (data)
    covariance = np.cov (data)

    if (display):
        # Display the Normal Distribution graph
        fig, ax = plt.subplots(figsize=(9,6))
        ax.set_title(title)
        ax.set_xlabel(xtitle)

        ax.text(1.0, 0.99,'Mean     = %f '%(mean), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)
        ax.text(1.0, 0.95,'Variance   = %f '%(variance), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)
        ax.text(1.0, 0.91,'CoVariance = %f '%(covariance), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)
        ax.text(1.0, 0.87,'STD        = %f '%(sd), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)

        plt.hist(data, 100, density=True,alpha=0.6, color='g')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf (x, mean, sd)
        plt.plot(x, p, 'k', linewidth=2)
        plt.show ()

#Function to dispaly all the fearures
def Report (title,averagebright,averagevariance):
    print ("Title %s "%(title))
    for idx in range (0,len (averagebright)):
        print ("Image ID %d Mean %f Variance %f "%(idx,averagebright[idx],averagevariance [idx]))


# Main Program
if __name__ == "__main__":
    
    # Store both Features(AverageBrightness and Average Variance) in following lists for Train0 Image 
    averagebrighttrain0=[]
    averagevariancetrain0=[]

    # Store both Features(AverageBrightness and Average Variance) in following lists for Train1 Image 
    averagebrighttrain1=[]
    averagevariancetrain1=[]
    
    
    #Compute both Features(AverageBrightness and Average Variance) in following lists for Train0 Image 
    ComputeDataTrain (data0train.shape[2],data0train,averagebrighttrain0,averagevariancetrain0)
    
    #Compute both Features(AverageBrightness and Average Variance) in following lists for Train1 Image 
    ComputeDataTrain (data1train.shape[2],data1train,averagebrighttrain1,averagevariancetrain1)
    
    #Compute Estimate parameters of TrainImage 0 for distribution
    meanfeat1train0=np.array(averagebrighttrain0).mean()
    varfeatt1rain0=np.array(averagebrighttrain0).var()
    meanfeat2train0=np.array(averagevariancetrain0).mean()
    varfeat2train0=np.array(averagevariancetrain0).var()
    
    #Compute Estimate parameters of TrainImage 1 for distribution
    meanfeat1train1=np.array(averagebrighttrain1).mean()
    varfeatt1rain1=np.array(averagebrighttrain1).var()
    meanfeat2train1=np.array(averagevariancetrain1).mean()
    varfeat2train1=np.array(averagevariancetrain1).var()
    
    # Display Estimate parameters of TrainImage 0 and TrainImage 1 for distribution
    print("Estimated values for the parameters")
    #print("Mean of average brightness of Train0 data = %f Variance of average brightness of Train0 data = %f
    #Mean of average variance of Train0 data = %f Variance of of average variance of Train0 data = %f"%(meanfeat1train0,varfeatt1rain0,meanfeat2train0,varfeat2train0) )
    #print("Mean of average brightness of Train1 data = %f  Variance of average brightness of Train1 data = %f Mean of average variance of Train1 data = %f Variance of of average variance of Train1 data = %f"%(meanfeat1train1,varfeatt1rain1,meanfeat2train1,varfeat2train1) )
    print("Mean of average brightness of Train0 data = %f" %(meanfeat1train0))
    print("Variance of average brightness of Train0 data = %f" %(varfeatt1rain0))
    print("Mean of average variance of Train0 data = %f "%(meanfeat2train0))
    print("Variance of average variance of Train0 data = %f " %(varfeat2train0))
    #Displaying the normal distribution  graph for feature 1 and 2 of Train0 data.
    DisplayNormalDistribution ("Feature1 Train0 data ","Average brightness",averagebrighttrain0,display)
    DisplayNormalDistribution ("Feature2 Train0 data ","Average Variances",averagevariancetrain0,display)
    print("Mean of average brightness of Train1 data = %f" %(meanfeat1train1))
    print("Variance of average brightness of Train1 data = %f" %(varfeatt1rain1))
    print("Mean of average variance of Train1 data = %f "%(meanfeat2train1))
    print("Variance of average variance of Train1 data = %f " %(varfeat2train1))
    #Displaying the normal distribution  graph for feature 1 and 2 of Train1 data.
    DisplayNormalDistribution ("Feature1 Train1 data ","Average brightness",averagebrighttrain1,display)
    DisplayNormalDistribution ("Feature2 Train1 data ","Average Variances",averagevariancetrain1,display)
    
    #Initialize two lists to store the predicted label for each testing sample from Test0 Data and  Test1 Data.
    resultfor0=[]
    resultfor1=[]
    
    #Compute the  estimated normal distributions Naïve Bayes Classifier and use it produce a predicted label for Test0 Data samples.
    for i in range(data0test.shape[2]):
        
        imagetest0 = (np.array(data0test,dtype='float')[:,:,i]).reshape  (28,28)
        
        # computing  first feature of Test0 image
        meanimagetest0 = Mean (imagetest0,28*28)
    
        # computing secoond feature  of Test0 image
        vartest0=calculateAvgRowVar (imagetest0)

        #Computing estimated normal distributions for Test0 data sample with Train0 data.
        p1=ComputeNormaldist(meanfeat1train0, varfeatt1rain0 ,meanimagetest0 )#From Feature1 (Average brightness)
        p2=ComputeNormaldist(meanfeat2train0, varfeat2train0 ,vartest0 )#For Feature2 (Average variance)
        
        #implementing the Naive Bayes Classifier for Test0 data sample with Train0 data.
        Probfor00=0.5*p1*p2
    
        #Computing estimated normal distributions for Test0 data sample with Train1 data.
        q1=ComputeNormaldist(meanfeat1train1, varfeatt1rain1 ,meanimagetest0 )
        q2=ComputeNormaldist(meanfeat2train1, varfeat2train1 ,vartest0 )
        
        #implementing the Naive Bayes Classifier for Test0 data sample with Train1 data.
        Probfor01=0.5*q1*q2
        
        #Using Naive Bayes Classifier to produce a predicted label for testing sample.
        # Label the test sample with 0 or 1 based on the higher value.
        if(Probfor00>Probfor01):
            resultfor0.append(0)
        else:
            resultfor0.append(1)
        #result0 stores all the estimated labels from the data samples in Test0 Data.
    #Compute the  estimated normal distributions Naïve Bayes Classifier and use it produce a predicted label for Test1 Data samples.
    for i in range(data1test.shape[2]):
        image2 = (np.array(data1test,dtype='float')[:,:,i]).reshape  (28,28)
        
        # computing  first feature of Test1 image
        meanimage1 = Mean (image2,28*28)
        
        # computing secoond feature  of Test1 image
        vartest1=calculateAvgRowVar  (image2)
        
        #Computing estimated Probability density function for Test1 data sample with Train0 data.
        p11=ComputeNormaldist(meanfeat1train0, varfeatt1rain0 ,meanimage1 )
        p21=ComputeNormaldist(meanfeat2train0, varfeat2train0 ,vartest1 )
        
        #implementing the Naive Bayes Classifier for Test1 data sample with Train0 data.
        probfor10=p21*0.5*p11
        
        #Computing estimated Probability density function for Test1 data sample with Train1 data.
        q11=ComputeNormaldist(meanfeat1train1, varfeatt1rain1 ,meanimage1 )
        q21=ComputeNormaldist(meanfeat2train1, varfeat2train1 ,vartest1 )
        
        #implementing the Naive Bayes Classifier for Test1 data sample with Train1 data.
        probfor11=q21*0.5*q11
        
        
        #Using Naive Bayes Classifier to produce a predicted label for testing sample.
        # Label the test sample with 0 or 1 based on the higher value.
        if(probfor10<probfor11):
            resultfor1.append(1)
        else:
            resultfor1.append(0)
    #result1 stores all the estimated labels from the data samples in Test1 Data.
    # Getting labels for Test0 data and Test1 data. 
    test0Label=(np.transpose(data0testLabel)).reshape(980)
    test1Label=(np.transpose(data1testLabel)).reshape(1135)
    
    #concatenating those two label data elements in a single list.
    test=list(test0Label)+list(test1Label)
    result=list(resultfor0+resultfor1)
    #Computing the classification accuracy for both “0” and “1” in the testing Data set.
    #with comparing predicted labels andgiven Test0Label and Test1Label files.
    countfor0=0
    countfor1=0
    count=0
    totalLabel=data0test.shape[2]+data1test.shape[2]
    for i in range(980):
        if (test0Label[i]==resultfor0[i]):
            countfor0=countfor0+1
    Accuracyfor0=(countfor0/data0test.shape[2])*100
    print()
    print("classification accuracy for test Data0 sample is : %0.3f%s" %(Accuracyfor0,'%'))
    
    for i in range(1135):
        if (test1Label[i]==resultfor1[i]):
            countfor1=countfor1+1
    Accuracyfor1=(countfor1/data1test.shape[2])*100
    print()
    print("classification accuracy for test Data1 sample is : %0.3f%s" %(Accuracyfor1,'%'))
    
    #Computing the overall classification accuracy  in the testing Data set. 
        
    for i in range(totalLabel):
        if (test[i]==result[i]):
            count=count+1
    Accuracy=(count/totalLabel)*100
    print()
    print("Overall classification accuracy is : %0.3f%s" %(Accuracy,'%'))
    


# In[ ]:




