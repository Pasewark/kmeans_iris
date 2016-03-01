from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def kmeans(X,k,maxIterations,threshold):
    clusters={}
    means=[X[i] for i in np.random.choice(len(X),k)] #set initial means randomly
    imeans=[i for i in means]
    currentIteration=0
    while currentIteration<maxIterations:
        for i in range(len(X)): clusters[i]=[] #initialize groups for each mean
        currentIteration+=1
        for point in X: #assign points to clusters
            distances=[np.linalg.norm(point-m) for m in means] #calculate distances to current means
            clusters[distances.index(min(distances))].append(point)
        for i in range(k):
            if len(clusters[i])==0:
                biggestCluster=0
                for cluster in clusters:
                    if len(clusters[cluster])>len(clusters[biggestCluster]): biggestCluster=cluster
                means[i]=clusters[biggestCluster][np.random.choice(len(clusters[biggestCluster]))]
        #calculate new means and find biggest difference from old to new means
        maxDiff=0
        for i in range(k):
            if len(clusters[i])>0:
                tempMean=means[i]
                means[i]=np.sum(clusters[i],axis=0)/len(clusters[i])
                maxDiff=max(maxDiff,np.linalg.norm(tempMean-means[i]))
        if maxDiff<threshold:
            return clusters,means
    return clusters,means

def plotResults(clusters,means,attribute1,attribute2):
    group1x=[point[attribute1] for point in clusters[0]]
    group1y=[point[attribute2] for point in clusters[0]]
    group2x=[point[attribute1] for point in clusters[1]]
    group2y=[point[attribute2] for point in clusters[1]]
    group3x=[point[attribute1] for point in clusters[2]]
    group3y=[point[attribute2] for point in clusters[2]]
    plt.plot(group1x,group1y,'ro',group2x,group2y,'go',group3x,group3y,'bo',means[0][0],means[0][1],'rs',means[1][0],means[1][1],'gs',means[2][0],means[2][1],'bs')
    plt.show()
        
    

iris=datasets.load_iris()
X=iris.data
y=iris.target
clusters,means=kmeans(X,3,100,.01)
plotResults(clusters,means,0,1)
