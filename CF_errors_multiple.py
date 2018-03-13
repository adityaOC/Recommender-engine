# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 19:35:05 2017

@author: adityagaonkr
"""
import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

testUserId = 7
userData = pd.DataFrame() 
movieInfoData = pd.DataFrame() 
userItemRatingMatrix = pd.DataFrame() 
movieInfo = pd.DataFrame() 
rmseDf = pd.DataFrame(columns=['itemId','rating','PR'])
resultDf = pd.DataFrame(columns=['userId','NOU','MAE','RMSE'])

def favoriteMovies(activeUser,N):
    #1. subset the dataframe to have the rows corresponding to the active user
    # 2. sort by the rating in descending order
    # 3. pick the top N rows
    topMovies=pd.DataFrame.sort_values(
        movieInfoData[movieInfoData.userId==activeUser],['rating'],ascending=[0])[:N]
    # return the title corresponding to the movies in topMovies 
    return list(topMovies.title)

#print(favoriteMovies(5,3)) # Prin



#print(userItemRatingMatrix.head())






from scipy.spatial.distance import correlation 
def similarity(user1,user2):
    user1=np.array(user1)-np.nanmean(user1) # we are first normalizing user1 by 
    # the mean rating of user 1 for any movie. Note the use of np.nanmean() - this 
    # returns the mean of an array after ignoring and NaN values 
    user2=np.array(user2)-np.nanmean(user2)
    # Now to find the similarity between 2 users
    # We'll first subset each user to be represented only by the ratings for the 
    # movies the 2 users have in common 
    commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
    # Gives us movies for which both users have non NaN ratings 
    if len(commonItemIds)==0:
        # If there are no movies in common 
        return 0
    else:
        user1=np.array([user1[i] for i in commonItemIds])
        user2=np.array([user2[i] for i in commonItemIds])
        return correlation(user1,user2)
    



# Using this similarity function, let's find the nearest neighbours of the active user
def nearestNeighbourRatings(activeUser,K):
    # This function will find the K Nearest neighbours of the active user, then 
    # use their ratings to predict the activeUsers ratings for other movies 
    similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,
                                  columns=['Similarity'])
    # Creates an empty matrix whose row index is userIds, and the value will be 
    # similarity of that user to the active User
    for i in userItemRatingMatrix.index:
        similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],
                                          userItemRatingMatrix.loc[i])
        # Find the similarity between user i and the active user and add it to the 
        # similarityMatrix 
    similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,
                                              ['Similarity'],ascending=[0])
    # Sort the similarity matrix in the descending order of similarity 
    nearestNeighbours=similarityMatrix[:K]
    # The above line will give us the K Nearest neighbours 
    
    # We'll now take the nearest neighbours and use their ratings 
    # to predict the active user's rating for every movie
    neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
    # There's something clever we've done here
    # the similarity matrix had an index which was the userId, By sorting 
    # and picking the top K rows, the nearestNeighbours dataframe now has 
    # a dataframe whose row index is the userIds of the K Nearest neighbours 
    # Using this index we can directly find the corresponding rows in the 
    # user Item rating matrix 
    predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])
    # A placeholder for the predicted item ratings. It's row index is the 
    # list of itemIds which is the same as the column index of userItemRatingMatrix
    #Let's fill this up now
    for i in userItemRatingMatrix.columns:
        # for each item 
        predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])
        sumOfSimilarity = 0
        # start with the average rating of the user
        for j in neighbourItemRatings.index:
            # for each neighbour in the neighbour list 
            if userItemRatingMatrix.loc[j,i]>0:
                # If the neighbour has rated that item
                # Add the rating of the neighbour for that item
                #    adjusted by 
                #    the average rating of the neighbour 
                #    weighted by 
                #    the similarity of the neighbour to the active user
                predictedRating += (userItemRatingMatrix.loc[j,i]
                                    -np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
                sumOfSimilarity = sumOfSimilarity + nearestNeighbours.loc[j, 'Similarity']
        # We are out of the loop which uses the nearest neighbours, add the 
        # rating to the predicted Rating matrix
         
        if sumOfSimilarity !=0:
            predictItemRating.loc[i,'Rating']=predictedRating/sumOfSimilarity
        else:
            predictItemRating.loc[i, 'Rating'] = predictedRating
    return predictItemRating



def topNRecommendations(activeUser,N):
    predictItemRating=nearestNeighbourRatings(activeUser,10)
    """ Use the 10 nearest neighbours to find the predicted ratings"""
    moviesAlreadyWatched=list(userItemRatingMatrix.loc[activeUser]
                              .loc[userItemRatingMatrix.loc[activeUser]>0].index)
    # find the list of items whose ratings which are not NaN
    """predictItemRating=predictItemRating.drop(moviesAlreadyWatched)"""
    topRecommendations=pd.DataFrame.sort_values(predictItemRating,
                                                ['Rating'],ascending=[0])[:N]
    # This will give us the list of itemIds which are the top recommendations 
    # Let's find the corresponding movie titles
    
    #print("moviesAlreadyWatched : {}".format(moviesAlreadyWatched))
    #print("predictItemRating : {}".format(list(predictItemRating["Rating"])))
    
    #print("topRecommendations :{}".format(topRecommendations))
    
    
    itemList = rmseDf['itemId'].values.tolist()
    itemList.sort()
    for itemId in itemList:
        itemId = int(itemId)
        #print(int(itemId))
        #print("predictItemRating {}: {}".format(itemId,predictItemRating.iloc[itemId]['Rating']))
        rmseDf.loc[rmseDf['itemId'] == itemId, 'PR'] = predictItemRating.iloc[itemId]['Rating']
        
    
    topRecommendationTitles=(movieInfo.loc[movieInfo.itemId.isin(topRecommendations.index)])
    return list(topRecommendationTitles.title)
    



#-----------------------------------------------------------------------------
def mainMethod():
    
    
    dataFile='/Users/adityagaonkr/Downloads/RS/ml-100k/u.data'
    userDataBeforeFilter=pd.read_csv(dataFile,sep="\t",header=None,
                 names=['userId','itemId','rating','timestamp'])
                                
    #---------------Drop records with user id
                 
    dropIndexes = userDataBeforeFilter.index[userDataBeforeFilter['userId'] == testUserId].tolist()
    userData = userDataBeforeFilter


    #---------------Store original ratings of test user
    
    
    i = 0
    for index in dropIndexes:
    
        rmseDf.loc[i] = userDataBeforeFilter.iloc[dropIndexes[i]]
        i+=1

   
                 
    movieInfoFile="/Users/adityagaonkr/Downloads/RS/ml-100k/u.item"

    movieInfo=pd.read_csv(movieInfoFile,sep="|", header=None, index_col=False,
                     names=["itemId","title"], usecols=[0,1],encoding = 'latin')

    movieInfoData=pd.merge(userData,movieInfo,left_on='itemId',right_on="itemId")

    userItemRatingMatrix=pd.pivot_table(userData, values='rating',
                                    index=['userId'], columns=['itemId'])


    #-----------------------------------------------------------------------------

    activeUser=testUserId
    #print("hello")
    #print(favoriteMovies(activeUser,5),"\n Recommendations: \n",topNRecommendations(activeUser,3))
    print("\n")
    print("Favorite movies: {}".format(favoriteMovies(activeUser,10)))
    print("\n")
    print("Recommendations: {}".format(topNRecommendations(activeUser,3)))

    print("----------RMSE----------------------------")
    #print("users average rating = {}".format(np.mean(rmseDf['rating'].values.tolist())))
    #rmseDf[rmseDf < 0] =np.mean(rmseDf['rating'].values.tolist())
    
    y_true = rmseDf['rating'].values.tolist()
    y_pred = rmseDf['PR'].values.tolist()
    RMSE = mean_squared_error(y_true, y_pred)
    print("RMSE  = {}".format(RMSE))

    
    y_true = rmseDf['rating'].values.tolist()
    y_pred = rmseDf['PR'].values.tolist()
    mabse = mean_absolute_error(y_true, y_pred)
    print("mean_absolute_error  = {}".format(mabse))
    
    resultDf.loc[resultDf['userId'] == activeUser, 'MAE'] = mabse
    resultDf.loc[resultDf['userId'] == activeUser, 'RMSE'] = RMSE
    resultDf.loc[resultDf['userId'] == activeUser, 'NOU'] =len(dropIndexes)

#for index in originalVector:
#    print(index,originalVector[index])
  




dataFile='/Users/adityagaonkr/Downloads/RS/ml-100k/u.data'
userDataBeforeFilter=pd.read_csv(dataFile,sep="\t",header=None,
                 names=['userId','itemId','rating','timestamp'])
                                
#---------------Drop records with user id
                 
dropIndexes = userDataBeforeFilter.index[userDataBeforeFilter['userId'] == testUserId].tolist()
userData = userDataBeforeFilter


    #---------------Store original ratings of test user
    
    
i = 0
for index in dropIndexes:
    
   rmseDf.loc[i] = userDataBeforeFilter.iloc[dropIndexes[i]]
   i+=1

   
                 
movieInfoFile="/Users/adityagaonkr/Downloads/RS/ml-100k/u.item"

movieInfo=pd.read_csv(movieInfoFile,sep="|", header=None, index_col=False,
                     names=["itemId","title"], usecols=[0,1],encoding = 'latin')

movieInfoData=pd.merge(userData,movieInfo,left_on='itemId',right_on="itemId")

userItemRatingMatrix=pd.pivot_table(userData, values='rating',
                                    index=['userId'], columns=['itemId'])


 #-----------------------------------------------------------------------------

activeUser=testUserId

print("\n")
print("Favorite movies: {}".format(favoriteMovies(activeUser,10)))
print("\n")
print("Recommendations: {}".format(topNRecommendations(activeUser,3)))

print("----------RMSE----------------------------")


y_true = rmseDf['rating'].values.tolist()
y_pred = rmseDf['PR'].values.tolist()
error = mean_squared_error(y_true, y_pred)
print("RMSE  = {}".format(error))


y_true = rmseDf['rating'].values.tolist()
y_pred = rmseDf['PR'].values.tolist()
mabse = mean_absolute_error(y_true, y_pred)
print("mean_absolute_error  = {}".format(mabse))



testUserList = [1,50,100,150,200,250]
resultDf['userId'] = userData['userId'].unique()
for u in range(1,6):
    
    testUserId = u
    mainMethod()  





print("------------------------_DONE------------------")




