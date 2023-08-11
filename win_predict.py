
import pandas as pd
import numpy as np

matches=pd.read_csv('matches.csv')
ball=pd.read_csv('deliveries.csv.zip')
# print(ball.head())
# print(matches.head())
# grouping of attributes as per our requirement for prediction

# delivery=ball.groupby(['match_id','inning']).sum()['total_runs']
# print(delivery.head())

# to get new dataset form from above we will reset index it.

delivery=ball.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
# print(delivery)
total_score_df=delivery[delivery['inning']==1]
# print(total_score_df) 

# now merging total_score_df with matches table using join operation
match_df=matches.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')
# print(match_df.to_string()) 
# print(match_df['team1'].unique()) # we eliminate those dont play now,and make a list of
# of teams playing..and also replace old name with new  like delhi

teams=['Sunrisers Hyderabad', 'Mumbai Indians', 
  'Royal Challengers Bangalore',
 'Kolkata Knight Riders',  'Kings XI Punjab' ,   
 'Chennai Super Kings', 'Rajasthan Royals','Delhi Capitals']

#  replacing names of the team for both the team i.e. team1 and bowling_team
match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

#now for deccan chargers to sunrisers hyderabad.
match_df['team1']=match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
# print(match_df[['team1','team2']])

# now removing teams
match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]
# print(match_df.shape)
# print(match_df['team1'].isin(teams))

# now removing docber louis matches
# print(match_df['dl_applied'].value_counts())
match_df=match_df[match_df['dl_applied']==0]
# print(match_df.shape)

# now we will take required columns from match_df to join with ball/delivery table so
match_df=match_df[['match_id','city','winner','total_runs']]
# print(match_df)
ball['batting_team']=ball['batting_team'].str.replace('Delhi Daredevils','Delhi Capitals')
ball['bowling_team']=ball['bowling_team'].str.replace('Delhi Daredevils','Delhi Capitals')

#now for deccan chargers to sunrisers hyderabad.
ball['batting_team']=ball['batting_team'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
ball['bowling_team']=ball['bowling_team'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
# print(ball[['batting team','bowling_team']])

# now removing teams
ball=ball[ball['batting_team'].isin(teams)]
ball=ball[ball['bowling_team'].isin(teams)]
# now will join match_id with ball table.
delivery_df=match_df.merge(ball,on='match_id')
# print(ball.shape)
# print(delivery_df)0
# print(ball.columns)
# print(match_df.columns)
# print(delivery_df.columns)
print((delivery_df['batting_team']=='Deccan Chargers').value_counts())

# delivery_df contains both innings history so we will take 2nd chasing innings only so
delivery_df=delivery_df[delivery_df['inning']==2]
# print(delivery_df.shape)
# print(delivery_df.columns)



#  now we will calculate runs_left,balls_left, and wicket_left.
# current total runs : and notice cumsum function.
delivery_df["current_runs"]=delivery_df.groupby("match_id").cumsum()["total_runs_y"]
# print(delivery_df)

delivery_df["runs_left"]=delivery_df["total_runs_x"]-delivery_df["current_runs"]
# print(delivery_df["runs_left"])

#  now we will calculate balls left in the game, as we know over and ball so
delivery_df["balls_left"]=120-((delivery_df["over"]-1)*6+delivery_df["ball"])
# other option...delivery_df["balls_left"]=126-delivery_df["over"]*6+delivery_df["ball"]
# print(delivery_df["balls_left"])

# wicket left calculation 
# print(delivery_df)
delivery_df["player_dismissed"]=delivery_df["player_dismissed"].fillna("0")
delivery_df["player_dismissed"]=delivery_df["player_dismissed"].apply(lambda x:x if x=="0" else "1")
# now converting string 1 and 0 into intezer throughout in column
delivery_df["player_dismissed"]=delivery_df["player_dismissed"].astype('int')
wickets=delivery_df.groupby('match_id').cumsum()["player_dismissed"].values
delivery_df["wicket_left"]=10-wickets
# print(delivery_df["wicket_left"])

#  now we have to get current run rate,required run rate
# so crr=current_run/over..and rrr=run_left/over and for over in crr case is ((120-delivery_df["balls_left"])/6) and
# over for rrr is delivery_df["balls_left"]/6 so
delivery_df["crr"]=delivery_df["current_runs"]/((120-delivery_df["balls_left"])/6)
delivery_df["rrr"]=delivery_df["runs_left"]/(delivery_df["balls_left"]/6)
# print(delivery_df[["crr","rrr"]])          remember to use 2 brackets.

def result(row):
   return 1 if row["winner"]==row["batting_team"] else 0
  # print(row["batting_team"])

delivery_df["result"]=delivery_df.apply(result,axis=1)
# print(delivery_df.shape)
# now we will take our required columns for prediction so
final_df=delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wicket_left','total_runs_x','crr','rrr','result']]
# print(final_df.shape)
# print(final_df)

# now since all ball are in sequence so to remove any kind of bias in prediction 

final_df=(final_df.sample(final_df.shape[0]))
# print(final_df.sample()) 

# now we are checking is there null value in final dataset and also infity because div with zero will give inf so we will remove all rows with null
# print(final_df.isnull().sum())
final_df.dropna(inplace=True)
# below creates inf as check using 
# print(final_df.describe())
final_df=final_df[final_df['balls_left']!=0]
# print(final_df.shape)
#  so our dataset now prepared now its time to put it on different models so yeah
from sklearn.model_selection import train_test_split
x=final_df.iloc[:,:-1]
y=final_df.iloc[:,-1]
# print(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=1)
# print(x_train.describe())
# print(x_train)

# now we will convert some columns which are string which might affect later
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
trf=ColumnTransformer([('trf',OneHotEncoder(drop='first',sparse=False),['batting_team','bowling_team','city'])],remainder='passthrough')

# then we will import pipeline and logisitic regression as we need prob
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
pipe=Pipeline(steps=[
   ('step1',trf),
   ('step2',LogisticRegression(solver='liblinear'))
  # ('step2',RandomForestClassifier())
])
# here in step2 of pipe we can put randomforest and accuracy is above 95 ,to get which team will win it is good but we want prob at each stage 
# so and it takes little more to time to cal,we can use cross valdn and tuning.
pipe.fit(x_train,y_train)
input=pd.DataFrame(x_test)
y_predict=pipe.predict(input)
# print(y_predict)
cols=['batting_team','bowling_team','city','runs_left','balls_left','wicket_left','total_runs_x','crr','rrr']
input=pd.DataFrame([['Sunrisers Hyderabad','Royal Challengers Bangalore','Bangalore',2,40,8,180,13.35,0.3]],columns=cols)
print(pipe.predict(input))
from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test,y_predict))
# prob=pipe.predict_proba(input)
# print(prob)

# import pickle
# pickle.dump(pipe,open('pipe.pkl','wb'))




