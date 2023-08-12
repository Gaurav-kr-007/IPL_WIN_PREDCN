import streamlit as st
import pickle
import pandas as pd
st.title('IPL WIN PREDICT')

teams=['Sunrisers Hyderabad', 'Mumbai Indians', 
  'Royal Challengers Bangalore',
 'Kolkata Knight Riders',  'Kings XI Punjab' ,   
 'Chennai Super Kings', 'Rajasthan Royals','Delhi Capitals']

cities=['Mumbai' ,'Abu Dhabi' ,'Bangalore' ,'Delhi' ,'Mohali' ,'Kimberley' ,'Kolkata',
 'Jaipur', 'Hyderabad' ,'Sharjah' ,'Chennai', 'Chandigarh' ,'Johannesburg',   
 'Pune', 'Bengaluru' ,'Durban', 'Cape Town', 'Visakhapatnam', 'Dharamsala' ,  
 'Centurion', 'Ahmedabad' ,'Ranchi' ,'Cuttack' ,'Indore', 'Port Elizabeth' ,  
 'Nagpur' ,'Raipur' ,'Bloemfontein', 'East London']

pipe=pickle.load(open('pipe.pkl','rb'))
col1,col2=st.columns(2)

with col1:
    batting_team=st.selectbox('select the batting team',sorted(teams))

with col2:
    bowling_team=st.selectbox('select the bowling team',sorted(teams))

selected_city=st.selectbox('select the hosting city',sorted(cities))
target=st.number_input('target')

col3,col4,col5=st.columns(3)
with col3:
    score=st.number_input('Score')
with col4:
    over=st.number_input('over completed')
with col5:
    wickets=st.number_input('wickets out')

if st.button('Predict probability'):
    runs_left=target-score
    balls_left=120-(over*6)
    wickets=10-wickets
    crr=score/over
    rrr=(runs_left*6)/balls_left

    input_df=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],
                           'balls_left':[balls_left],'wicket_left':[wickets],'total_runs_x':[target],'crr':[crr],
                           'rrr':[rrr],})

    result=pipe.predict_proba(input_df)
    lose=result[0][0]
    win=result[0][1]
    st.header(batting_team+"-"+str(round(win*100))+"%")
    st.header(bowling_team+"-"+str(round(lose*100))+"%")
    