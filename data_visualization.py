import os
import json
import pymongo
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly_calplot import calplot
from pandas.api.types import CategoricalDtype


# Connect to the local MongoDB
@st.cache_resource  # to run only once
def connect():
    MONGO_USER = "evaparaschou"
    MONGO_PASSWORD = "6T1RKSxP6Qsy2uX2"
    url = "mongodb+srv://" + MONGO_USER + ":" + MONGO_PASSWORD + "@cluster0.lz2xfyb.mongodb.net/?retryWrites=true&w=majority"
    return pymongo.MongoClient(url)


# Get access to the database
@st.cache_resource()
def get_database():
    database = client.wdm_fitbit.fitbit_eva
    return database


if __name__ == '__main__':
    # page format
    st.set_page_config(layout="wide", page_title="WDM")

    # change background color
    st.markdown(""" <style> .css-18ni7ap , .css-fg4pbf{ background: #ecf9f2 ;} </style> """, unsafe_allow_html=True)

    # set title
    st.markdown(""" <style> h1{ font-size: 60px; font-family: 'Cooper Black'; color: #2d5986;} </style> """,
                unsafe_allow_html=True)
    st.markdown('<h1>A snapshot of user\'s daily life</h1>', unsafe_allow_html=True)

    # header section
    col1header, col2header = st.columns(2)
    st.markdown(""" <style> h2{ font-size: 50px; font-family: 'Cooper Black'; color: #6699cc;} </style> """, unsafe_allow_html=True)
    st.markdown(""" <style> .css-5rimss p {font-size:22px ; font-family: 'Arial;} </style> """, unsafe_allow_html=True)
    with col1header:
        st.markdown('<h2>Motivation</h2>', unsafe_allow_html=True)
        st.markdown(' Wearable technologies, such as smartwatches, have entered our life, providing numerous capabilities. '
            'Due to their ability to capture various physiological and psychological data, they can act as personal sports, sleep, '
            'and digital health coaches.')

    with col2header:
        st.markdown('<h2>Goal</h2>', unsafe_allow_html=True)
        st.markdown(""" This dashboard presents an initial analysis of a user's daily life regarding activity, sleep, heart data,
         and various earned badges and set goals. It also aims to identify and indicate daily and monthly patterns through 
         interactive visualizations.""")

    # connect
    client = connect()
    db = get_database()

    # ------------ General user statistics ------------#

    # collect the related data
    df = pd.DataFrame()
    df_activity = pd.DataFrame(list(db.find({"type": "activity"}, {"data.steps": 1, "data.caloriesOut": 1, "_id": 0})))
    df['steps'] = df_activity["data"].apply(lambda d: d["steps"])
    df['caloriesOut'] = df_activity["data"].apply(lambda d: d["caloriesOut"])
    df_sleep = pd.DataFrame(list(db.find({"type": "sleep"}, {"data.efficiency": 1, "_id": 0})))
    df['efficiency'] = df_sleep["data"].apply(lambda d: d["efficiency"])
    df_heart = pd.DataFrame(list(db.find({"type": "heart"}, {"data.mean_heart_rate": 1, "_id": 0})))
    df['heart'] = df_heart["data"].apply(lambda d: d["mean_heart_rate"])

    # change css
    st.markdown(""" <style> .css-1wivap2 > div > p {font-size: 20px; font-weight: bold; text-align: center; color: #6699cc;} </style> """, unsafe_allow_html=True)

    # compute the metrics
    col1stats, col2stats, col3stats, col4stats = st.columns(4)
    steps_mean = df['steps'].mean().round(2)
    col1stats.metric(label='Steps avg. value', value=steps_mean)
    caloriesOut_mean = df['caloriesOut'].mean().round(2)
    col2stats.metric(label='Calories burnt avg. value', value=caloriesOut_mean)
    efficiency_mean = df['efficiency'].mean().round(2)
    col3stats.metric(label='Sleep efficiency avg. value', value=efficiency_mean)
    heart_mean = df['heart'].mean().round(2)
    col4stats.metric(label='Heart rate avg. value', value=heart_mean)

    st.markdown('\n')
    st.markdown('\n')

    # Split all the page in 2 big columns
    st.markdown(""" <style> .css-keje6w {padding-left: 5px;} </style> """, unsafe_allow_html=True)
    col1main, col2main = st.columns(2)
    with col1main:
        # create 2 sub-columns for the smaller visualizations
        col1left, col2left = st.columns(2)
        with col1left:


            # ------------ Visualization 2 ------------#
            st.markdown('\n')

            # info section
            info_2 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-circle-info" style="color: #2d5986;"></i> Daily average sleep duration and efficiency score                                                                                                                                                                        
            '''
            st.markdown(info_2, unsafe_allow_html=True)
            st.markdown('\n')

            # collect the related data
            df_sleep = pd.DataFrame(
                list(db.find({"type": "sleep"}, {"data.date": 1, "data.duration": 1, "data.efficiency": 1, "_id": 0})))
            df_2 = pd.DataFrame()
            df_2["Date"] = df_sleep["data"].apply(lambda d: d['date'])
            df_2['Duration (hours)'] = (df_sleep["data"].apply(lambda d: d["duration"])) / (1000 * 3600)
            df_2['Efficiency (%)'] = df_sleep["data"].apply(lambda d: d["efficiency"])
            # date engineering
            df_2['Date'] = pd.to_datetime(df_2['Date'], errors='coerce')
            df_2['days_of_week_numbers'] = df_2["Date"].dt.dayofweek
            df_2['Day'] = df_2['Date'].dt.day_name()
            df_2 = df_2.groupby(['Day'], as_index=False).mean()
            cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            cat_type = CategoricalDtype(categories=cats, ordered=True)
            df_2 = df_2.groupby(['Day']).sum().reindex(cats)
            df_2 = df_2.reset_index()
            df_2 = df_2.drop(columns=["days_of_week_numbers"])

            # create the menu
            plot_menu = {"Duration (hours)": "Duration (hours)", "Efficiency (%)": "Efficiency (%)"}
            plot_var_name = st.selectbox("Select column to visualize", list(plot_menu.keys()), 0)
            plot_var = plot_menu[plot_var_name]

            # create the plot
            c = alt.Chart(df_2).mark_line().encode(x=alt.X('Day', sort=None), y=plot_var)
            st.altair_chart(c, use_container_width=True)

            # result
            result_2 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-magnifying-glass" style="color: #2d5986;"></i> User's sleep duration is significantly 
            higher during weekends and unstable during weekdays, while sleep efficiency is, in general, high and stable.                                                                                                                                                                 
            '''
            st.markdown(result_2, unsafe_allow_html=True)
            st.markdown('\n')


            # ------------ Visualization 4 ------------#
            st.markdown('\n')

            # info section
            info_4 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-circle-info" style="color: #2d5986;"></i> Overview of number of steps on March                                                                                                                                                                      
            '''
            st.markdown(info_4, unsafe_allow_html=True)
            st.markdown('\n')

            # collect the related data
            df_activity4 = pd.DataFrame(list(db.find({"type": "activity"}, {"data.steps": 1, "data.date": 1, "_id": 0})))
            df_4 = pd.DataFrame()
            df_4["Date"] = df_activity4["data"].apply(lambda d: d["date"])
            df_4["Steps"] = df_activity4["data"].apply(lambda d: d["steps"])
            # date engineering
            df_4['Date'] = pd.to_datetime(df_4['Date'], errors='coerce')
            df_4['month'] = df_4["Date"].dt.month
            df_4 = df_4.loc[df_4['month'] == 3]
            df_4.drop(columns=['month'], inplace=True)
            df_4 = df_4.rename(columns={'Date': 'index'}).set_index('index')

            # create the plot
            st.bar_chart(df_4)

            # result
            result_4 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-magnifying-glass" style="color: #2d5986;"></i> User's number of steps on March varies a lot every day, 
            indicating an unstable everyday life this month.                                                                                                                                                              
            '''
            st.markdown(result_4, unsafe_allow_html=True)
            st.markdown('\n')


            # ------------ Visualization 6 ------------#
            st.markdown('\n')

            # info section
            info_6 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-circle-info" style="color: #2d5986;"></i> Relatedness of calories with respective goal                                                                                                                                                             
            '''
            st.markdown(info_6, unsafe_allow_html=True)
            st.markdown('\n')

            # collect the related data
            df_goals = pd.DataFrame(list(db.find({"type": "goals"}, {"data.caloriesOut": 1})))
            df_goals['caloriesOut'] = df_goals["data"].apply(lambda d: d["caloriesOut"])
            caloriesOut_goal = df_goals['caloriesOut'].values[0]

            # create the plot
            fig = go.Figure(data=[go.Pie(labels=['Avg. Calories Burnt', 'Distance from Goal'],
                                         values=[caloriesOut_mean, caloriesOut_goal - caloriesOut_mean],
                                         hole=.4,
                                         showlegend=False,
                                         marker=dict(colors=['#636EFA', '#EF553B'],
                                                     line=dict(color='#FFFFFF', width=2),
                                                     ))],
                            layout=dict(width=350, height=400, margin=dict(l=30, r=30, t=40, b=30), ))
            fig.update_traces(hole=.4, marker=dict(colors=['#636EFA', '#EF553B'], line=dict(color='#FFFFFF', width=2)))
            fig.update_layout(title='Calories Burnt')
            st.plotly_chart(fig, use_container_width=True)

            # result
            result_6 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-magnifying-glass" style="color: #2d5986;"></i> It is evident that the goal concerning calories was on 
            average achieved for the most days.                                                                                                                                                                     
            '''
            st.markdown(result_6, unsafe_allow_html=True)
            st.markdown('\n')


            # ------------ Visualization 8 ------------#
            st.markdown('\n')

            # info section
            info_8 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-circle-info" style="color: #2d5986;"></i> Number of different badge types                                                                                                                                                                     
            '''
            st.markdown(info_8, unsafe_allow_html=True)
            st.markdown('\n')

            # collect the data
            df_badges = pd.DataFrame(list(db.find({"type": "badges"}, {"data.badgeType": 1, "_id": 0})))
            df_8 = pd.DataFrame()
            df_8['Badge type'] = df_badges["data"].apply(lambda d: d["badgeType"])
            df_8 = df_8['Badge type'].value_counts()

            # create the plot
            st.bar_chart(df_8)

            # results section
            result_8 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-magnifying-glass" style="color: #2d5986;"></i> User\'s most common badge type is related to the daily number of steps.                                                                                                                                                                      
            '''
            st.markdown(result_8, unsafe_allow_html=True)
            st.markdown('\n')

        with col2left:


            # ------------ Visualization 3 ------------#
            st.markdown('\n')

            # info section
            info_3 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-circle-info" style="color: #2d5986;"></i> Sleep stages distribution per day                                                                                                                                                                    
            '''
            st.markdown(info_3, unsafe_allow_html=True)
            st.markdown('\n')

            # collect the related data
            df_sleep = pd.DataFrame(list(db.find({"type": "sleep"}, {"data.date": 1, "data.deep": 1, "data.light": 1, "data.rem": 1, "data.wake": 1, "_id": 0})))
            df_3 = pd.DataFrame()
            df_3["date"] = df_sleep["data"].apply(lambda d: d['date'])
            df_3["Deep"] = df_sleep["data"].apply(lambda d: d["deep"]) / 60
            df_3["Light"] = df_sleep["data"].apply(lambda d: d["light"]) / 60
            df_3["Rem"] = df_sleep["data"].apply(lambda d: d["rem"]) / 60
            df_3["Wake"] = df_sleep["data"].apply(lambda d: d["wake"]) / 60
            # convert the date column to a pandas datetime object and extract the day of the week
            df_3['date'] = pd.to_datetime(df_3['date'])
            df_3['Days of week'] = df_3['date'].dt.day_name()
            # melt the DataFrame to long format, grouping by day of the week
            melted_df = pd.melt(df_3, id_vars=['Days of week'], value_vars=['Deep', 'Light', 'Rem', 'Wake'], var_name='Sleep stages')
            # group the melted DataFrame by day of the week and count type, and sum the value column
            grouped_df = melted_df.groupby(['Days of week', 'Sleep stages']).mean().reset_index()
            grouped_df = grouped_df.rename(columns={'value': 'Sleep Duration(Hours)'})

            # create the plot
            chart = alt.Chart(grouped_df).mark_bar().encode(
                x=alt.X('Days of week', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
                y='Sleep Duration(Hours)', color='Sleep stages', tooltip=['Days of week', 'Sleep stages', 'Sleep Duration(Hours)']
            ).properties(width=alt.Step(80), title="Average sleep categories duration per day of the week")
            st.altair_chart(chart, use_container_width=True)

            # result
            result_3 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-magnifying-glass" style="color: #2d5986;"></i> Sleep duration is increased in weekends, whereas light 
            sleep stage is observed at high percentages each day of the week.                                                                                                                                                                                                                                                                                                                      
            '''
            st.markdown(result_3, unsafe_allow_html=True)
            st.markdown('\n')


            # ------------ Visualization 5 ------------#
            st.markdown('\n')

            # info section
            info_5 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-circle-info" style="color: #2d5986;"></i> Average active minutes per day                                                                                                                                                                   
            '''
            st.markdown(info_5, unsafe_allow_html=True)
            st.markdown('\n')

            # collect the related data
            df_activity2 = pd.DataFrame(list(db.find({"type": "activity"}, {"data.date": 1, "data.sedentaryMinutes": 1, "data.veryActiveMinutes": 1,
                                                      "data.lightlyActiveMinutes": 1, "data.moderatelyActiveMinutes": 1, "_id": 0})))
            df_5 = pd.DataFrame()
            df_5["date"] = df_activity2["data"].apply(lambda d: d["date"])
            df_5["Sedentary Minutes"] = df_activity2["data"].apply(lambda d: d["sedentaryMinutes"]) / 60
            df_5["Very Active Minutes"] = df_activity2["data"].apply(lambda d: d["veryActiveMinutes"]) / 60
            df_5["Lightly Active Minutes"] = df_activity2["data"].apply(lambda d: d["lightlyActiveMinutes"]) / 60
            df_5["Moderately Active Minutes"] = df_activity2["data"].apply(lambda d: d["moderatelyActiveMinutes"]) / 60
            # convert the date column to a pandas datetime object and extract the day of the week
            df_5['date'] = pd.to_datetime(df_5['date'])
            df_5['Days of week'] = df_5['date'].dt.month_name()
            # group the data by day of the week
            grouped_df = df_5.groupby('Days of week').mean().reset_index()

            # create the plot
            chart = alt.Chart(grouped_df).mark_bar().encode(
                x=alt.X('Days of week:N', sort=['February', 'March', 'April'], axis=alt.Axis(title='Months')),
                y=alt.Y('value:Q', axis=alt.Axis(title='Activity Duration (Hours)')),
                color=alt.Color('variable:N', scale=alt.Scale(scheme='category10'), legend=alt.Legend(title='Intensity of Activity'))
            ).transform_fold(['Sedentary Minutes', 'Very Active Minutes', 'Lightly Active Minutes', 'Moderately Active Minutes'],
                as_=['variable', 'value']).properties(width=alt.Step(1),  # Adjust the bar width
                title="Average active minutes per month")
            st.altair_chart(chart, use_container_width=True)

            # result
            result_5 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-magnifying-glass" style="color: #2d5986;"></i> April was the most active month, whereas in all months 
            the average very active and moderatevely active minutes are significantly low.                                                                                                                                                                      
            '''
            st.markdown(result_5, unsafe_allow_html=True)
            st.markdown('\n')


            # ------------ Visualization 7 ------------#
            st.markdown('\n')

            # info section
            info_7 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-circle-info" style="color: #2d5986;"></i> Correlation between heart rate and steps                                                                                                                                                                 
            '''
            st.markdown(info_7, unsafe_allow_html=True)
            st.markdown('\n')

            # collect data
            df_activity = pd.DataFrame(list(db.find({"type": "activity"}, {"data.steps": 1, "_id": 0})))
            df_7 = pd.DataFrame()
            df_7['steps'] = df_activity["data"].apply(lambda d: d["steps"])
            df_7['steps'] = (df_7['steps'] - df_7['steps'].min()) / (df_7['steps'].max() - df_7['steps'].min())
            df_heart = pd.DataFrame(list(db.find({"type": "heart"}, {"data.mean_heart_rate": 1, "_id": 0})))
            df_7['heart rate'] = df_heart["data"].apply(lambda d: d["mean_heart_rate"])
            df_7['heart rate'] = (df_7['heart rate'] - df_7['heart rate'].min()) / (
                        df_7['heart rate'].max() - df_7['heart rate'].min())

            # create the plot
            fig = px.scatter(df_7, x='steps', y='heart rate')
            st.plotly_chart(fig, use_container_width=True)

            # result
            result_7 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-magnifying-glass" style="color: #2d5986;"></i> Steps and heart rate appear to have no correlation based on the scatter plot.                                                                                                                                                                      
            '''
            st.markdown(result_7, unsafe_allow_html=True)
            st.markdown('\n')


            # ------------ Visualization 9 ------------#
            st.markdown('\n')

            # info section
            info_9 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-circle-info" style="color: #2d5986;"></i> Calendar of number of achieved goals                                                                                                                                                                 
            '''
            st.markdown(info_9, unsafe_allow_html=True)
            st.markdown('\n')

            # collect the related data
            df_activity = pd.DataFrame(list(db.find({"type": "activity"}, {"data.steps": 1, "data.caloriesOut": 1, "data.floors": 1, "data.date": 1, "_id": 0})))
            df_goals = pd.DataFrame(list(db.find({"type": "goals"}, {"data.steps": 1, "data.caloriesOut": 1, "data.floors": 1})))
            df_goals['steps'] = df_goals["data"].apply(lambda d: d["steps"])
            df_goals['caloriesOut'] = df_goals["data"].apply(lambda d: d["caloriesOut"])
            df_goals['floors'] = df_goals["data"].apply(lambda d: d["floors"])
            df9 = pd.DataFrame(np.zeros((len(df_activity))))
            for i, row in df_activity.iterrows():
                if row['data']['caloriesOut'] >= df_goals['caloriesOut'][0]:
                    df9.loc[i, :] += 1
                if row['data']['steps'] >= df_goals['steps'][0]:
                    df9.loc[i, :] += 1
                if row['data']['floors'] > df_goals['floors'][0]:
                    df9.loc[i, :] += 1
            dates_df = df_activity['data'].apply(lambda x: x['date'])
            x = [0]
            dates = pd.to_datetime(dates_df.values.tolist())
            values = df9[0].values.tolist()
            dummy_df = pd.DataFrame({"dates": dates, "values": values})

            # create the plot
            fig = calplot(dummy_df, x="dates", y="values", start_month=2, end_month=4, total_height=300)
            st.plotly_chart(fig, use_container_width=True)

            # result
            result_9 = '''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
            <i class="fa-solid fa-magnifying-glass" style="color: #2d5986;"></i> The darker the color, the most badges earned. 
            We observe that the user has no standard daily patterns regarding the achieved goals, only that March was more intense than the other months.                                                                                                                                                              
            '''
            st.markdown(result_9, unsafe_allow_html=True)
            st.markdown('\n')

    with col2main:


        # ------------ Visualization 10 ------------#
        st.markdown('\n')

        # info section
        info_10 = '''
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
        <i class="fa-solid fa-circle-info" style="color: #2d5986;"></i> Level of engagement                                                                                                                                                                
        '''
        st.markdown(info_10, unsafe_allow_html=True)
        st.markdown('\n')

        # general comment
        df_activity = pd.DataFrame(list(db.find({"type": "activity"}, {"data.steps": 1, "data.floors": 1, "_id": 0})))
        days = str(len(df_activity))
        st.markdown(""" In total, the dataset contains user's information for """ + days + """ days. """)

        # separate columns
        col1stats2, col2stats2, col3stats2 = st.columns(3)

        # collect data
        df_10_activity = pd.DataFrame(list(db.find({"type": "activity"})))
        df_10_sleep = pd.DataFrame(list(db.find({"type": "sleep"})))
        df_10_heart = pd.DataFrame(list(db.find({"type": "heart"})))

        # display metrics
        col1stats2.metric(label='Activity Data', value=str(len(df_10_activity)) + '/' + days + ' days')
        col2stats2.metric(label='Sleep Data', value=str(len(df_10_sleep)) + '/' + days + ' days')
        col3stats2.metric(label='Heart Data', value=str(len(df_10_heart)) + '/' + days + ' days')

        # result
        result_10 = '''
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
        <i class="fa-solid fa-magnifying-glass" style="color: #2d5986;"></i> The user did not wear the smartwatch during the night
        so the analysis of these data might need to be revised.                                                                                                                                                         
        '''
        st.markdown(result_10, unsafe_allow_html=True)
        st.markdown('\n')


        # ------------ Visualization 11 ------------#
        st.markdown('\n')

        # info section
        info_11 = '''
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
        <i class="fa-solid fa-circle-info" style="color: #2d5986;"></i> Relatedness of steps and floors with respective goals                                                                                                                                                              
        '''
        st.markdown(info_11, unsafe_allow_html=True)
        st.markdown('\n')

        # collect the related data
        df_activity = pd.DataFrame(list(db.find({"type": "activity"}, {"data.steps": 1, "data.floors": 1, "_id": 0})))
        df_11 = pd.DataFrame()
        df_11['steps'] = df_activity["data"].apply(lambda d: d["steps"])
        df_11['floors'] = df_activity["data"].apply(lambda d: d["floors"])
        df_goals = pd.DataFrame(list(db.find({"type": "goals"}, {"data.steps": 1, "data.floors": 1})))
        df_goals['steps'] = df_goals["data"].apply(lambda d: d["steps"])
        df_goals['floors'] = df_goals["data"].apply(lambda d: d["floors"])
        df_11['steps_goal'] = df_goals['steps'][0]
        df_11['floors_goal'] = df_goals['floors'][0]

        # create the menu
        plot_menu = {"Steps": "steps", "Floors": "floors"}
        plot_var_name = st.selectbox("Select column to visualize", list(plot_menu.keys()), 0)
        plot_var = plot_menu[plot_var_name]

        # create the plot
        if plot_var == 'steps':
            df_11_plot = df_11[['steps', 'steps_goal']]
            st.line_chart(df_11_plot)
        if plot_var == 'floors':
            df_11_plot = df_11[['floors', 'floors_goal']]
            st.line_chart(df_11_plot)

        # result
        result_11 = '''
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
        <i class="fa-solid fa-magnifying-glass" style="color: #2d5986;"></i> User achieves the step goals almost the half of the days, 
        while most commonly exceeds its floors goal.                                                                                                                                                       
        '''
        st.markdown(result_11, unsafe_allow_html=True)
        st.markdown('\n')


        # ------------ Visualization 12 ------------#
        st.markdown('\n')

        # info section
        info_12 = '''
                   <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
                   <i class="fa-solid fa-circle-info" style="color: #2d5986;"></i> Heart rate (avg, man, mix per day) distribution                                                                                                                                                                      
                   '''
        st.markdown(info_12, unsafe_allow_html=True)
        st.markdown('\n')

        # collect the related data
        df_heart = pd.DataFrame(list(db.find({"type": "heart"},
                                             {"data.date": 1, "data.mean_heart_rate": 1, "data.max_heart_rate": 1,
                                              "data.min_heart_rate": 1, "_id": 0})))
        df_12 = pd.DataFrame()
        df_12["Date"] = df_heart["data"].apply(lambda d: d['date'])
        df_12['Date'] = pd.to_datetime(df_12['Date'], errors='coerce')
        df_12['Mean heart'] = df_heart["data"].apply(lambda d: d["mean_heart_rate"])
        df_12['Max heart'] = df_heart["data"].apply(lambda d: d["max_heart_rate"])
        df_12['Min heart'] = df_heart["data"].apply(lambda d: d["min_heart_rate"])

        # create the menu
        plot_menu = {"Mean heart rate": "Mean heart", "Max heart rate": "Max heart", "Min heart rate": "Min heart"}
        plot_var_name = st.selectbox("Select column to visualize", list(plot_menu.keys()), 0)
        plot_var = plot_menu[plot_var_name]

        # create the plot
        st.area_chart(df_12, x="Date", y=plot_var)

        # result
        result_12 = '''
                   <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">                                                                                                    
                   <i class="fa-solid fa-magnifying-glass" style="color: #2d5986;"></i> User's mean heart rate ranges mostly between 80 and 90, 
                   while on 19/03/2023 user appears more abnormal patterns.                                                                                                                                                                      
                   '''
        st.markdown(result_12, unsafe_allow_html=True)
        st.markdown('\n')


    # footer section
    footer = """
    <style>
        footer{
            visibility: visible;
            background-color: #2d5986;
        }
        footer:after{
            content: 'Created by: Athanasiadis Georgios, Gkiouzelis Nikolaos, Paraschou Eva | Github link: https://github.com/eparascho/mental-health-analytics | DWS MSc @AUTh, Web Data Mining, May 2023';
            display: block;
            position: relative;
            color: white;
            font-size: 18px;
        }
        .css-164nlkn{
            padding: 1rem 4rem 7rem 2rem;
        }
    </style>
    """
    st.markdown(footer, unsafe_allow_html=True)
