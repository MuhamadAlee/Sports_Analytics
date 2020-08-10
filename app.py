import pickle
import numpy as np
from flask import Flask, render_template, request, redirect

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', teams=['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                                                'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                                                'Delhi Daredevils', 'Sunrisers Hyderabad'])


def regressing(userDetails):
    runs = int(userDetails['runs'])
    wickets = int(userDetails['wickets'])
    overs = float(userDetails['overs'])
    runs_last_5 = userDetails['runs_5']
    wickets_last_5 = userDetails['wickets_5']
    bat_team = userDetails['bat_team']
    bowl_team = userDetails['bowl_team']
    cols = ['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5',
            'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
            'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians',
            'bat_team_Other', 'bat_team_Rajasthan Royals',
            'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
            'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
            'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians',
            'bowl_team_Other', 'bowl_team_Rajasthan Royals',
            'bowl_team_Royal Challengers Bangalore',
            'bowl_team_Sunrisers Hyderabad']
    if bat_team != "Chennai Super Kings":
        bat_team_index = np.where(cols == "bat_team_" + bat_team)[0]
        bat_flag = 0
    else:
        bat_flag = 1

    if bowl_team != "Chennai Super Kings":
        bowl_team_index = np.where(cols == "bowl_team_" + bowl_team)[0]
        bowl_flag = 0
    else:
        bowl_flag = 1

    x = np.zeros(len(cols))
    x[0] = runs
    x[1] = wickets
    x[2] = overs
    x[3] = runs_last_5
    x[4] = wickets_last_5

    if bat_flag == 0:
        if bat_team_index >= 0:
            x[bat_team_index] = 1
    if bowl_flag == 0:
        if bowl_team_index >= 0:
            x[bowl_team_index] = 1

    file = open('Model/Prediction.pkl', 'rb')
    model = pickle.load(file)
    file.close()

    return model.predict([x])


@app.route('/getinfo', methods=['GET', 'POST'])
def getinfo():
    if request.method == 'POST':
        userDetails = request.form
        val = regressing(userDetails)
        val = round(val[0],0)
    return render_template('result.html', value=val)


if __name__ == '__main__':
    app.run(debug=True)
