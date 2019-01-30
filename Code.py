import copy
import math
from random import *
import matplotlib.pyplot as plt
import numpy as np
# from Code import evaluate
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def evaluate(l, prob, approach):
    rmse = 0
    wl = copy.deepcopy(l)
    h = 0
    while h < 168:
        x = 0

        while x < 8:
            y = 0

            while y < 8:
                if approach == "lr":
                    reg = LinearRegression()
                elif approach == 'ridge':
                    reg = linear_model.Ridge(alpha=.5)
                elif approach == 'rbfsvr':
                    reg = SVR(kernel='rbf', C=1e3, gamma=0.1)
                elif approach == 'lsvr':
                    reg = SVR(kernel='linear', C=1e3)
                elif approach == 'polysvr':
                    reg = SVR(kernel='poly', C=1e3, degree=2)
                hour = h
                i = 0
                xTrain = []
                yTrain = []

                while hour < 732:
                    r = random()
                    if r < prob:
                        wl[hour][x][y] = -1
                    else:
                        xTrain.append(hour)
                        yTrain.append(wl[hour][x][y])

                    hour += 168
                    i += 1
                if len(xTrain) != 0:
                    t = 0

                    while t < len(xTrain):
                        xTrain[t] = ([xTrain[t]])
                        t += 1
                    xTrain = np.array(xTrain)
                    yTrain = np.array(yTrain)

                    reg.fit(xTrain, yTrain)
                    hour = h

                    while hour < 732:
                        if wl[hour][x][y] == -1:
                            wl[hour][x][y] = int(reg.predict(np.array([[hour]]))[0])
                            if wl[hour][x][y] < 0:
                                wl[hour][x][y] = 0

                            areaMid = 0
                            num = 0
                            if x != 0:
                                if wl[hour][x - 1][y] != -1:
                                    areaMid += wl[hour][x - 1][y]
                                    num += 1
                            if x != 7:
                                if wl[hour][x + 1][y] != -1:
                                    areaMid += wl[hour][x + 1][y]
                                    num += 1
                            if y != 0:
                                if wl[hour][x][y - 1] != -1:
                                    areaMid += wl[hour][x][y - 1]
                                    num += 1
                            if y != 7:
                                if wl[hour][x][y + 1] != -1:
                                    areaMid += wl[hour][x][y + 1]
                                    num += 1
                            if num != 0:
                                areaMid = areaMid / num

                            hourMid = 0
                            num = 0
                            if hour != 0:
                                if wl[hour - 1][x][y] != -1:
                                    hourMid += wl[hour - 1][x][y]
                                    num += 1
                            if hour != 731:
                                if wl[hour + 1][x][y] != -1:
                                    hourMid += wl[hour + 1][x][y]

                                    num += 1

                            if num != 0:
                                hourMid = hourMid / num


                            num = 1
                            # if areaMid != 0:
                            #     wl[hour][x][y] += areaMid
                            #     num += 1
                            if hourMid != 0:
                                wl[hour][x][y] += hourMid
                                num += 1
                            wl[hour][x][y] = wl[hour][x][y] / num

                            #eval
                            # print (l[hour][x][y] - wl[hour][x][y])
                            rmse += ((l[hour][x][y] - wl[hour][x][y]) * (l[hour][x][y] - wl[hour][x][y]))
                        hour += 168

                y += 1
            x += 1
        h += 1
    return math.sqrt(rmse)

def evaluateDay(l, prob, approach):
    sum = 0
    rmse = 0
    wl = copy.deepcopy(l)
    h = 0
    while h < 24:
        x = 0

        while x < 8:
            y = 0

            while y < 8:
                if approach == "lr":
                    reg = LinearRegression()
                elif approach == 'ridge':
                    reg = linear_model.Ridge(alpha=.5)
                elif approach == 'rbfsvr':
                    reg = SVR(kernel='rbf', C=1e3, gamma=0.1)
                elif approach == 'lsvr':
                    reg = SVR(kernel='linear', C=1e3)
                elif approach == 'polysvr':
                    reg = SVR(kernel='poly', C=1e3, degree=2)
                hour = h
                i = 0
                xTrain = []
                yTrain = []

                while hour < 732:
                    r = random()
                    if r < prob:
                        sum += wl[hour][x][y]
                        wl[hour][x][y] = -1
                    else:
                        xTrain.append(hour)
                        yTrain.append(wl[hour][x][y])

                    hour += 24
                    i += 1
                if len(xTrain) != 0:
                    t = 0

                    while t < len(xTrain):
                        xTrain[t] = ([xTrain[t]])
                        t += 1
                    xTrain = np.array(xTrain)
                    yTrain = np.array(yTrain)

                    reg.fit(xTrain, yTrain)
                    hour = h

                    while hour < 732:
                        if wl[hour][x][y] == -1:
                            wl[hour][x][y] = int(reg.predict(np.array([[hour]]))[0])
                            if wl[hour][x][y] < 0:
                                wl[hour][x][y] = 0

                            #eval
                            # print (l[hour][x][y] - wl[hour][x][y])
                            rmse += ((l[hour][x][y] - wl[hour][x][y]) * (l[hour][x][y] - wl[hour][x][y]))
                        hour += 24
                y += 1
            x += 1
        h += 1
    print(sum)
    return math.sqrt(rmse)

def evalu(l, rl, wl, prob):
    rmse = 0
    vl = copy.deepcopy(l)
    h = 0
    while h < 24:
        x = 0

        while x < 8:
            y = 0

            while y < 8:
                hour = h
                i = 0

                while hour < 732:
                    r = random()
                    if r < prob:
                        vl[hour][x][y] = -1

                    hour += 24
                    i += 1
                    while hour < 732:
                        if vl[hour][x][y] == -1:
                            vl[hour][x][y] = (rl[hour][x][y] + wl[hour][x][y])/2
                            if vl[hour][x][y] < 0:
                                vl[hour][x][y] = 0

                            # eval
                            # print (l[hour][x][y] - wl[hour][x][y])
                            rmse += ((l[hour][x][y] - vl[hour][x][y]) * (l[hour][x][y] - vl[hour][x][y]))
                        hour += 24
                y += 1
            x += 1
        h += 1
    return math.sqrt(rmse)

def predict(wl, approach):
    h = 0
    while h < 168:
        x = 0

        while x < 8:
            y = 0

            while y < 8:
                if approach == "lr":
                    reg = LinearRegression()
                elif approach == 'ridge':
                    reg = linear_model.Ridge(alpha=.5)
                elif approach == 'rbfsvr':
                    reg = SVR(kernel='rbf', C=1e3, gamma=0.1)
                elif approach == 'lsvr':
                    reg = SVR(kernel='linear', C=1e3)
                elif approach == 'polysvr':
                    reg = SVR(kernel='poly', C=1e3, degree=2)
                hour = h
                i = 0
                xTrain = []
                yTrain = []
                while hour < 732:
                    if wl[hour][x][y] != -1:
                        xTrain.append(hour)
                        yTrain.append(wl[hour][x][y])
                    hour += 168
                    i += 1

                if len(xTrain) != 0:
                    t = 0

                    while t < len(xTrain):
                        xTrain[t] = ([xTrain[t]])
                        t += 1
                    xTrain = np.array(xTrain)
                    yTrain = np.array(yTrain)

                    reg.fit(xTrain, yTrain)
                    hour = h
                    while hour < 732:
                        if wl[hour][x][y] == -1:
                            wl[hour][x][y] = int(reg.predict(np.array([[hour]]))[0])
                            if wl[hour][x][y] < 0:
                                wl[hour][x][y] = 0

                            areaMid = 0
                            num = 0
                            if x != 0:
                                if wl[hour][x-1][y] != -1:
                                    areaMid += wl[hour][x-1][y]
                                    num +=1
                            if x != 7:
                                if wl[hour][x+1][y] != -1:
                                    areaMid += wl[hour][x+1][y]
                                    num += 1
                            if y != 0:
                                if wl[hour][x][y-1] != -1:
                                    areaMid += wl[hour][x][y-1]
                                    num += 1
                            if y != 7:
                                if wl[hour][x][y+1] != -1:
                                    areaMid += wl[hour][x][y+1]
                                    num += 1
                            if num != 0:
                                areaMid = areaMid/num


                            hourMid = 0
                            num = 0
                            if hour != 0:
                                if wl[hour-1][x][y] != -1:
                                    hourMid += wl[hour-1][x][y]
                                    num += 1
                            if hour != 731:
                                if wl[hour+1][x][y] != -1:
                                    hourMid += wl[hour+1][x][y]

                                    num += 1

                            if num != 0:
                                hourMid = hourMid / num


                            num = 1
                            # if areaMid != 0:
                            #     wl[hour][x][y] += areaMid
                            #     num += 1
                            if hourMid != 0:
                                wl[hour][x][y] += hourMid
                                num += 1
                            wl[hour][x][y] = wl[hour][x][y]/num
                        hour += 168
                else:

                    hour = h
                    while hour < 732:
                        if wl[hour][x][y] == -1:

                            hourMid = 0
                            num = 0
                            if hour != 0:
                                if wl[hour - 1][x][y] != -1:
                                    hourMid += wl[hour - 1][x][y]

                                    num += 1
                            if hour != 731:
                                if wl[hour + 1][x][y] != -1:
                                    hourMid += wl[hour + 1][x][y]

                                    num += 1

                            if num != 0:
                                hourMid = hourMid / num

                            wl[hour][x][y] = 0
                            num = 1
                            if hourMid != 0:
                                wl[hour][x][y] += 2*hourMid
                                num += 2
                            wl[hour][x][y] = wl[hour][x][y] / num

                        hour += 168

                y += 1
            x += 1
        h += 1

def predictDay(wl, approach):
    h = 0
    while h < 24:
        x = 0

        while x < 8:
            y = 0

            while y < 8:
                if approach == "lr":
                    reg = LinearRegression()
                elif approach == 'ridge':
                    reg = linear_model.Ridge(alpha=.5)
                elif approach == 'rbfsvr':
                    reg = SVR(kernel='rbf', C=1e3, gamma=0.1)
                elif approach == 'lsvr':
                    reg = SVR(kernel='linear', C=1e3)
                elif approach == 'polysvr':
                    reg = SVR(kernel='poly', C=1e3, degree=2)
                hour = h
                i = 0
                xTrain = []
                yTrain = []
                while hour < 732:
                    if wl[hour][x][y] != -1:
                        xTrain.append(hour)
                        yTrain.append(wl[hour][x][y])
                    hour += 24
                    i += 1

                if len(xTrain) != 0:
                    t = 0

                    while t < len(xTrain):
                        xTrain[t] = ([xTrain[t]])
                        t += 1
                    xTrain = np.array(xTrain)
                    yTrain = np.array(yTrain)

                    reg.fit(xTrain, yTrain)
                    hour = h
                    while hour < 732:
                        if wl[hour][x][y] == -1:
                            wl[hour][x][y] = int(reg.predict(np.array([[hour]]))[0])

                            if wl[hour][x][y] < 0:
                                wl[hour][x][y] = 0
                        hour += 24
                y += 1
            x += 1
        h += 1

def weekPlot(l, h, name):
    num = [[0 for x in range(5)] for x in range(8)]
    time = [[0 for x in range(5)] for x in range(8)]
    x = 0
    while x < 8:
        hour = h
        i = 0
        while hour < 732:
            time[x][i] = hour
            num[x][i] = l[hour][x][5]
            hour += 168
            i += 1
        x += 1

    fig = plt.figure()
    cnt = 1
    while cnt < 8:
        plt.plot(time[cnt], num[cnt])
        cnt += 1

    plt.title("Week Plot")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    # plt.show()
    fig.savefig(name + str(h) + "_Week_Plot.png")

def dayPlot(l, h, name):
    num = [[0 for x in range(31)] for x in range(8)]
    time = [[0 for x in range(31)] for x in range(8)]
    x = 0
    while x < 8:
        hour = h
        i = 0
        while hour < 732:
            time[x][i] = hour
            num[x][i] = l[hour][x][5]
            hour += 24
            i += 1
        x += 1

    fig = plt.figure()
    cnt = 1
    while cnt < 8:
        plt.plot(time[cnt], num[cnt])
        cnt += 1

    plt.title("Week Plot")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    # plt.show()
    fig.savefig(name + str(h) + "_Day_Plot.png")


fin = open('input.txt', 'r')
f = open('data.txt', 'r')
fout = open("output.txt","w+")

l = [[[0 for x in range(8)]for x in range(8)]for x in range(732)]
i = 0
while i < 732:
    if i == 0:
        cnt = 1
    else:
        cnt = 3
    for line in f:
        if cnt == 1 or cnt == 2:
            print ("first lines")
        else:
            l[i][cnt - 3] = [int(num) for num in line.split(' ')]
            if cnt >= 10:
                break
        cnt += 1
    i += 1


print ('1')
wl = copy.deepcopy(l)
predict(wl, "lr")

print ('2')
rl = copy.deepcopy(l)
predictDay(rl, "rbfsvr")


print ('3')
rwl = copy.deepcopy(l)
predict(rwl, "rbfsvr")

#evaluate
print("Linear Regression Prediction MSRE")
print(evaluate(l, 0.1, 'lr'))

print("My Prediction")
print(evaluateDay(l, 0.1, 'rbfsvr'))

print("polysvr Dayyy Prediction MSRE")
print(evaluate(l, 0.1, 'rbfsvr'))

print("Mid between two approach")
print(evalu(l, rl, wl, 0.1))


h = 0
for h in range (4):
    weekPlot(l, h, "Before_Prediction_")
    dayPlot(l, h, "Before_Prediction_")
h = 0
for h in range (2):
    weekPlot(wl, h, "After_Linear_Prediction_")
    weekPlot(rl, h, "After_PolySVR_Prediction_")
    dayPlot(wl, h, "After_Linear_Prediction_")
    dayPlot(rl, h, "After_Linear_Prediction_")



for line in fin:
    input = [int(num) for num in line.split(' ')]
    fout.write(str(rl[input[2]][input[0]][input[1]])+'\n')