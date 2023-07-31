import csv
import sys
import os
from datetime import datetime


def readFile(fileName,i):
    #f = open("download/18_5_18/Kickstarter039.csv", "rb")
    f = open(fileName, "rt", encoding="utf8")  #"rb"
    reader = csv.reader(f)
    # countriesEasternCentralEurope=['SI','HR','CZ','PL','BG','HU','RO','UA','SK','LV','LT','EE','RS']
    #countriesEasternCentralEurope = ['SI','HR', 'CZ', 'PL', 'BG', 'HU', 'RO', 'AL', 'SK', 'LV', 'LT', 'EE']
    countriesEasternCentralEurope = ['HR']
    rowsEasternCentralEurope=[]

    for j,row in enumerate(reader):
        # dates.append(row[12])
        # countries.append(row[4])
        if i==0 and j==0:
            rowsEasternCentralEurope.append(row)
        for country in countriesEasternCentralEurope:
            if row[22].find('"country":' + '"' + country) > -1:  #WRONG llok location, not country
                rowsEasternCentralEurope.append(row)
                break

    # next(reader)
    # for row in reader:
    #     rowsEasternCentralEurope.append(row)
    #answer=[value for value in row if value in countriesEasternCentralEurope]
    f.close()
    return rowsEasternCentralEurope


folder='c:/work/crowdfunding/download/06_22/'    #01_22   18_5_18
listing = os.listdir(folder)



rowsSelected=[]
# write header line
#f = open("download/12_7_18/Kickstarter039.csv", "rb")

# f = open("download/18_5_18/Kickstarter039.csv", "rt",encoding="utf8")
# reader = csv.reader(f)
# row1 = next(reader)
# rowsSelected.append(row1)
dates=[]
countries=[]
for i,file in enumerate(listing):

    answer=readFile(folder+file,i)
    rowsSelected=rowsSelected+answer
print("rowsSelected.__len__()")

resultFile = open('CampaignsHR_06_22.csv', 'w',encoding="utf8")

# with open('/pythonwork/thefile_subset11.csv', 'w', newline='') as outfile:
#     writer = csv.writer(outfile)
with resultFile:
    writer = csv.writer(resultFile)
    writer.writerows(rowsSelected)
resultFile.close()


