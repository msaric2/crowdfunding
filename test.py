import csv
import sys
import os


def readFile(fileName):
    #f = open("download/18_5_18/Kickstarter039.csv", "rb")
    f = open(fileName, "rb")
    reader = csv.reader(f)
    # countriesEasternCentralEurope=['SI','HR','CZ','PL','BG','HU','RO','UA','SK','LV','LT','EE','RS']
    #countriesEasternCentralEurope = ['SI','HR', 'CZ', 'PL', 'BG', 'HU', 'RO', 'AL', 'SK', 'LV', 'LT', 'EE']
    countriesEasternCentralEurope = ['HR']
    rowsEasternCentralEurope=[]

    for row in reader:
        for country in countriesEasternCentralEurope:
            if row[32].find('"country":' + '"' + country) > -1:
            #if row[4].find(country) > -1:    #2018 country is in columnn row[32], 2019 in the column row[4]   !!!!!!!
                rowsEasternCentralEurope.append(row)
                break

    # next(reader)
    # for row in reader:
    #     rowsEasternCentralEurope.append(row)
    #answer=[value for value in row if value in countriesEasternCentralEurope]
    f.close()
    return rowsEasternCentralEurope


folder='c:/work/crowdfunding/download/18_5_18/'    #18_5_18
listing = os.listdir(folder)



rowsSelected=[]
# write header line
#f = open("download/12_7_18/Kickstarter039.csv", "rb")
f = open("download/18_5_18/Kickstarter039.csv", "rb")
reader = csv.reader(f)
row1 = next(reader)
rowsSelected.append(row1)

for file in listing:
    answer=readFile(folder+file)
    rowsSelected=rowsSelected+answer
print rowsSelected.__len__()

resultFile = open('CampaignsHRv2.csv', 'wb')
with resultFile:
    writer = csv.writer(resultFile)
    writer.writerows(rowsSelected)
resultFile.close()


