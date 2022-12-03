import glob
import csv
import datetime
from traceback import print_tb
import os

from regex import R


# system_data = bike_sharing_data()
# data_list = []
# # file_dir_path = "./00.datasets/chicago_bike_dataset/"
# for csv_file in glob.glob(file_dir_path):
#     file = open(csv_file, 'r', encoding='utf-8')
#     csv_reader = csv.reader(file)
#     for line in csv_reader:
#         # line = list(map(int, line))
#         data_list.append(line[:])

file_path = "./00.datasets/chicago_bike_dataset/Divvy_Bicycle_Stations_Historical.csv"
file = open(file_path, "r", encoding="utf-8")
csv_reader = csv.reader(file)
# count_limit = 1000
next(csv_reader)

"""
[0] ID,
[1] Timestamp,
[2] Station Name,
[3] Address,
[4] Total Docks,
[5] Docks in Service,
[6] Available Docks,
[7] Available Bikes,
[8] Percent Full,
[9] Status,
[10-12] Latitude,Longitude,Location,
[13] Record
"""

# line = ["adfjka","adfjka","adfjka","adfjka","adfjka",""]
# if line.__contains__(""):
#     print("done")
chicago_bike_data = "./00.datasets/chicago_bike_dataset/date/"
if not os.path.isdir(chicago_bike_data):
    os.mkdir(chicago_bike_data)

cur = None
file_writer =None
count = 0
converted_file=None
for i, line in enumerate(csv_reader):
    # line = list(map(int, line))
    # if not line.__contains__(""):
    time_stamp = line[1].split(" ")
    time_stamp = datetime.datetime.strptime(line[1],'%m/%d/%Y %I:%M:%S %p')
    date = time_stamp.strftime("%Y%m%d")
    # print(date.date)
    # datetime. strptime(date_time_str, '%d/%m/%y %H:%M:%S')
    count +=1
    if cur==None:
        cur = date
        converted_file = open(chicago_bike_data+"/"+cur+".csv",'w', newline='', encoding='utf-8')
        file_writer = csv.writer(converted_file)
        file_writer.writerow(line)
    elif cur == date:
        file_writer.writerow(line)
    elif cur != date:
        print(cur+" has "+str(count-1)+" rows.")
        count=1
        converted_file.close()
        cur = date
        converted_file = open(chicago_bike_data+"/"+cur+".csv",'w', newline='', encoding='utf-8')
        file_writer = csv.writer(converted_file)
        file_writer.writerow(line)
            
            
            
        
        


# for i in 
# converted_file = open(chicago_bike_data+"/ssn_compatible.csv",'w', newline='', encoding='utf-8')
# file_writer = csv.writer(converted_file)
# file_writer.writerow(["DATE","UUID_NO","TIME_INDEX","SRC","DST"])
# for data in converted_raw_data_list:
#     file_writer.writerow([data.date, data.user, data.time, data.source, data.destination])

# raw_file = "./00.datasets/2019_csv_files/*.csv"
# for i,csv_file in enumerate(glob.glob(raw_file)):
#     processed_list = raw_type_converter(csv_file)
#     converted_raw_data_list += processed_list




# print(system_data.station_id_name_pair)
# print(len(system_data.station_id_name_pair))
# # print(data_list)
# for data in data_list:
#     print(data)
