import glob
import csv
import datetime
from traceback import print_tb


class trip_data:
    def __init__(
        self, trip_id, start_dt, start_d_id, end_dt, end_d_id, duration
    ) -> None:
        self.trip_id = trip_id
        self.start_date = datetime.date.fromisoformat(start_dt.split(" ")[0])
        self.start_time = datetime.time.fromisoformat(start_dt.split(" ")[1])
        self.start_dock_id = start_d_id
        self.end_date = datetime.date.fromisoformat(end_dt.split(" ")[0])
        self.end_time = datetime.time.fromisoformat(end_dt.split(" ")[1])
        self.end_dock_id = end_d_id
        self.duration = duration
        self.timeslot_conversion()
    
    def timeslot_conversion(self):
        start_t = self.start_time
        end_t = self.end_time
        self.start_t_cvt = (start_t.hour*60)+start_t.minute
        if start_t.second!=0:
            self.start_t_cvt+=1
        self.end_t_cvt = (end_t.hour*60)+end_t.minute
        if end_t.second!=0:
            self.end_t_cvt+=1
        print(start_t.isoformat())
        print(self.start_t_cvt)

    def __repr__(self) -> str:
        return str(
            "(trip_#)"
            + self.trip_id
            + ":: ("
            # + self.start_date.isoformat() #YYYY-MM-DD#
            + self.start_date.strftime("%Y%m%d")  # YYYYMMDD#
            + " "
            + self.start_time.isoformat()
            + ") "
            + self.start_dock_id
            + " -> ("
            # + self.end_date.isoformat() #YYYY-MM-DD#
            + self.end_date.strftime("%Y%m%d")  # YYYYMMDD#
            + " "
            + self.end_time.isoformat()
            + ") "
            + self.end_dock_id
        )


class bike_sharing_data:
    def __init__(self) -> None:
        self.station_id_name_pair = {}
        self.total_data = []

system_data = bike_sharing_data()
data_list = []
# file_dir_path = "./00.datasets/chicago_bike_dataset/"
# for csv_file in glob.glob(file_dir_path):
#     file = open(csv_file, 'r', encoding='utf-8')
#     csv_reader = csv.reader(file)
#     for line in csv_reader:
#         # line = list(map(int, line))
#         data_list.append(line[:])

file_path = "./00.datasets/chicago_bike_dataset/Divvy_Bicycle_Stations_Historical.csv"
file = open(file_path, "r", encoding="utf-8")
csv_reader = csv.reader(file)
count_limit = 1000
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

for i, line in enumerate(csv_reader):
    # line = list(map(int, line))
    if i >= count_limit:
        break
    # data_list.append(line[1:])
    if (
        line[1] != ""
        and line[2] != ""
        and line[5] != ""
        and line[3] != ""
        and line[7] != ""
        and line[4] != ""
        and line[6] != ""
        and line[8] != ""
    ):
        data_list.append(
            trip_data(line[1], line[2], line[5], line[3], line[7], line[4])
        )
        if not system_data.station_id_name_pair.__contains__(line[5]):
            system_data.station_id_name_pair[line[5]]=line[6]
        if not system_data.station_id_name_pair.__contains__(line[7]):
            system_data.station_id_name_pair[line[7]]=line[8]
        




print(system_data.station_id_name_pair)
print(len(system_data.station_id_name_pair))
# print(data_list)
for data in data_list:
    print(data)
