import numpy as np 
import pandas as pd
import sys
import csv
import glob
import folium
from folium.plugins import HeatMap, MarkerCluster, FeatureGroupSubGroup

from event_prediction_data_ import *

file_cnt = 10
# file_cnt = 1 => 252 (144) ??
# file_cnt = 2 => 381 (288) ??
# file_cnt = 10 => 1401 (1440) ??

boundary = [0.00,0.20,0.40,0.60,0.80,sys.maxsize/100]

(data_dict, station_list, station_total_dock, location, max_val) = historical_station_logs(file_cnt)

data_list_categorical = []
for lock, station_name in enumerate(data_dict):
    cate_temp = []
    for x in data_dict[station_name]:
        per = x / station_total_dock[station_name]
        
        for ca in range(1,len(boundary)):
            if int(per*100) in range(int(boundary[ca-1]*100),int(boundary[ca]*100)):
                cate_temp.append(ca-1)
                break
                
    data_list_categorical.append(cate_temp)

'''
data_list_categorical =
[1:[1401],
 2:[1401],
 3:[1401],
 ...
 611:[1401]]
 '''

# map 초기설정

average_lat = sum(lat for lat, long in location.values()) / len(location)
average_long = sum(long for lat, long in location.values()) / len(location)

map = folium.Map(location=[average_lat, average_long],
                 zoom_start=12,
                 tiles = "CartoDB positron")

category_colors = {
    4: 'red',
    3: 'orange',
    2: 'green',
    1: 'blue',
    0: 'purple'
}

# 데이터셋 나누기

# time_step개씩 평균
def folium_dataset(hour = 12):
    time_step = 6 * hour
    averaged_data_list = []

    for station_data in data_list_categorical:
        averaged_station_data = []

        for i in range(0, len(station_data), time_step):
            group = station_data[i:i+time_step]
            group_average = round(sum(group) / len(group))
            averaged_station_data.append(group_average)

        averaged_data_list.append(averaged_station_data)
    
    return averaged_data_list

# 한 지도에 모두 표시 (multi layer)
def multi_layer_map_cate(dataset):
    # dataset = [time1, time2, time3, time4]
    groups = {}
    for i in range(5):
        group = FeatureGroupSubGroup(map, f"group{i}")
        map.add_child(group)
        groups[i] = group

    for station, category in zip(location.keys(), dataset):
        color = category_colors[category]
        folium.CircleMarker(location[station],
                            radius=2,
                            color = color,
                            fill = True).add_to(groups[category])

    folium.LayerControl(collapsed=False).add_to(map)

    return map

def multi_layer_map_time(cate, dataset):
    # dataset = [time1, time2, time3, time4]
    time_steps = len(dataset)
    groups = {}
    for i in range(time_steps):
        group = FeatureGroupSubGroup(map, f"time_period{i}")
        map.add_child(group)
        groups[i] = group
        
    for time_step, data in enumerate(dataset):
        for station, category in zip(location.keys(), data):
            if category == cate:
                folium.CircleMarker(location[station],
                                    color=category_colors[time_step],
                                    radius=2,
                                    fill = True).add_to(groups[time_step])


    folium.LayerControl(collapsed=False).add_to(map)
    
    return map


def single_layer_map(dataset):
    for idx, data in enumerate(dataset):
        for category in range(5):
            
            for station, value in zip(location.keys(), data):
                if value == category:
                    folium.CircleMarker(location[station],
                                        popup=f"{station} - Category: {value}",
                                        radius=2).add_to(map)
            
            map.save(f'{idx}_cate_{category}.html')



data_2 = folium_dataset(12)
am_data = [station_data[0] for station_data in data_2]
pm_data = [station_data[1] for station_data in data_2]


data_4 = folium_dataset(6)
time1 = [station_data[0] for station_data in data_4]
time2 = [station_data[1] for station_data in data_4]
time3 = [station_data[2] for station_data in data_4]
time4 = [station_data[3] for station_data in data_4]


#single_layer_map([time1, time2, time3, time4])
#multi_layer_map_time(4, [time1, time2, time3, time4]).save('map_cate4.html')
#multi_layer_map_cate(time2).save('map_time2.html')


'''
# 한 지도에 모두 표시 (색 변화)

for station, category in zip(location.keys(), am_data):
    color = category_colors[category]
    folium.CircleMarker(location[station],
                        color=color,
                        radius=2,
                        fill = True).add_to(map)

map.save('map.html')
'''

# file_cnt = day
# file_cnt 인자로 날짜/시간별 표시