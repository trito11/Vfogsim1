'''
Các hàm tính toán 

khoảng các giữa 2 tọa độ(long1,lat1,long2,lat2) => distance

Tốc độ truyền(channel_banwidth, pr, distance, path_loss_exponent, sigmasquare) => time
'''

from math import radians, cos, sin, asin, sqrt
import numpy as np
import random
from geopy.distance import geodesic
from pyproj import Proj, transform
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # haversine formula 
    dlat = lat2 - lat1
    dlon = lon2 - lon1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    # Convert kilometers to meters
    
    return km

'''
Toc do truyen ko day
'''
def getRateTransData(channel_banwidth, pr, distance, path_loss_exponent, sigmasquare):
    return (channel_banwidth * np.log2(
            1 + pr / np.power(distance,path_loss_exponent) / sigmasquare
        )
    ) 

# Hàm chuyển đổi thời gian dạng chuỗi thành giây
def time_to_seconds(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

def calculate_intermediate_coordinate(lat1, lon1, lat2, lon2, ratio):
    # Tính tọa độ của xe tại thời điểm x theo tỷ lệ n
    constant = np.pi / 180
    R = 6371
    φ1 = lat1 * constant
    λ1 = lon1 * constant
    φ2 = lat2 * constant
    λ2 = lon2 * constant
    
    delta = np.arccos(np.sin(φ1) * np.sin(φ2) + np.cos(φ1) * np.cos(φ2) * np.cos(λ2 - λ1))  # Great circle distance
    d = R * delta  # Distance between two points
    
    d *= ratio  # Scale the distance
    
    y = np.sin(λ2 - λ1) * np.cos(φ2)
    x = np.cos(φ1) * np.sin(φ2) - np.sin(φ1) * np.cos(φ2) * np.cos(λ2 - λ1)
    θ = np.arctan2(y, x)
    brng = (θ * 180 / np.pi + 360) % 360  # in degrees
    brng = brng * constant

    φ3 = np.arcsin(np.sin(φ1) * np.cos(d / R) + np.cos(φ1) * np.sin(d / R) * np.cos(brng))
    λ3 = λ1 + np.arctan2(np.sin(brng) * np.sin(d / R) * np.cos(φ1), np.cos(d / R) - np.sin(φ1) * np.sin(φ3))

    return φ3 / constant, λ3 / constant

def utm_to_latlon(easting, northing, zone_number=35, zone_letter='V'):
    p1 = Proj(proj='utm', zone=zone_number, ellps='WGS84', datum='WGS84')
    lon, lat = p1(easting, northing, inverse=True)
    return lat, lon

def distance_between_points(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers