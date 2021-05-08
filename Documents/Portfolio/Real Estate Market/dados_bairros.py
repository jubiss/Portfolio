import pandas as pd

df = pd.read_csv('valuation_data.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)

df = df[df['latitude'] !='None']
lat_long = df[['latitude','longitude']].drop_duplicates().values

from geopy.geocoders import  GoogleV3

geolocator = GoogleV3(api_key='Chave API')

new_info = []
new_locations = []
for i in range(9700,len(lat_long)):
    bairro = -1
    if ((float(lat_long[i][0])<=90) and (float(lat_long[i][0])>=-90) and 
        (float(lat_long[i][1])<=90) and (float(lat_long[i][1])>=-90)):
        
        location = geolocator.reverse(lat_long[i])
        for j in location.raw.get('address_components'):
            if 'sublocality' in j.get('types'):
                bairro = j.get('long_name')
    
        if bairro == -1:
            bairro = location.raw.get('address_components')[0].get('long_name')
        new_info.append([location.address,bairro,location.point])
        new_locations.append(location)
    else: 
        new_info.append(['error','invalid_loc',lat_long[i]])
        new_locations.append(lat_long[i])

#geo_info = pd.DataFrame(get_info)
"""
#i = 9741

new_info_dataframe = pd.DataFrame(new_info)

latitude = []
longitude = []
altitude = []
for i in new_info:
    latitude.append(i[2][0])
    longitude.append(i[2][1])
    try:
        altitude.append(i[2][2])
    except:
        altitude.append(0)


new_info_dataframe['latitude'] = latitude
new_info_dataframe['longitude'] = longitude
new_info_dataframe['altitude'] = altitude


new_info_dataframe.to_csv('Dados geográficos parciais 7000-9700.csv')
"""