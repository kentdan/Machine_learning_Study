import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd

work_hash='9bb07a5e09b09da62cd27426972a16f29a246bec415aa14abce9e3a1'
cstr_MetaFileName='data/C2A2_data/BinSize_d{}.csv'.format(25)
cstrWorkFile='data/C2A2_data/BinnedCsvs_d{}/{}.csv'.format(25, work_hash)

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')
