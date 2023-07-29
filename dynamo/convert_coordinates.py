from bng_latlon import OSGB36toWGS84
# OSGB36toWGS84(538890, 177320)
# (51.477795, -0.001402)

from bng_latlon import WGS84toOSGB36
start_y_north, start_x_east = WGS84toOSGB36(53.474641, -2.244644)
end_x_east = start_x_east + 2000
end_y_north = start_y_north

end_y, end_x = OSGB36toWGS84(end_y_north, end_x_east)
print(end_y, end_x)