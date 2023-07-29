# Tunnel Length: 2 km (2000 meters)
tunnel_length = 2000

# Step Size: 0.1 meters

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from bng_latlon import WGS84toOSGB36

# Starting Longitude (X) and Latitude (Y) coordinates for Greater Manchester
start_longitude = -2.244644
start_latitude = 53.474641

# Ending Longitude (X) and Latitude (Y) coordinates for Greater Manchester
end_longitude = -2.244748
end_latitude = 53.492618

# Depth Change per Step: 10 meters / Number of Steps
# depth_change_per_step = 10 / (tunnel_length / step_size)

steps = 100
step_size_x = (start_longitude - end_longitude) / steps
step_size_y = (start_latitude - end_latitude) / steps
X = np.arange(end_longitude, start_longitude+1e-10, step_size_x)

def between(start, end):
    return (end + start) / 2

middle_longitude = between(end_longitude, start_longitude)
print([end_longitude, end_longitude + 0.00001, middle_longitude, start_longitude - 0.00001, start_longitude])

target_z_cs = [0., -2.0, -20, -2.0, 0.]
# target_z_cs = [0., -0.5, -1, -0.5, 0.]
csZ = CubicSpline([end_longitude, end_longitude + 0.00001, middle_longitude, start_longitude - 0.00001, start_longitude],
                  target_z_cs)
csY = CubicSpline([end_longitude, middle_longitude, start_longitude],
                  [end_latitude, between(end_latitude, start_latitude)+0.001, start_latitude])

# Y = np.arange(end_latitude, start_latitude+1e-10, step_size_y)
Y = csY(X)
Z = csZ(X)
Z[-1] = 0.

y_northings_1st, x_eastings_1st = WGS84toOSGB36(Y[0], X[0])
with open('D:\\UNIVERSITY OF PORTSMOUTH\\Dissertation Research\\DYNAMO\\Tunnel\\tunnel_coords.csv', 'w+') as file:
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        z = Z[i]
        y_northings, x_eastings = WGS84toOSGB36(y, x)
        y_northings -= y_northings_1st
        x_eastings  -= x_eastings_1st

        scale_factor = 1000
        x_eastings  *= scale_factor
        y_northings *= scale_factor
        z           *= scale_factor
        if i % 10 == 0:
            file.write(f"{x_eastings},{y_northings},{z},My_Solid,Void\n")
        else:
            file.write(f"{x_eastings},{y_northings},{z},,\n")


# exit()
# # Print the generated coordinates
# for coord in coordinates:
#     print(f"X (Longitude): {coord[0]:.6f}, Y (Latitude): {coord[1]:.6f}, Z (Depth): {coord[2]:.1f}")

# if (i // 10) % 2 == 1:

# XY Projection
plt.figure(figsize=(8, 6))
plt.plot(X, Y, '-o', label='Tunnel Path', markersize=2)
plt.xlabel('Longitude (X)')
plt.ylabel('Latitude (Y)')
plt.title('XY Projection of the Tunnel')
plt.legend()
plt.grid(True)
plt.show()

# XZ Projection
plt.figure(figsize=(8, 6))
plt.plot(X, Z, '-o', label='Tunnel Path', markersize=2)
plt.xlabel('Longitude (X)')
plt.ylabel('Depth (Z)')
plt.title('XZ Projection of the Tunnel')
plt.legend()
plt.grid(True)
plt.show()


exit()