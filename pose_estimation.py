from numpy import *
import numpy as np
import scipy.optimize
import csv
import json


def rotmat2euler(mat, rads = 1):  # Function that computes roll pitch and yaw from rotation matrix

    phi = arctan2(mat[2, 0], mat[2, 1])
    theta = arccos(mat[2, 2])
    gamma = -1 * arctan2(mat[0, 2], mat[1, 2])

    if rads == 1:
        return phi, theta, gamma

    else:
        phi = (phi/3.14159265359) * 180
        theta = (theta/3.14159265359) * 180
        gamma = (gamma/3.14159265359) * 180

        return phi, theta, gamma


def rigid_transform_3D(A, B):  # Function the solves translation vector and rotation matrix given 2 views
    assert len(A) == len(B)
    N = A.shape[0];  # total points
    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)

    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB
    U, S, Vt = linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T

    return R, t


def open_config(file_name):  # Function that loads the model points from file
    model = []
    with open(file_name) as input_file:
        json_decode = json.load(input_file)
        for j in range(31):
            xyz = json_decode['lighthouse_config']['modelPoints'][j]
            model.append(xyz)
    return model


def load_points(file_name, lighthouse="L", count=1, steps=1, dlimiter=' '):  # Returns angles from csv file for any
    act = ""                                                                 # specified lighthouse and number of poses
    i = 0                                                                    # count will determine how many to poses to
    location_x = np.zeros(32)                                                # skip
    location_y = np.zeros(32)
    xyflag = 0
    parseflag = 0
    all_points = []
    all_ids = []
    with open(file_name, newline='') as csvfile:  # Loads csv
        track = csv.reader(csvfile, delimiter=dlimiter, quotechar='|')
        for sweep in track:
            if sweep[0] != lighthouse:  # Discards any data that's not related to lighthouse
                continue
            if act != sweep[1]:  # Will wait until second column changes X-Y. Buggy.
                i += 1
                act = sweep[1]
            if i > steps*2:  # Will wait until X and Y are collected and count the number of poses parsed
                xyflag = 1
                parseflag += 1
                i = 1
            if xyflag == 1:  # Will store X-Y values in list with IDs in order
                sensor_ids = []
                sensor_points = []
                for k in range(0, 31):
                    if location_x[k] != 0 and location_y[k] != 0:  # Rejects data without a pair
                        x = (location_x[k]/400000)*3.14159265359
                        y = (location_y[k]/400000)*3.14159265359
                        point = [x, y]
                        sensor_points.append(point)
                        sensor_ids.append(k)
                xyflag = 0
                all_points.append(sensor_points)
                all_ids.append(sensor_ids)
                location_x = np.zeros(32)
                location_y = np.zeros(32)
            if parseflag == count:  # Will break when number of poses are parsed
                break
            if sweep[1] == 'X':
                location_x[int(sweep[4])] = int(sweep[6].strip())  # + int(sweep[7])/2
            else:
                location_y[int(sweep[4])] = int(sweep[6].strip())  # + int(sweep[7])/2
    return all_points, all_ids

model_points = open_config("LHR-B4ABXXXX-Charles.json")
pointsa, idsa = load_points("live_center.csv", "L", count=1)
pose_num = 0
ids = []
zGuess = array([3, 3, 3, 3])  # Initial guess, may not converge.

for points in pointsa:
    print("Pose:", pose_num)
    ids = idsa[pose_num]
    pose_num += 1

    if len(ids) < 4:  # Ignores data with less than 4 points.
        continue

    k1 = pow((model_points[ids[0]][0] - model_points[ids[1]][0]), 2)\
         + pow((model_points[ids[0]][1] - model_points[ids[1]][1]), 2)\
         + pow((model_points[ids[0]][2] - model_points[ids[1]][2]), 2)
    k2 = pow((model_points[ids[2]][0] - model_points[ids[1]][0]), 2)\
         + pow((model_points[ids[2]][1] - model_points[ids[1]][1]), 2)\
         + pow((model_points[ids[2]][2] - model_points[ids[1]][2]), 2)
    k3 = pow((model_points[ids[0]][0] - model_points[ids[2]][0]), 2)\
         + pow((model_points[ids[0]][1] - model_points[ids[2]][1]), 2)\
         + pow((model_points[ids[0]][2] - model_points[ids[2]][2]), 2)
    k4 = pow((model_points[ids[0]][0] - model_points[ids[3]][0]), 2)\
         + pow((model_points[ids[0]][1] - model_points[ids[3]][1]), 2)\
         + pow((model_points[ids[0]][2] - model_points[ids[3]][2]), 2)

    theta_1 = points[0][0]
    theta_2 = points[1][0]
    theta_3 = points[2][0]
    theta_4 = points[3][0]
    phi_1 = points[0][1]
    phi_2 = points[1][1]
    phi_3 = points[2][1]
    phi_4 = points[3][1]


    def equations(z):  # Equation used by scipy
        r1 = z[0]
        r2 = z[1]
        r3 = z[2]
        r4 = z[3]

        F = empty(4)
        F[0] = pow(r1, 2)+pow(r2, 2)-2*r1*r2*(sin(theta_1)*sin(theta_2)*cos(phi_1-phi_2)+cos(theta_1)*cos(theta_2))-k1
        F[1] = pow(r2, 2)+pow(r3, 2)-2*r2*r3*(sin(theta_2)*sin(theta_3)*cos(phi_2-phi_3)+cos(theta_2)*cos(theta_3))-k2
        F[2] = pow(r3, 2)+pow(r1, 2)-2*r1*r3*(sin(theta_3)*sin(theta_1)*cos(phi_3-phi_1)+cos(theta_3)*cos(theta_1))-k3
        F[3] = pow(r4, 2)+pow(r1, 2)-2*r1*r4*(sin(theta_4)*sin(theta_1)*cos(phi_4-phi_1)+cos(theta_4)*cos(theta_1))-k4
        return F


    # Need to find the best nonlinear solver for this
    # z = scipy.optimize.fsolve(equations, zGuess, xtol=1.5e-05)
    z = scipy.optimize.least_squares(equations, zGuess)
    zGuess = z.x  # Using the previous result as initial guess
    n = 4
    # print('Solution: ', z.x)

    r1 = z.x[0]
    r2 = z.x[1]
    r3 = z.x[2]
    r4 = z.x[3]

    A_2 = np.matrix([[model_points[ids[0]][0], model_points[ids[0]][1], model_points[ids[0]][2]],
                     [model_points[ids[1]][0], model_points[ids[1]][1], model_points[ids[1]][2]],
                     [model_points[ids[2]][0], model_points[ids[2]][1], model_points[ids[2]][2]],
                     [model_points[ids[3]][0], model_points[ids[3]][1], model_points[ids[3]][2]]])
    B_2 = np.matrix([[r1*sin(phi_1)*cos(theta_1), r1*cos(phi_1), r1*sin(phi_1)*sin(theta_1)],
                     [r2*sin(phi_2)*cos(theta_2), r2*cos(phi_2), r2*sin(phi_2)*sin(theta_2)],
                     [r3*sin(phi_3)*cos(theta_3), r3*cos(phi_3), r3*sin(phi_3)*sin(theta_3)],
                     [r4*sin(phi_4)*cos(theta_4), r4*cos(phi_4), r4*sin(phi_4)*sin(theta_4)]])

    ret_R, ret_t = rigid_transform_3D(B_2, A_2)
    A2 = (ret_R * B_2.T) + tile(ret_t, (1, n))
    A2 = A2.T

    err = A2 - A_2
    err = multiply(err, err)
    err = sum(err)
    rmse = sqrt(err / n)

    phi, theta, gamma = rotmat2euler(ret_R, False)

    print("Translation Vector")
    print(ret_t)

    print("Rotation Matrix")
    print(ret_R)

    print("Euler Angles")
    print(phi, theta, gamma)

    dist_r = sqrt(ret_t[0]**2 + ret_t[1]**2 + ret_t[2]**2)
    print("Distance", dist_r)
    print("RMSE:", rmse)

