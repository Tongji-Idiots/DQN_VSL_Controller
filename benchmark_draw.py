import numpy as np
import xml.etree.ElementTree as ET
from scipy.interpolate import spline
import matplotlib.pyplot as plt

def draw():
    tree = ET.parse("benchmark.data.xml")
    root = tree.getroot()
    ori, cum, nn = root.find("origin"), root.find("custom"), root.find("nn")

    total_time = 500
    frame = np.arange(1,total_time)

    ori_ms, ori_ttt = np.zeros(total_time), np.zeros(total_time)
    cum_ms, cum_ttt = np.zeros(total_time), np.zeros(total_time)
    nn_ms, nn_ttt = np.zeros(total_time), np.zeros(total_time)

    for data in ori:
        ori_ms[int(data.attrib["frame"])] = float(data.attrib["ms"])
        ori_ttt[int(data.attrib["frame"])] = float(data.attrib["ttt"])

    for data in cum:
        cum_ms[int(data.attrib["frame"])] = float(data.attrib["ms"])
        cum_ttt[int(data.attrib["frame"])] = float(data.attrib["ttt"])

    for data in nn:
        nn_ms[int(data.attrib["frame"])] = float(data.attrib["ms"])
        nn_ttt[int(data.attrib["frame"])] = float(data.attrib["ttt"])   

    point_num = 500
    frame_new = np.linspace(frame.min(), frame.max(), point_num)

    ori_ms_smooth = spline(frame, ori_ms, frame_new)
    ori_ttt_smooth = spline(frame, ori_ttt, frame_new)
    
    cum_ms_smooth = spline(frame, cum_ms, frame_new)
    cum_ttt_smooth = spline(frame, cum_ttt, frame_new)
    
    nn_ms_smooth = spline(frame, nn_ms, frame_new)
    nn_ttt_smooth = spline(frame, nn_ttt, frame_new)

    plt.plot(frame_new,ori_ms_smooth, color="red", label="ori_ms")
    plt.plot(frame_new,cum_ms_smooth, color="green", label="cus_ms")
    plt.plot(frame_new,nn_ms_smooth, color="blue", label="nn_ms")
    plt.show()

if __name__ == "__main__":
    draw()