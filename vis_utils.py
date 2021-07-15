import numpy as np
import matplotlib.pyplot as plt


#def plot(flows, legend):
#    x=np.linspace(0,len(flows[0])/32, len(flows[0]))
#    leg = plt.legend()
#    leg_lines = leg.get_lines()
#    plt.setp(leg_lines, linewidth=4)
#    for s in flows:
#        plt.plot(x,s)
#    
#    for l in legend:
#        plt.legend(l)
#
#    plt.show()


def plot_dictionary(dictionary):
    plt.bar(dictionary.keys(), dictionary.values())#, width#, color='g')
    plt.show()


def plot(sequences, legends, frames = None, frequency = 8, ylabel = None):
    if frames is None:
        x = np.linspace(0, len(flows[0])/frequency, len(flows[0]))
    else:
        x = np.array(frames)/frequency

    if ylabel != None:
        plt.ylabel(ylabel)

    #leg = plt.legend()
    #leg_lines = leg.get_lines()
    #plt.setp(leg_lines, linewidth=4)
    for s in sequences:
        plt.plot(x,s)
    plt.legend(legends)
    plt.show()


def plot_probs_1(all_probs, labels, vid_probs, imu_probs, frames, vid_threshold, imu_threshold, title, pId, frequency = 8):
    vid_thresholds = np.full(len(frames), vid_threshold)
    imu_thresholds = np.full(len(frames), imu_threshold)
    x = np.array(frames)/frequency
    labels = [None if l==0 else l for l in labels]
    plt.ylabel(title + ' ({})'.format(pId))
    plt.plot(x,labels,'yo')
    #plt.plot(x,all_probs, color='orange')
    plt.plot(x,vid_probs, linewidth=0.5, alpha=1, color='blue')
    plt.plot(x,imu_probs, linewidth=1, alpha=0.8, color='orange')
    plt.plot(x,vid_thresholds, linewidth=0.5, linestyle='dashed')
    plt.plot(x,imu_thresholds, linewidth=0.5, linestyle='dashed')
    #plt.legend(['label', 'both', 'vidoe', 'imu', 'vidoe threshold', 'imu threshold'])
    plt.legend(['label', 'vidoe', 'imu', 'vidoe threshold', 'imu threshold'])
    plt.show()

def plot_probs_2(all_probs, vid_labels, imu_labels, vid_probs, imu_probs, frames, vid_threshold, imu_threshold, title, pId, frequency = 8):
    vid_thresholds = np.full(len(frames), vid_threshold)
    imu_thresholds = np.full(len(frames), imu_threshold)
    x = np.array(frames)/frequency
    vid_labels = [None if l==0 else 1.011 for l in vid_labels]
    imu_labels = [None if l==0 else 1.006 for l in imu_labels]
    plt.ylabel(title + ' ({})'.format(pId))
    plt.plot(x,vid_labels, 'r.')
    plt.plot(x,imu_labels, 'b.')
    plt.plot(x,vid_probs, linewidth=0.5, alpha=1, color='red')
    plt.plot(x,imu_probs, linewidth=1.5, alpha=0.4, color='blue')
    plt.plot(x,vid_thresholds, linewidth=0.5, linestyle='dashed', color='red')
    plt.plot(x,imu_thresholds, linewidth=0.5, linestyle='dashed', color='blue')
    plt.legend(['video label', 'imu label', 'vidoe', 'imu', 'vidoe threshold', 'imu threshold'])
    plt.show()



