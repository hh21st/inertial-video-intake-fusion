import argparse
import plotly
import plotly.graph_objects as go

def plot(title, labels, links):
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 10,
          line = dict(color = "black", width = 0.5),
          label = labels,
          color = "rgba(20, 20, 20, 0.8)"
        ),
        link = links )])

    fig.update_layout(title_text=title, font_size=14)
    plotly.offline.plot(fig)

def plot_tp_fn(title, tp_vid_only_fus_tp, tp_vid_only_fus_fn, tp_imu_only_fus_tp, tp_imu_only_fus_fn, tp_vid_overlap_fus_tp, tp_vid_overlap_fus_fn, fn_overlap_vid_fus_tp, fn_overlap_vid_fus_fn, add_labels = True):

    TP_V_O = ['TP (V-I)',0]
    TP_I_O = ['TP (I-V)',1]
    TP_V_I = ['TP (I∩V)',2]
    FN_V_I = ['FN (I∩V)',3]
    TP_Fus = ['TP (Fusion)',4]
    FN_Fus = ['FN (Fusion)',5]

    dict_tp_vid_only_fus_tp    = [TP_V_O[1], TP_Fus[1], tp_vid_only_fus_tp   , 'rgba(237, 125, 49, 0.8)']
    dict_tp_vid_only_fus_fn    = [TP_V_O[1], FN_Fus[1], tp_vid_only_fus_fn   , 'rgba(237, 125, 49, 0.95)']
    dict_tp_imu_only_fus_tp    = [TP_I_O[1], TP_Fus[1], tp_imu_only_fus_tp   , 'rgba(68, 114, 196, 0.6)']
    dict_tp_imu_only_fus_fn    = [TP_I_O[1], FN_Fus[1], tp_imu_only_fus_fn   , 'rgba(68, 114, 196, 0.6)']
    dict_tp_vid_overlap_fus_tp = [TP_V_I[1], TP_Fus[1], tp_vid_overlap_fus_tp, 'rgba(165, 165, 165, 0.8)']
    dict_tp_vid_overlap_fus_fn = [TP_V_I[1], FN_Fus[1], tp_vid_overlap_fus_fn, 'rgba(140, 140, 140, 0.95)']
    dict_fn_overlap_vid_fus_tp = [FN_V_I[1], TP_Fus[1], fn_overlap_vid_fus_tp, 'rgba(255, 192, 0, 0.95)']
    dict_fn_overlap_vid_fus_fn = [FN_V_I[1], FN_Fus[1], fn_overlap_vid_fus_fn, 'rgba(255, 192, 0, 0.8)']
    if add_labels:
        labels = [TP_V_O[0],TP_I_O[0],TP_V_I[0],FN_V_I[0],TP_Fus[0],FN_Fus[0]]
    else:
        labels = ['','','','','','']
    links = dict(
          source = [dict_tp_vid_only_fus_tp[0],dict_tp_vid_only_fus_fn[0],dict_tp_imu_only_fus_tp[0],dict_tp_imu_only_fus_fn[0],dict_tp_vid_overlap_fus_tp[0],dict_tp_vid_overlap_fus_fn[0],dict_fn_overlap_vid_fus_tp[0],dict_fn_overlap_vid_fus_fn[0]],
          target = [dict_tp_vid_only_fus_tp[1],dict_tp_vid_only_fus_fn[1],dict_tp_imu_only_fus_tp[1],dict_tp_imu_only_fus_fn[1],dict_tp_vid_overlap_fus_tp[1],dict_tp_vid_overlap_fus_fn[1],dict_fn_overlap_vid_fus_tp[1],dict_fn_overlap_vid_fus_fn[1]],
          value  = [dict_tp_vid_only_fus_tp[2],dict_tp_vid_only_fus_fn[2],dict_tp_imu_only_fus_tp[2],dict_tp_imu_only_fus_fn[2],dict_tp_vid_overlap_fus_tp[2],dict_tp_vid_overlap_fus_fn[2],dict_fn_overlap_vid_fus_tp[2],dict_fn_overlap_vid_fus_fn[2]],
          color  = [dict_tp_vid_only_fus_tp[3],dict_tp_vid_only_fus_fn[3],dict_tp_imu_only_fus_tp[3],dict_tp_imu_only_fus_fn[3],dict_tp_vid_overlap_fus_tp[3],dict_tp_vid_overlap_fus_fn[3],dict_fn_overlap_vid_fus_tp[3],dict_fn_overlap_vid_fus_fn[3]]
      )
    plot(title, labels, links)

def plot_fp1(title, fp_vid_only_fus_fp, fp_vid_only_not_in_fus_fp, fp_imu_only_fus_fp, fp_imu_only_not_in_fus_fp, fp_vid_overlap_fus_fp, fp_vid_overlap_not_in_fus_fp, add_labels = True):

    FP_V_O = ['FP1 (V-I)',0]
    FP_I_O = ['FP1 (I-V)',1]
    FP_V_I = ['FP1 (I∩V)',2]
    FP_Fus = ['FP1 (Fusion)',3]
    FPnFus = ['FP1 (Not in Fusion)',4]

    dict_fp_vid_only_fus_fp           = [FP_V_O[1], FP_Fus[1], fp_vid_only_fus_fp          , 'rgba(196, 237, 49, 0.8)']
    dict_fp_vid_only_not_in_fus_fp    = [FP_V_O[1], FPnFus[1], fp_vid_only_not_in_fus_fp   , 'rgba(196, 237, 49, 0.8)']
    dict_fp_imu_only_fus_fp           = [FP_I_O[1], FP_Fus[1], fp_imu_only_fus_fp          , 'rgba(69, 179, 196, 0.8)']
    dict_fp_imu_only_not_in_fus_fp    = [FP_I_O[1], FPnFus[1], fp_imu_only_not_in_fus_fp   , 'rgba(69, 179, 196, 0.8)']
    dict_fp_vid_overlap_fus_fp        = [FP_V_I[1], FP_Fus[1], fp_vid_overlap_fus_fp       , 'rgba(184, 147, 147, 0.8)']
    dict_fp_vid_overlap_not_in_fus_fp = [FP_V_I[1], FPnFus[1], fp_vid_overlap_not_in_fus_fp, 'rgba(184, 147, 147, 0.8)']

    if add_labels:
        labels = [FP_V_O[0],FP_I_O[0],FP_V_I[0],FP_Fus[0],FPnFus[0]]
    else:
        labels = ['','','','','','']
    links = dict(
          source = [dict_fp_vid_only_fus_fp[0],dict_fp_vid_only_not_in_fus_fp[0],dict_fp_imu_only_fus_fp[0],dict_fp_imu_only_not_in_fus_fp[0],dict_fp_vid_overlap_fus_fp[0],dict_fp_vid_overlap_not_in_fus_fp[0]],
          target = [dict_fp_vid_only_fus_fp[1],dict_fp_vid_only_not_in_fus_fp[1],dict_fp_imu_only_fus_fp[1],dict_fp_imu_only_not_in_fus_fp[1],dict_fp_vid_overlap_fus_fp[1],dict_fp_vid_overlap_not_in_fus_fp[1]],
          value  = [dict_fp_vid_only_fus_fp[2],dict_fp_vid_only_not_in_fus_fp[2],dict_fp_imu_only_fus_fp[2],dict_fp_imu_only_not_in_fus_fp[2],dict_fp_vid_overlap_fus_fp[2],dict_fp_vid_overlap_not_in_fus_fp[2]],
          color  = [dict_fp_vid_only_fus_fp[3],dict_fp_vid_only_not_in_fus_fp[3],dict_fp_imu_only_fus_fp[3],dict_fp_imu_only_not_in_fus_fp[3],dict_fp_vid_overlap_fus_fp[3],dict_fp_vid_overlap_not_in_fus_fp[3]]
      )
    plot(title, labels, links)

def plot_fp2(title, fp_vid_only_fus_fp, fp_vid_only_not_in_fus_fp, fp_imu_only_fus_fp, fp_imu_only_not_in_fus_fp, fp_vid_overlap_fus_fp, fp_vid_overlap_not_in_fus_fp, add_labels = True):

    FP_V_O = ['FP2 (V-I)',0]
    FP_I_O = ['FP2 (I-V)',1]
    FP_V_I = ['FP2 (I∩V)',2]
    FP_Fus = ['FP2 (Fusion)',3]
    FPnFus = ['FP2 (Not in Fusion)',4]

    dict_fp_vid_only_fus_fp           = [FP_V_O[1], FP_Fus[1], fp_vid_only_fus_fp          , 'rgba(237, 80, 49, 0.8)']
    dict_fp_vid_only_not_in_fus_fp    = [FP_V_O[1], FPnFus[1], fp_vid_only_not_in_fus_fp   , 'rgba(237, 80, 49, 0.8)']
    dict_fp_imu_only_fus_fp           = [FP_I_O[1], FP_Fus[1], fp_imu_only_fus_fp          , 'rgba(69, 77, 196, 0.8)']
    dict_fp_imu_only_not_in_fus_fp    = [FP_I_O[1], FPnFus[1], fp_imu_only_not_in_fus_fp   , 'rgba(69, 77, 196, 0.8)']
    dict_fp_vid_overlap_fus_fp        = [FP_V_I[1], FP_Fus[1], fp_vid_overlap_fus_fp       , 'rgba(166, 170, 146, 0.8)']
    dict_fp_vid_overlap_not_in_fus_fp = [FP_V_I[1], FPnFus[1], fp_vid_overlap_not_in_fus_fp, 'rgba(166, 170, 146, 0.8)']

    if add_labels:
        labels = [FP_V_O[0],FP_I_O[0],FP_V_I[0],FP_Fus[0],FPnFus[0]]
    else:
        labels = ['','','','','','']
    links = dict(
          source = [dict_fp_vid_only_fus_fp[0],dict_fp_vid_only_not_in_fus_fp[0],dict_fp_imu_only_fus_fp[0],dict_fp_imu_only_not_in_fus_fp[0],dict_fp_vid_overlap_fus_fp[0],dict_fp_vid_overlap_not_in_fus_fp[0]],
          target = [dict_fp_vid_only_fus_fp[1],dict_fp_vid_only_not_in_fus_fp[1],dict_fp_imu_only_fus_fp[1],dict_fp_imu_only_not_in_fus_fp[1],dict_fp_vid_overlap_fus_fp[1],dict_fp_vid_overlap_not_in_fus_fp[1]],
          value  = [dict_fp_vid_only_fus_fp[2],dict_fp_vid_only_not_in_fus_fp[2],dict_fp_imu_only_fus_fp[2],dict_fp_imu_only_not_in_fus_fp[2],dict_fp_vid_overlap_fus_fp[2],dict_fp_vid_overlap_not_in_fus_fp[2]],
          color  = [dict_fp_vid_only_fus_fp[3],dict_fp_vid_only_not_in_fus_fp[3],dict_fp_imu_only_fus_fp[3],dict_fp_imu_only_not_in_fus_fp[3],dict_fp_vid_overlap_fus_fp[3],dict_fp_vid_overlap_not_in_fus_fp[3]]
      )
    plot(title, labels, links)

def plot_tp_fn_DIS(add_labels):
    title = 'OREBA-DIS'
    tp_vid_only_fus_tp    = 136
    tp_vid_only_fus_fn    = 2
    tp_imu_only_fus_tp    = 42
    tp_imu_only_fus_fn    = 64
    tp_vid_overlap_fus_tp = 593
    tp_vid_overlap_fus_fn = 0
    fn_overlap_vid_fus_tp = 1
    fn_overlap_vid_fus_fn = 71
    plot_tp_fn(title, tp_vid_only_fus_tp, tp_vid_only_fus_fn, tp_imu_only_fus_tp, tp_imu_only_fus_fn, tp_vid_overlap_fus_tp, tp_vid_overlap_fus_fn, fn_overlap_vid_fus_tp, fn_overlap_vid_fus_fn, add_labels)

def plot_tp_fn_SHA(add_labels):
    title = 'OREBA-SHA'
    tp_vid_only_fus_tp    = 21
    tp_vid_only_fus_fn    = 7
    tp_imu_only_fus_tp    = 82
    tp_imu_only_fus_fn    = 29
    tp_vid_overlap_fus_tp = 639
    tp_vid_overlap_fus_fn = 9
    fn_overlap_vid_fus_tp = 0
    fn_overlap_vid_fus_fn = 16
    plot_tp_fn(title, tp_vid_only_fus_tp, tp_vid_only_fus_fn, tp_imu_only_fus_tp, tp_imu_only_fus_fn, tp_vid_overlap_fus_tp, tp_vid_overlap_fus_fn, fn_overlap_vid_fus_tp, fn_overlap_vid_fus_fn, add_labels)

def plot_fp1_DIS(add_labels):
    title = 'OREBA-DIS (FP1)'
    fp_vid_only_fus_fp           = 14
    fp_vid_only_not_in_fus_fp    = 0
    fp_imu_only_fus_fp           = 0
    fp_imu_only_not_in_fus_fp    = 11
    fp_vid_overlap_fus_fp        = 6
    fp_vid_overlap_not_in_fus_fp = 0
    plot_fp1(title, fp_vid_only_fus_fp, fp_vid_only_not_in_fus_fp, fp_imu_only_fus_fp, fp_imu_only_not_in_fus_fp, fp_vid_overlap_fus_fp, fp_vid_overlap_not_in_fus_fp, add_labels)

def plot_fp2_DIS(add_labels):
    title = 'OREBA-DIS (FP2)'
    fp_vid_only_fus_fp           = 39
    fp_vid_only_not_in_fus_fp    = 0
    fp_imu_only_fus_fp           = 23
    fp_imu_only_not_in_fus_fp    = 77
    fp_vid_overlap_fus_fp        = 10
    fp_vid_overlap_not_in_fus_fp = 0
    plot_fp2(title, fp_vid_only_fus_fp, fp_vid_only_not_in_fus_fp, fp_imu_only_fus_fp, fp_imu_only_not_in_fus_fp, fp_vid_overlap_fus_fp, fp_vid_overlap_not_in_fus_fp, add_labels)

def plot_fp1_SHA(add_labels):
    title = 'OREBA-SHA (FP1)'
    fp_vid_only_fus_fp           = 5
    fp_vid_only_not_in_fus_fp    = 1
    fp_imu_only_fus_fp           = 8
    fp_imu_only_not_in_fus_fp    = 12
    fp_vid_overlap_fus_fp        = 9
    fp_vid_overlap_not_in_fus_fp = 2
    plot_fp1(title, fp_vid_only_fus_fp, fp_vid_only_not_in_fus_fp, fp_imu_only_fus_fp, fp_imu_only_not_in_fus_fp, fp_vid_overlap_fus_fp, fp_vid_overlap_not_in_fus_fp, add_labels)

def plot_fp2_SHA(add_labels):
    title = 'OREBA-SHA (FP2)'
    fp_vid_only_fus_fp           = 84
    fp_vid_only_not_in_fus_fp    = 55
    fp_imu_only_fus_fp           = 19
    fp_imu_only_not_in_fus_fp    = 55
    fp_vid_overlap_fus_fp        = 27
    fp_vid_overlap_not_in_fus_fp = 3
    plot_fp2(title, fp_vid_only_fus_fp, fp_vid_only_not_in_fus_fp, fp_imu_only_fus_fp, fp_imu_only_not_in_fus_fp, fp_vid_overlap_fus_fp, fp_vid_overlap_not_in_fus_fp, add_labels)

def main(args=None):
    plot_tp_fn_DIS(False)
    plot_tp_fn_SHA(False)
    plot_fp1_DIS(False)
    plot_fp2_DIS(False)
    plot_fp1_SHA(False)
    plot_fp2_SHA(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add other labels to prob files')
    args = parser.parse_args()
    main(args)
