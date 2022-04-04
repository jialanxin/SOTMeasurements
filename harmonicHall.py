import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pathlib
import re
import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import mu_0
import json

def recognize_filename(data_filename):
    patterns = re.match( r'\d+_I (.*)A_H-(.*)Oe', data_filename)
    if patterns:
        I = patterns.group(1)
        H = patterns.group(2)
        label = {"I": float(I), "H": float(H)}
        return label
    else:
        raise ValueError("The filename is not in the correct format.")
def get_data(data_file_path ,label):
    data_content = pd.read_csv(data_file_path, sep='\t')
    data_content = data_content.iloc[:,:10]
    data_content = data_content.rename(columns={"Field":"Angle"})
    data_content["Angle"] = data_content["Angle"] /10000
    data_content["2nd_R_H"] = data_content["2nd Y"]/label["I"]*np.sqrt(2)
    return data_content

def first_order_sine(x, a, b, phi_0):
    x_real = x/180*np.pi
    phi_0_real = phi_0/180*np.pi
    return a*np.sin(2*(x_real-phi_0_real)) + b

def first_order_fit(data_content, fig):
    popt,pocv = curve_fit(first_order_sine, data_content["Angle"], data_content["1st R_H"])
    amplitude, bias, angular_correction = popt[0], popt[1], popt[2]
    data_content["Correct_Angle"] = data_content["Angle"]-angular_correction
    data_content["Unbiased_1st_R_H"] = data_content["1st R_H"]-bias
    fitted_curve = amplitude*np.sin(2*data_content["Correct_Angle"]/180*np.pi)
    fig.add_trace(go.Scatter(x=data_content["Correct_Angle"], y=data_content["Unbiased_1st_R_H"], mode="markers"),row=1, col=1)
    fig.add_trace(go.Scatter(x=data_content["Correct_Angle"], y=fitted_curve, mode="lines"), row=1, col=1)
    R_PHE = np.abs(amplitude)
    return data_content, fig, R_PHE

def second_order_cos_phi(x,amplitude,offset):
    x_real = x/180*np.pi
    return amplitude*np.cos(x_real)+offset
def second_order_FL(x,amplitude,offset):
    x_real = x/180*np.pi
    return amplitude*np.cos(x_real)*np.cos(2*x_real)+offset
def second_order_complex(x,amp_phi,amp_FL,offset):
    x_real = x/180*np.pi
    return amp_phi*np.cos(x_real)+amp_FL*np.cos(2*x_real)*np.cos(x_real)+offset
# def second_order_fit_onestep(data_content, fig):
#     popt,pocv = curve_fit(second_order_complex, data_content["Correct_Angle"], data_content["2nd_R_H"])
#     amp_phi, amp_FL, offset = popt[0], popt[1], popt[2]
#     data_content["2nd_R_cos_phi"] = second_order_cos_phi(data_content["Correct_Angle"],amplitude_cos_phi,offset)
#     correct_angles_no_FL = []
#     second_Y_no_FL = []
#     for angle_no_FL in [45, 135, 225, 315]:
#         df_sort = data_content.iloc[(data_content["Correct_Angle"]-angle_no_FL).abs().argsort()[:1]]
#         correct_angles_no_FL.append(df_sort["Correct_Angle"].values[0])
#         second_Y_no_FL.append(df_sort["2nd_R_H"].values[0])
#     fig.add_trace(go.Scatter(x=correct_angles_no_FL, y=second_Y_no_FL, mode="markers"), row=2, col=1)
#     fig.add_trace(go.Scatter(x=data_content["Correct_Angle"], y=data_content["2nd_R_cos_phi"], mode="lines"), row=2, col=1)

#     return data_content, fig
def second_order_fit(data_content, fig, label):
    correct_angles_no_FL = []
    second_Y_no_FL = []
    for angle_no_FL in [45, 135, 225, 315]:
        df_sort = data_content.iloc[(data_content["Correct_Angle"]-angle_no_FL).abs().argsort()[:1]]
        correct_angles_no_FL.append(df_sort["Correct_Angle"].values[0])
        second_Y_no_FL.append(df_sort["2nd_R_H"].values[0])
    fig.add_trace(go.Scatter(x=correct_angles_no_FL, y=second_Y_no_FL, mode="markers"), row=2, col=1)
    popt,pocv = curve_fit(second_order_cos_phi, correct_angles_no_FL, second_Y_no_FL)
    amplitude_cos_phi, offset_cos_phi = popt[0], popt[1]
    label["amp_cos_phi"] = amplitude_cos_phi
    data_content["2nd_R_cos_phi"] = second_order_cos_phi(data_content["Correct_Angle"],amplitude_cos_phi,offset_cos_phi)
    fig.add_trace(go.Scatter(x=data_content["Correct_Angle"], y=data_content["2nd_R_cos_phi"], mode="lines"), row=2, col=1)

    data_content["2nd_R_FL"] = data_content["2nd_R_H"]-data_content["2nd_R_cos_phi"]
    fig.add_trace(go.Scatter(x=data_content["Correct_Angle"], y=data_content["2nd_R_FL"], mode="markers"), row=3, col=1)
    popt,pocv = curve_fit(second_order_FL, data_content["Correct_Angle"], data_content["2nd_R_FL"])
    amplitude_FL, offset_FL = popt[0], popt[1]
    fig.add_trace(go.Scatter(x=data_content["Correct_Angle"], y=second_order_FL(data_content["Correct_Angle"],amplitude_FL,offset_FL), mode="lines"), row=3, col=1)
    B_FL_Oe = amplitude_FL/2/label["R_PHE"]*label["H"]/1e4
    label["B_FL_Oe"] = B_FL_Oe

    fig.add_trace(go.Scatter(x=data_content["Correct_Angle"], y=data_content["2nd_R_H"], mode="markers"),row=4, col=1)

    return data_content, fig, label

def fit_single_file(data_file_path, label):
    fig = make_subplots(rows=4, cols=1,x_title=r'$\text{Planar angle }\varphi \text{/deg}$', subplot_titles=("1st order", r'$\text{2nd order cos}\varphi$', "2nd order FL", "2nd order All"))
    data_content = get_data(data_file_path, label)
    data_content, fig, R_PHE = first_order_fit(data_content, fig)
    label["R_PHE"] = R_PHE
    data_content, fig, label = second_order_fit(data_content,fig, label)
    fig.update_layout(title=f"{label['I']}_I_{label['H']}_H_{label['B_FL_Oe']}_B_FL_Oe",showlegend=False)
    fig.update_yaxes(title_text=r'$R^\omega_{\text{xy}}/\Omega$', row=1, col=1)
    fig.update_yaxes(title_text=r'$R^{2\omega}_{\mathrm{cos}\varphi}/\Omega$', row=2, col=1)
    fig.update_yaxes(title_text=r'$R^{2\omega}_{\text{FL}}/\Omega$', row=3, col=1)
    fig.update_yaxes(title_text=r'$R^{2\omega}_{\text{xy}}/\Omega$', row=4, col=1)
    return fig


if __name__ == "__main__":
    data_folder = pathlib.Path("sample2")
    data_file_paths = list(data_folder.glob('*.lvm'))
    preprocess_folder = data_folder/"preprocess"
    preprocess_folder.mkdir(exist_ok=True)
    for data_file_path in data_file_paths:
        try:
            label = recognize_filename(data_file_path.stem)
        except ValueError:
            continue
        fig = make_subplots(rows=4, cols=1,x_title=r'$\text{Planar angle }\varphi \text{/deg}$', subplot_titles=("1st order", r'$\text{2nd order cos}\varphi$', "2nd order FL", "2nd order All"))
        data_content = get_data(data_file_path, label)
        data_content, fig, R_PHE = first_order_fit(data_content, fig)
        label["R_PHE"] = R_PHE
        data_content, fig, label = second_order_fit(data_content,fig, label)
        with open(preprocess_folder/f"{label['I']}_I_{label['H']}_H.json", "w") as f:
            json.dump(label, f)
        fig.update_layout(title=f"{label['I']}_I_{label['H']}_H_{label['B_FL_Oe']}_B_FL_Oe",showlegend=False)
        fig.update_yaxes(title_text=r'$R^\omega_{\text{xy}}/\Omega$', row=1, col=1)
        fig.update_yaxes(title_text=r'$R^{2\omega}_{\mathrm{cos}\varphi}/\Omega$', row=2, col=1)
        fig.update_yaxes(title_text=r'$R^{2\omega}_{\text{FL}}/\Omega$', row=3, col=1)
        fig.update_yaxes(title_text=r'$R^{2\omega}_{\text{xy}}/\Omega$', row=4, col=1)
        fig.write_image(preprocess_folder/f"{label['I']}_I_{label['H']}_H.png")


