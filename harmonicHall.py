import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pathlib
import re
import numpy as np
from scipy.optimize import curve_fit
import json
from datetime import datetime

class Recognizer:
    def __init__(self, match_Regex):
        self.match_Regex = match_Regex
    def recognize_filename(self, data_file_path):
        data_filename = data_file_path.stem
        patterns = re.match(self.match_Regex, data_filename)
        if patterns:
            return self.recognize_detail(patterns)
        else:
            raise ValueError("File name does not match the pattern")
    def recognize_detail(self,patterns):
        pass
    
class NormalRecognizer(Recognizer):
    def __init__(self):
        super().__init__(r'\d+_I\s(.*)A_H(.*)Oe6221')
    def recognize_detail(self,patterns):
            I = patterns.group(1)
            H = patterns.group(2)
            label = {"I": float(I), "H": float(H)}  
            return label

class HITRecognizer(Recognizer):
    def __init__(self):
        super().__init__(r'(\d+)_H_(.*) Oe_Iac_(.*) A_T_(.*) K')
    def recognize_detail(self,patterns):
        timestamp = patterns.group(1)
        H = patterns.group(2)
        I = patterns.group(3)
        T = patterns.group(4)
        label = {"I": float(I), "H": round(float(H)), "T": round(float(T)),"time":timestamp}
        return label

class ZurichHITRecognizer(HITRecognizer):
    def __init__(self):
        self.match_Regex = r'(\d+)_Zurich_H_(.*) Oe_Iac_(.*) A_T_(.*) K'

class GLIHITRecognizer(Recognizer):
    def __init__(self):
        super().__init__(r'\d+gate(.*)_leak(.*)_idx(.*)_H_(.*) Oe_Iac_(.*) A_T_(.*) K')
    def recognize_detail(self,patterns):
        gate = patterns.group(1)
        L = patterns.group(2)
        idx = patterns.group(3)
        H = patterns.group(4)
        I = patterns.group(5)
        T = patterns.group(6)
        label = {"I": float(I), "H": float(H), "T": float(T), "gate": float(gate), "idx": int(idx),"L": float(L)}
        return label

class GIIHRecognizer(Recognizer):
    def __init__(self):
        super().__init__(r'\d+_I\s(.*)A_H(.*)Oe6221idx_(.*)_gate_(.*)')
    def recognize_detail(self,patterns):
        I = patterns.group(1)
        H = patterns.group(2)
        idx = patterns.group(3)
        gate = patterns.group(4)
        label = {"I": float(I), "H": float(H), "idx": int(idx), "gate": float(gate)}
        return label


def get_data(data_file_path):
    data_content = pd.read_csv(data_file_path, sep='\t')
    data_content = data_content.iloc[:,:10]
    data_content = data_content.rename(columns={"Field":"Position"})
    return data_content

def get_data_PPMS(data_file_path):
    data_content = pd.read_csv(data_file_path, sep='\t',skiprows=[0,1,2],header=None, names=["Position", "1st X", "1st Y", "1st R", "1st Theta", "2nd X", "2nd Y", "2nd R", "2nd Theta", "1st R_H", "Frequency", "Z_Position", "Z_1st X", "Z_1st Y", "Z_1st R", "Z_1st Theta", "Z_2nd X", "Z_2nd Y", "Z_2nd R", "Z_2nd Theta", "Z_1st R_H"])
    data_content["ScanIndex"] = 0
    data_content.loc[73:,"ScanIndex"] = 1
    return data_content


class FirstOrderFitter:
    def __init__(self) -> None:
        self.params = {"amp_sine_2phi":0,"offset":0,"phase_correction":0}
        self.unbiased_first_X = None
        self.fitted_curve = None
        self.corrected_angles = None
    def fit(self,angles,first_X):
        popt,pcov = curve_fit(self.fitting_function,angles,first_X)
        for (i,key) in enumerate(self.params):
            self.params[key] = popt[i]
        corrected_angles = angles - self.params["phase_correction"]
        unbiased_first_X = first_X - self.params["offset"]
        params_zero_offset = self.params.copy()
        params_zero_offset["offset"] = 0
        fitted_curve = self.fitting_function(corrected_angles,**params_zero_offset)
        raw_fig = go.Scatter(x=corrected_angles, y=unbiased_first_X, mode="markers")
        fitted_fig = go.Scatter(x=corrected_angles, y=fitted_curve, mode="lines")
        self.unbiased_first_X = unbiased_first_X
        self.fitted_curve = fitted_curve
        self.corrected_angles = corrected_angles
        return raw_fig, fitted_fig, corrected_angles
    @staticmethod
    def fitting_function(x,amp_sine_2phi, offset , phase_correction):
        x_real = x/180*np.pi
        phi_0_real = phase_correction/180*np.pi
        return amp_sine_2phi*np.sin(2*(x_real-phi_0_real)) + offset
    def show_residual(self):
        fig = go.Scatter(x=self.corrected_angles,y=self.fitted_curve-self.unbiased_first_X,mode="lines")
        return fig

class FirstOrderFitterCosinePhiSinePhi(FirstOrderFitter):
    def __init__(self):
        super().__init__()
        self.params = {"amp_sine_2phi":0,"offset":0,"phase_correction":0,"amp_cosine_phi":0,"amp_sine_phi":0}
    @staticmethod
    def fitting_function(x, amp_sine_2phi, offset, phase_correction, amp_cosine_phi,amp_sine_phi):
        x_real = x/180*np.pi
        phi_0_real = phase_correction/180*np.pi
        correct_x = x_real-phi_0_real
        return amp_sine_2phi*np.sin(2*correct_x)+ amp_cosine_phi*np.cos(correct_x) + amp_sine_phi*np.sin(correct_x) + offset

def second_order_cos_phi(x,amplitude,offset):
    x_real = x/180*np.pi
    return amplitude*np.cos(x_real)+offset
def second_order_FL(x,amplitude,offset):
    x_real = x/180*np.pi
    return amplitude*np.cos(x_real)*np.cos(2*x_real)+offset
def second_order_complex(x,amp_phi,amp_FL,offset):
    x_real = x/180*np.pi
    return amp_phi*np.cos(x_real)+amp_FL*np.cos(2*x_real)*np.cos(x_real)+offset
def second_order_complex_sine_2phi(x,amp_phi,amp_FL,amp_sine_2phi,offset):
    return second_order_complex(x,amp_phi,amp_FL,offset)+amp_sine_2phi*np.sin(2*x/180*np.pi)
def second_order_complex_sine_phi(x,amp_phi,amp_FL,amp_sine_phi,offset):
    return second_order_complex(x,amp_phi,amp_FL,offset)+amp_sine_phi*np.sin(x/180*np.pi)
def second_order_complex_sine_phi_sine_2phi(x,amp_phi,amp_FL,amp_sine_phi,amp_sine_2phi,offset):
    return second_order_complex_sine_phi(x,amp_phi,amp_FL,amp_sine_phi,offset)+amp_sine_2phi*np.sin(2*x/180*np.pi)

class SecondOrderFitter:
    def __init__(self):
        self.params = {"amp_phi":0, "amp_FL":0,"offset":0}
    def fit(self,corrected_angles,second_Y):
        popt,pcov = curve_fit(self.fitting_function,corrected_angles,second_Y)
        perr = np.sqrt(np.diag(pcov))
        err_dict = {}
        for i,key in enumerate(self.params):
            self.params[key] = popt[i]
            err_dict[f"{key}_err"] = perr[i]
        params_zero_offset  = self.params.copy()
        params_zero_offset["offset"] = 0
        second_Y_fit_curve = self.fitting_function(corrected_angles,**params_zero_offset)
        unbiased_second_Y = second_Y - self.params["offset"]
        raw_fig = go.Scatter(x=corrected_angles, y=unbiased_second_Y, mode="markers")
        fitted_fig = go.Scatter(x=corrected_angles, y=second_Y_fit_curve, mode="lines")
        self.params |= err_dict
        return raw_fig,fitted_fig,self.params
    @staticmethod
    def fitting_function(x,amp_phi,amp_FL,offset):
        return second_order_complex(x,amp_phi,amp_FL,offset)
        
class SecondOrderFitterSinePhiSine2Phi(SecondOrderFitter):
    def __init__(self):
        self.params = {"amp_phi": 0, "amp_FL": 0, "amp_sine_phi": 0,"amp_sine_2phi": 0, "offset":0}
    @staticmethod
    def fitting_function(x, amp_phi,amp_FL,amp_sine_phi,amp_sine_2phi,offset):
        return second_order_complex_sine_phi_sine_2phi(x,amp_phi,amp_FL,amp_sine_phi,amp_sine_2phi,offset)



    
def draw_HHV(data_content,label):
    fig = make_subplots(rows=2, cols=1,vertical_spacing=0.3)
    angles = data_content["Position"].iloc[1:]
    first_X = data_content["Z_1st X"].iloc[1:]
    second_Y = data_content["Z_2nd Y"].iloc[1:]
    first_order_fitter = FirstOrderFitterCosinePhiSinePhi()
    first_X_raw_fig, first_X_fit_fig,corrected_angles = first_order_fitter.fit(angles, first_X)
    fig.add_trace(first_X_fit_fig, row=1, col=1)
    fig.add_trace(first_X_raw_fig, row=1, col=1)
    second_order_fitter = SecondOrderFitterSinePhiSine2Phi()
    second_Y_raw_fig, second_Y_fit_fig,second_Y_params =  second_order_fitter.fit(corrected_angles, second_Y)
    fig.add_trace(second_Y_raw_fig, row=2, col=1)
    fig.add_trace(second_Y_fit_fig, row=2, col=1)
    font_dict=dict(family='Arial',
               size=26,
               color='black'
               )
    fig.update_layout({"title":f"H_{label['H']}Oe_I_{label['I']}A_T_{label['T']}K","template":"plotly_white","titlefont":font_dict})
    fig.update_yaxes(  # axis label
                 showline=True,  # add line at x=0
                 linecolor='black',  # line color
                 linewidth=2.4, # line size
                 ticks='outside',  # ticks outside axis
                 tickfont=font_dict, # tick label font
                 mirror='allticks',  # add ticks to top/right axes
                 tickwidth=2.4,  # tick width
                 tickcolor='black',  # tick color
                 )
    fig.update_yaxes(title_text="$\Large{V^\omega/V}$",row=1,col=1)
    fig.update_yaxes(title_text="$\Large{V^{2\omega}/V}$",row=2,col=1)
    fig.update_xaxes(
                    title_text="$\Large{Angles/deg}$",
                    showline=True,
                    showticklabels=True,
                    linecolor='black',
                    linewidth=2.4,
                    ticks='outside',
                    tickfont=font_dict,
                    mirror='allticks',
                    tickwidth=2.4,
                    tickcolor='black',
                    titlefont=font_dict
                    )
    fig.write_image(preprocess_folder/f"{data_file_path.stem}.png")
    label |=  second_Y_params
    with open(preprocess_folder/f"{data_file_path.stem}.json", "w") as f:
        json.dump(label, f)


    

if __name__ == "__main__":
    data_folder = pathlib.Path("Pt10onCrPS4-2.101")
    data_file_paths = list(data_folder.glob('*.lvm'))
    preprocess_folder = data_folder/"preprocess"
    preprocess_folder.mkdir(exist_ok=True)
    recognizer = HITRecognizer()
    for data_file_path in data_file_paths:
        try:
            label = recognizer.recognize_filename(data_file_path)
            measuretime = datetime.strptime(label["time"],"%Y%m%d%H%M%S")
            day = datetime(2022,9,11)
            if measuretime > day:
                print(label)
                data_content = get_data_PPMS(data_file_path)
                draw_HHV(data_content, label)
        except ValueError:
            continue

