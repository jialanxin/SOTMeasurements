import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import numpy as np
import pathlib
import re
import json

def recongize_data_label(data_file:pathlib.Path):
    template=re.compile("\d{6}_(.*?)G_(.*?)_(\d+)r")
    filename = data_file.stem
    match_result = template.match(filename)
    if match_result == None:
        raise RuntimeError("Fail to recongize_data_lable")
    microwave_frequency = match_result.group(1)
    sample_name = match_result.group(2)
    rotate_angle = match_result.group(3)
    data_label = {"mwfreq":float(microwave_frequency),"name":sample_name,"ang":float(rotate_angle)}
    data_label_string = f"{data_label['name']}_{data_label['ang']}deg_{data_label['mwfreq']}GHz"
    return data_label, data_label_string

def STFMR(H,A,S,H_res,Delta_H,Baseline):
    asymmetric = Asymmetric(H,A,S,H_res,Delta_H,Baseline)
    symmetric = Symmetric(H,A,S,H_res,Delta_H,Baseline)
    return asymmetric+symmetric+Baseline
def Asymmetric(H,A,S,H_res,Delta_H,Baseline):
    return A*Delta_H*(H-H_res)/(Delta_H**2+(H-H_res)**2)
def Symmetric(H,A,S,H_res,Delta_H,Baseline):
    return S*Delta_H**2/(Delta_H**2+(H-H_res)**2)

def fitting(data_file:pathlib.Path):
    data = pd.read_csv(data_file,sep="\t")
    data = data[data["Field/Oe"]<=1600]
    H = data["Field/Oe"]
    V = data["X/V"]
    V_sigma = np.ones_like(V)*0.3e-6
    popt,_ = curve_fit(STFMR,H,V,sigma=V_sigma,bounds=([-np.inf,-np.inf,0,0,-np.inf],[np.inf,np.inf,2000,1000,np.inf]))
    V_fit = STFMR(H,*popt)
    V_sym = Symmetric(H,*popt)+popt[-1]
    V_asy = Asymmetric(H,*popt)+popt[-1]
    fitting_result = {"A":popt[0],"S":popt[1],"H_res":popt[2],"Delta_H":popt[3],"Baseline":popt[4],"H":H.tolist(),"V":V.tolist(),"V_fit":V_fit.tolist(),"V_sym":V_sym.tolist(),"V_asy":V_asy.tolist()}
    return fitting_result
def draw(fitting_result,data_label_string:str):
    print(data_label_string)
    H = fitting_result["H"]
    V = fitting_result["V"]
    V_fit = fitting_result["V_fit"]
    V_sym = fitting_result["V_sym"]
    V_asy = fitting_result["V_asy"]
    fig = go.Figure()
    fig.update_layout(title=data_label_string,xaxis_title="H/Oe",yaxis_title="V_mix/V")
    fig.add_trace(go.Scatter(x=H,y=V,name="V"))
    fig.add_trace(go.Scatter(x=H,y=V_fit,name="V_fit"))
    fig.add_trace(go.Scatter(x=H,y=V_sym,name="V_sym"))
    fig.add_trace(go.Scatter(x=H,y=V_asy,name="V_asy"))
    return fig
def save_results(data_label,data_label_string,fitting_result,fig):
    img_name = data_label_string+".png"
    img_path = sub_path/img_name
    fig.write_image(img_path)
    json_name = data_label_string+".json"
    json_path = sub_path/json_name
    with open(json_path,"w") as f:
        json.dump({**data_label,**fitting_result},f)

if __name__=="__main__":
    folder_path = pathlib.Path("./sample7_take2")
    data_files = list(folder_path.glob("*.txt"))
    sub_path = folder_path/"preprocess"
    sub_path.mkdir(exist_ok=True)


    for data_file in data_files:
        data_label, data_label_string = recongize_data_label(data_file)
        # if data_label["ang"]==135:
        fitting_result = fitting(data_file)
        fig = draw(fitting_result,data_label_string)
        save_results(data_label,data_label_string,fitting_result,fig)

    









