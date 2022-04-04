import streamlit as st
from stfmr import fitting,draw
from harmonicHall import fit_single_file, recognize_filename
with st.sidebar:
    choice = st.radio('Choose a function',('STFMR','Harmonic Hall'))
if choice == 'STFMR':
    st.header('STFMR')
    data = st.file_uploader("Upload a STFMR data file", type="txt")
    if data is not None:
        result = fitting(data)
        fig = draw(result,"Empty")
        st.write(fig)
elif choice == 'Harmonic Hall':
    st.header('Harmonic Hall')
    data = st.file_uploader("Upload a harmonic Hall data file", type="lvm")
    if data is not None:
        filename = st.text_input("Copy the filename of lvm file")
        label = recognize_filename(filename)
        fig = fit_single_file(data,label)
        st.write(fig)

