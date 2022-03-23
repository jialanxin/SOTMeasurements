import streamlit as st
from preprocess import fitting,draw
st.header('Quick Preview')
data = st.file_uploader("Upload a STFMR data file", type="txt")
if data is not None:
    result = fitting(data)
    fig = draw(result,"Empty")
    st.write(fig)

