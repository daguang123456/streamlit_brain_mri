import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps

st.title("使用谷歌的可教机器进行图像分类")
st.header("脑肿瘤MRI分类示例")
st.text("上传彩色脑部MRI的jpg图像，将图像分类为肿瘤或无肿瘤")

uploaded_file = st.file_uploader("选择脑部磁共振成像...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='上传了核磁共振成像。', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'keras_model.h5')
    if label == 0:
        st.write("MRI扫描有脑肿瘤")
    else:
        st.write("磁共振扫描健康")
