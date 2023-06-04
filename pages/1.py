import streamlit as st

st.set_page_config(layout="wide")
st.title("자동 채점 및 피드백 모델")
st.write("**조원** : 김명식, 김재훈, 김지영, 신인섭, 윤예린, 정유진")
st.header("설명추가")
    
st.title("문항 설계의 전반적인 원리")
st.subheader("문항 설계 목표")
st.write("1. 식의 계산 단원에서 학생들이 가지고 있는 인지 요소 분석")
st.write("2. 식의 계산 단원에서 학생들이 가지고 있는 오개념 및 오류 분석")

st.subtitle("식의 계산 단원 내 지식맵 구성")
image_path = "save/2-7 모범답안.png-.png"
st.image(image_path, caption='2-7모범답안')

