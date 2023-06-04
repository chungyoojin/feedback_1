from model import *
from transformers import AutoTokenizer
from transformers import BertTokenizer
import streamlit as st
import torch
import numpy as np

st.title("자동 채점 모델 기반 자동 피드백")
st.write("**팀원** : 수학교육과 김명식, 김재훈, 김지영, 신인섭, 윤예린, 정유진")



st.subheader("문항2-7")
st.markdown("높이가 $( 2x )^{2} $인 삼각형의 넓이가 $48x^{3}y^{2}$일 때 이 삼각형의 밑변의 길이를 구하시오")
response = st.text_input('답안 :', "답안을 작성해주세요", key='answer_input_2_7')


model_name_2_7 = "2-7_kc_rnn_sp_92" #모델 이름 넣어주기 확장자는 넣지말기!
#모델에 맞는 hyperparameter 설정
vs = 92 #vocab size
emb = 16 #default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32 #default 값 지정 안했으면 건드리지 않아도 됨
nh = 4 #default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu" #default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
#output_d 설정
output_d_2_7 = 7 #자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)



model_2_7 = RNNModel(output_d_2_7, c) #RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
# model = ATTModel(output_d, c) #ATTModel 쓰는경우

model_2_7.load_state_dict(torch.load("./save/"+model_name_2_7+".pt"))

#자신에게 맞는 모델로 부르기
tokenizer_2_7 = AutoTokenizer.from_pretrained("./save/"+ model_name_2_7) #sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

enc = tokenizer(response)["input_ids"] #sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len :
    pad = (max_len - l) * [0] + enc
else : pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1,max_len)
y = model_2_7(pad_ten)
label_2_7 = y.squeeze().detach().cpu().numpy().round()

if st.button('피드백 받기', key='button2_7_1'):
    
    #output차원에 맞추어 피드백 넣기
    
    st.write(response)
    if len(label_2_7)>= 5 :
         if label_2_7[0] == 1 and label_2_7[1] == 1 and label_2_7[2] == 1 and label_2_7[3] == 1 and label_2_7[4] == 1 and label_2_7[6] == 1 :
            st.success('구하고자 하는 것을 미지수로 설정하여 곱의 거듭제곱과 거듭제곱의 나눗셈, 다항식의 나눗셈을 이용하여 삼각형의 넓이를 잘 구했구나!', icon="✅")
         elif label_2_7[0] == 1 and label_2_7[1] == 1 and label_2_7[2] == 1 and label_2_7[3] == 1 and label_2_7[4] == 0 :
            st.success('곱의 거듭제곱과 거듭제곱의 나눗셈, 다항식의 나눗셈을 이용하여 삼각형의 넓이를 거의 구했구나! 중간 과정에 실수가 없는지 확인해보자!', icon="ℹ️")
         elif label_2_7[0] == 1 and label_2_7[1] == 1 and label_2_7[2] == 1 and label_2_7[3] == 0 and label_2_7[4] == 0 :
            st.success('곱의 거듭제곱을 잘 이용했구나! 삼각형의 넓이를 구하는 방법과 다항식의 나눗셈 과정을 다시 한 번 확인해보자!', icon="ℹ️")
         elif label_2_7[0] == 1 and label_2_7[1] == 1 and label_2_7[2] == 0 and label_2_7[3] == 0 and label_2_7[4] == 0 :
            st.success('곱의 거듭제곱을 잘 이용했구나! 다항식의 나눗셈 과정을 다시 한 번 확인해보자!', icon="ℹ️")
         elif label_2_7[0] == 1 and label_2_7[1] == 0 and label_2_7[2] == 0 and label_2_7[3] == 0 and label_2_7[4] == 0 :
            st.success('곱의 거듭제곱을 잘 이용했구나! 다항식의 나눗셈 과정을 다시 한 번 확인해보자!', icon="ℹ️")
         elif label_2_7[0] == 0 and label_2_7[1] == 0 and label_2_7[2] == 0 and label_2_7[3] == 0 and label_2_7[4] == 0 and label_2_7[6] == 1:
            st.success('곱의 거듭제곱 계산과 다항식의 나눗셈 과정을 다시 한 번 확인해보자!', icon="ℹ️")
         else :   
            st.info('곱의 거듭제곱 계산과 다항식의 나눗셈 과정을 복습하자!', icon="⚠️")

    st.button('피드백 받기 버튼을 눌러보세요!')
if st.button('삼각형의 넓이는?', key='button2_7_2'):
    st.write('(삼각형의 넓이)=(밑변)x(높이)÷2')


if st.button('힌트', key='button2_7_3'):
    st.write('곱의 거듭제곱과 다항식의 나눗셈을 이용하여 식을 정리하세요!')


if st.button('모범답안', key='button2_7_4'):
    image_path = "save/2-7 모범답안.png-.png"
    st.image(image_path, caption='2-7모범답안')

    
