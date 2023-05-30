from model import *
from transformers import AutoTokenizer
from transformers import BertTokenizer
import streamlit as st
import torch
import numpy as np

"""
여기서부터는 웹에 들어갈 내용
관련된 함수 참고 : https://docs.streamlit.io/
"""

st.title("자동 채점 모델 기반 자동 피드백")
st.write("**팀원** : 수학교육과 김명식, 김재훈, 김지영, 신인섭, 윤예린, 정유진")

st.subheader("문항1-1")
st.markdown("$a^{2} times a^{5} $ 를 구하시오")


st.subheader("문항1-2")
st.markdown("$\left (x^{4} \right )^{3} \times \left(x^{2} \right )^{5} $ 를 구하시오")

st.subheader("문항1-3")
st.markdown("$b^{3} \div b^{6} $ 을 구하시오")


st.subheader("문항1-4")
st.markdown("$a^{12} \div a^{3} \div a^{9} $ 를 구하시오")


st.subheader("문항1-5")
st.markdown("$\left ( 2 a^{4} \right )^{3} $ 을 구하시오")


st.subheader("문항1-6")
st.markdown("$\left (b/3 \right )^{4}$  를 구하시오")


st.subheader("문항1-7")
st.markdown("$\left ( 2^4 \right )^x \times \left ( 2^2 \right )^x = 2^3 \times 2^{ 3x }$ 일 때 자연수 $x$의 값을 구하시오")


st.subheader("문항1-8")
st.markdown("저장 매체의 용량을 나타내는 단위로 $B, KB, MB$등이 있고, $1 KB=2^{10}  B $ , $1 MB=2^{10}  KB $ 이다. 찬혁이가 컴퓨터로 용량이 $36MB$인 자료를 내려받으려고 한다. 이 컴퓨터에서 1초당 내려받는 자료의 용량이 $9 \times 2^{20} B$ 일 때 , 찬혁이가 자료를 모두 내려받는 데 몇 초가 걸리는지 구하시오")


st.subheader("문항2-1")
st.markdown("$2x \times 3xy $  를 구하시오")


st.subheader("문항2-2")
st.markdown("$x^{2}y \times 3x^{3}y^{2} $ 를 구하시오")


st.subheader("문항2-3")
st.markdown("6a^{2}b \div 2ab $ 를 구하시오")


st.subheader("문항2-4")
st.markdown("$( 4x^{2} - 2x + 1 ) + ( x^{2} + x - 4 )  $ 를 구하시오")


st.subheader("문항2-5")
st.markdown("$2 ( 3y^{2} - y + 2 ) - ( y^{2} - 2y + 3 ) 를 구하시오")


st.subheader("문항2-6")
st.markdown("( - 12x^{3}y^{2} ) \div \square \times 18x^{3}y^{3} = 8x^{2}y^{3}$ 일 때 \square 안에 알맞은 식을 구하시오")

            
            
st.subheader("문항2-7")
st.markdown("$높이가 ( 2x^{2} )^{2} 인 삼각형의 넓이가 48x^{3}y^{2}일 때 이 삼각형의 밑변의 길이를 구하시오")
            
            
st.subheader("문항2-8")
st.markdown("$A = 3x -2y, B = x + 3y$일 때, $3A -2( A - B )$를 $x, y$에 대한 식으로 나타내시오")
            

            
st.subheader("문항2-9")
st.markdown("$ 6/5 \left (-10x+15y \right )- 1/2 \left(4x+6y \right )=ax+by$ 라 할 떄, 상수 $a+b$의 값을 구하시오")

            
st.subheader("문항3-1")
st.markdown("$2a \left (a + b \ right ) $ 를 구하시오")

            
st.subheader("문항3-2")
st.markdown("$( 8a^{2} + 4ab ) \div 2a $ 를 구하시오")

            
st.subheader("문항3-3")
st.markdown("$A \div 3y/2 = 4x^{2}y + 2xy +6$ 일 때 다항식 $A$ 를 구하시오")

            
response = st.text_input('답안 :', "답안을 작성해주세요")

"""
자신의 모델에 맞는 변수 설정해주기
"""

model_name = "2-6_rnn_sp_92" #모델 이름 넣어주기 확장자는 넣지말기!
#모델에 맞는 hyperparameter 설정
vs = 92 #vocab size
emb = 16 #default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32 #default 값 지정 안했으면 건드리지 않아도 됨
nh = 4 #default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu" #default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
#output_d 설정
output_d = 5 #자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)


"""
model과 tokneizer 부르기   주석처리는 #으로
"""
model = RNNModel(output_d, c) #RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
# model = ATTModel(output_d, c) #ATTModel 쓰는경우

model.load_state_dict(torch.load("./save/"+model_name+".pt"))

#자신에게 맞는 모델로 부르기
tokenizer = AutoTokenizer.from_pretrained("./save/"+ model_name) #sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

"""
자동 채점해주는 코드
"""
enc = tokenizer(response)["input_ids"] #sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len :
    pad = (max_len - l) * [0] + enc
else : pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1,max_len)
y = model(pad_ten)
label = y.squeeze().detach().cpu().numpy().round()

if st.button('피드백 받기'):
    """
    output차원에 맞추어 피드백 넣기
    """
    st.write(response)
    if label[1] == 1:
        st.success('(다항식) 곱하기 (단항식)을 잘하는구나!', icon="✅")
    else :
        st.info('(다항식) 곱하기 (단항식)을 잘 생각해보자!', icon="ℹ️")
else : 
    st.button('피드백 받기 버튼을 눌러보세요!')
