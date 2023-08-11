from Rulebase import *
from G2P import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
# 로드할 모델과 토크나이저의 경로
model_dir = '/home/jovyan/AIproject/new_saved_model'
tokenizer_dir = '/home/jovyan/AIproject/new_saved_model'

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model =  AutoModelForSeq2SeqLM.from_pretrained(model_dir)
# 테스트할 입력 텍스트


input_text = input('영어 가사를 입력하세요: ')

# 영어 가사의 음절 수 추출
p_number = countsyl(input_text)
print ('음절 수:', p_number)

# 입력 텍스트를 토큰화
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 모델을 사용하여 번역 생성
output = model.generate(input_ids, max_length=30, repetition_penalty=2.0)


# 생성된 번역을 디코딩
translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print ('단순 번역 결과 :', translated_text)


testtxt= translated_text
result=(make(testtxt,p_number))
for i in result:
    print(i)