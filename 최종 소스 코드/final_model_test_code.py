import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    
    # BLEU 점수를 계산하기 위해 nltk의 bleu_score 패키지 사용
    smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
    bleu_score = nltk.translate.bleu_score.sentence_bleu(reference, candidate, smoothing_function=smoothing_function.method1)
    return bleu_score

def evaluate_translation(model, tokenizer, test_data, device):
    total_bleu_score = 0.0
    num_examples = len(test_data)

    for example in test_data:
        source_text = example['en_original']
        target_text = example['ko']

        # 소스 텍스트를 토큰화합니다.
        #encoded_input = tokenizer.encode_plus(source_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        #input_ids = encoded_input['input_ids'].to(device)
        #attention_mask = encoded_input['attention_mask'].to(device)
        
        # 모델을 통해 번역된 결과를 얻습니다.
        #outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
        #translated_ids = outputs[0].tolist()[0]
        #translated_text = tokenizer.decode(translated_ids, skip_special_tokens=False)
        input_text = source_text
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        output = model.generate(input_ids, max_length=50, repetition_penalty=2.0).to(device)
        translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # BLEU 점수 계산
        bleu_score = calculate_bleu(target_text, translated_text)
        total_bleu_score += bleu_score

    # 평균 BLEU 점수 계산
    avg_bleu_score = total_bleu_score / num_examples

    return avg_bleu_score

# Load the test JSON dataset
with open('/home/jovyan/AIproject/test_example.json', 'r') as f:
    test_dataset = json.load(f)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/home/jovyan/AIproject/new_saved_model")
model = AutoModelForSeq2SeqLM.from_pretrained("/home/jovyan/AIproject/new_saved_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate the model on the test dataset
avg_bleu_score = evaluate_translation(model, tokenizer, test_dataset['data'], device)

print("Average BLEU score:", avg_bleu_score)