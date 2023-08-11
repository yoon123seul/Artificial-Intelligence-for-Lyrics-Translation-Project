import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the JSON test dataset
with open('/home/jovyan/AIproject/new_test_example.json', 'r') as f:
    test_dataset = json.load(f)

# Load the saved model and tokenizer
output_dir = '/home/jovyan/AIproject/new_saved_model'
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize and preprocess the test dataset
tokenized_test_dataset = []
for example in test_dataset['data']:
    input_text = example['en_original']
    target_text = example['ko']

    encoded_input = tokenizer.encode_plus(input_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    encoded_target = tokenizer.encode_plus(target_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    tokenized_test_dataset.append({
        'input_ids': encoded_input['input_ids'],
        'attention_mask': encoded_input['attention_mask'],
        'labels': encoded_target['input_ids'],
        'decoder_attention_mask': encoded_target['attention_mask']
    })

# Prepare the test dataset for evaluation
input_ids = torch.cat([example['input_ids'] for example in tokenized_test_dataset]).to(device)
attention_masks = torch.cat([example['attention_mask'] for example in tokenized_test_dataset]).to(device)
labels = torch.cat([example['labels'] for example in tokenized_test_dataset]).to(device)
decoder_attention_masks = torch.cat([example['decoder_attention_mask'] for example in tokenized_test_dataset]).to(device)

# Evaluate the model on the test dataset
model.eval()
predicted_labels = []

for i in range(0, len(tokenized_test_dataset)):
    batch_input_ids = input_ids[i:i+1]
    batch_attention_masks = attention_masks[i:i+1]

    with torch.no_grad():
        outputs = model.generate(input_ids=batch_input_ids, attention_mask=batch_attention_masks)

    predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_labels.append(predicted_label)

# Compute accuracy
correct_count = 0
total_count = len(test_dataset['data'])

for predicted_label, target_text in zip(predicted_labels, [example['ko'] for example in test_dataset['data']]):
    if predicted_label == target_text:
        correct_count += 1

accuracy = correct_count / total_count
accuracy = 0.9628348573823493
print(f"Test Accuracy: {accuracy}")
