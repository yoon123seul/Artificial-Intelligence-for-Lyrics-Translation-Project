from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json

# Load the JSON dataset for fine-tuning
with open('/home/jovyan/AIproject/train_exapmle.json', 'r') as f:
    dataset = json.load(f)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-en2ko")
model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-en2ko")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Tokenize and preprocess the dataset
tokenized_dataset = []
for example in dataset['data']:
    input_text = example['en_original']
    target_text = example['ko']

    encoded_input = tokenizer.encode_plus(input_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    encoded_target = tokenizer.encode_plus(target_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    tokenized_dataset.append({
        'input_ids': encoded_input['input_ids'],
        'attention_mask': encoded_input['attention_mask'],
        'labels': encoded_target['input_ids'],
        'decoder_attention_mask': encoded_target['attention_mask']
    })

# Prepare the dataset for fine-tuning
input_ids = torch.cat([example['input_ids'] for example in tokenized_dataset]).to(device)
attention_masks = torch.cat([example['attention_mask'] for example in tokenized_dataset]).to(device)
labels = torch.cat([example['labels'] for example in tokenized_dataset]).to(device)
decoder_attention_masks = torch.cat([example['decoder_attention_mask'] for example in tokenized_dataset]).to(device)

# Fine-tune the model
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) #1e-5
num_epochs = 10
batch_size = 8

for epoch in range(num_epochs):
    total_loss = 0

    for i in range(0, len(tokenized_dataset), batch_size):
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_masks = attention_masks[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        batch_decoder_attention_masks = decoder_attention_masks[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks,
                        labels=batch_labels, decoder_attention_mask=batch_decoder_attention_masks)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(tokenized_dataset)}")

# Save the fine-tuned model
output_dir = '/home/jovyan/AIproject/new_saved_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
#print(f"Fine-tuned model and tokenizer saved in {output_dir}")