import pandas as pd
import re
import tiktoken

# 
df = pd.read_csv('C:\\Users\\Nitish kumar\\OneDrive\\Desktop\\AGENTIC_AI ROADMAP\\90-AGENTIC_AI_ROADMAP\\long-doc.csv')
# 2. Python List Comprehension for Cleaning
def clean_text(text):
    if pd.isna(text): return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove non-alphanumeric characters except spaces
    text = text.strip().lower()       # Remove spaces and lowercase
    return text
df['cleaned_text'] = df['raw_text'].apply(clean_text)
df = df.dropna(subset=['cleaned_text'])

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
df['token_count'] = df['cleaned_text'].apply(lambda x: len(encoding.encode(x)))
df = df[df['cleaned_text'].str.len() > 0]

print(df[['cleaned_text', 'token_count']])
df.to_csv('cleaned_long-doc.csv', index=False)
# Applying the function using Pandas

