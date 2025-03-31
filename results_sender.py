import pandas as pd
import yagmail

df = pd.read_csv('results_underway.csv', on_bad_lines="skip")
chunk_size = 1000
num_chunks = 24

with open("email_credentials.txt", "r") as f:
    email_password = f.read().strip()

yag = yagmail.SMTP("torbjorn.solstorm@gmail.com", email_password)
recipient = "torbjorn.smed@gmail.com"
subject = "Data from Python Script"

df = df.iloc[-(chunk_size * num_chunks):]  # Take only the last chunk_size * num_chunks rows

for i in range(num_chunks):
    chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size]
    chunk_file = f'results_chunk_{i + 1}.csv'
    chunk.to_csv(chunk_file, index=False)
    
    body = f'Solstorm run - results chunk {i + 1}'
    yag.send(subject=subject, contents=body, attachments=chunk_file)
    print(f"Email {i + 1} sent successfully with {chunk_file}!")

yag.close()