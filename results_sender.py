import pandas as pd
import yagmail

df = pd.DataFrame('results_underway.csv')
df.to_csv(f'results_s_.csv')

with open("email_credentials.txt", "r") as f:
    email_password = f.read().strip()

yag = yagmail.SMTP("torbjorn.solstorm@gmail.com", email_password)
recipient = "torbjorn.smed@gmail.com"
subject = "Data from Python Script"
body = f'Solstorm run -results underway'
attachment = f'results_s_.csv'

yag.send( subject=subject, contents=body, attachments=attachment)
print("Email sent successfully!")
yag.close()