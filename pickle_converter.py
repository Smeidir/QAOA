import pandas as pd
import yagmail


with open("email_credentials.txt", "r") as f:
    email_password = f.read().strip()



yag = yagmail.SMTP("torbjorn.solstorm@gmail.com", email_password)
recipient = "torbjorn.smed@gmail.com"
subject = "Data from Python Script"
body = 'Solstorm run -papergraph - results_papergraph_depth_{2, 4, 6, 8, 10}.csv'
attachment = 'results/results_papergraph_depth_{2, 4, 6, 8, 10}_size_fixed.csv'

yag.send( subject=subject, contents=body, attachments=attachment)
print("Email sent successfully!")
yag.close()
