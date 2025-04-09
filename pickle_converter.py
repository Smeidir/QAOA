import pandas as pd
import yagmail


with open("email_credentials.txt", "r") as f:
    email_password = f.read().strip()

df = pd.read_csv("results_papergraph_depth_{2, 4, 6}.csv")
df = df.drop(columns=["obj_func_evals"], errors="ignore")
df.to_csv("results_papergraph_depth_{2, 4, 6}.csv")

df = pd.read_csv("results_papergraph_depth_{8, 10}.csv")
df = df.drop(columns=["obj_func_evals"], errors="ignore")
df.to_csv("results_papergraph_depth_{8, 10}.csv")

yag = yagmail.SMTP("torbjorn.solstorm@gmail.com", email_password)
recipient = "torbjorn.smed@gmail.com"
subject = "Data from Python Script"
body = f'Solstorm run -papergraph - results_papergraph_depth_{2, 4, 6}.csv'
attachment = f'results/results_papergraph_depth_{2, 4, 6}.csv'

yag.send( subject=subject, contents=body, attachments=attachment)
print("Email sent successfully!")
yag.close()

yag = yagmail.SMTP("torbjorn.solstorm@gmail.com", email_password)
recipient = "torbjorn.smed@gmail.com"
subject = "Data from Python Script"
body = f'Solstorm run -papergraph - results_papergraph_depth_{8, 10}.csv'
attachment = f'results/results_papergraph_depth_{8, 10}.csv'

yag.send( subject=subject, contents=body, attachments=attachment)
print("Email sent successfully!")
yag.close()