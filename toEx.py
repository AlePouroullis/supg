import pandas as pd
##Convert panda Frames to excel sheets
df = pd.read_csv(r"SUPG_experiments\BSUPGBOX")
df.to_excel(r"SUPG_experiments\BSUPGBOX.xlsx")