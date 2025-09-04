import glob
import shutil
import pandas as pd
import os

# # names = [
# #     "ABFW", "ABFX", "ABFY", "ABFZ", "ABGA", "ABGB", "ABGC", "ABGD", "ABGE", "ABGF",
# #     "ABGG", "ABGH", "ABGI", "ABGJ", "ABGK", "ABGL", "ABGM", "ABGN", "ABGO", "ABGP",
# #     "ABGQ", "ABGR", "ABGS", "ABGT", "ABGU", "ABGV", "ABGW", "ABGX", "ABGY", "ABGZ",
# #     "ABHA", "ABHB", "ABHC", "ABHD", "ABHE", "ABHF", "ABHG", "ABHH", "ABHI", "ABHJ",
# #     "ABHK", "ABHL", "ABHM", "ABHN", "ABHO", "ABHP", "ABHQ", "ABHR", "ABHS", "ABHT",
# #     "ABHU", "ABHV", "ABHW", "ABHX", "ABHY", "ABHZ", "ABIA", "ABIB", "ABIC", "ABID",
# #     "ABIE", "ABIF", "ABIG"
# # ]
# sheets = ["master-white", "master-black"]

# sheet_path = "data/hari_BC/Original/Sample list.xlsx"

# donor_dict = {}

# # # Read a specific sheet by name
# # df = pd.read_excel(sheet_path, sheet_name=sheets[1])
# # for index, row in df.iterrows():
# #     donor_id = row["Donor #"]
# #     if donor_id not in donor_dict:
# #         # Assign a name from the list if available
# #         donor_dict[donor_id] = names.pop(0)

# # Rename files
# for sheet in sheets:
#     df = pd.read_excel(sheet_path, sheet_name=sheet)
#     print(df.columns)
#     for index, row in df.iterrows():
#         pid = row["PseudoID"]
#         barcode = row["Barcode"]
#         num = barcode.split("-")
#         year = row["Donation Year"]
#         old_name = f"{barcode}.svs"
#         if len(num) == 2:
#             new_name = f"{pid}-{num[-1]}_{year}.svs"
#         else:
#             new_name = f"{pid}_{year}.svs"
#         df.loc[index, "New Name"] = new_name

#     df.to_csv(f"data/hari_BC/{sheet}.csv", index=False)

