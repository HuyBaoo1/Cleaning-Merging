import pandas as pd
import os
from openpyxl import load_workbook
from pandas import ExcelWriter

def save_to_excel(df, file_path, sheet_name):
    """
    Save a DataFrame to an Excel file. If the file exists, append the sheet; otherwise, create a new file.
    """
    if os.path.exists(file_path):
        # Open the existing workbook
        with ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            # Write the DataFrame to the specified sheet
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        # Create a new workbook and save the sheet
        with ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
