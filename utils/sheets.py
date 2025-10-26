import pandas as pd
from googleapiclient.discovery import build
from google.oauth2 import service_account
import os

def read_sheet(sheet_name: str) -> pd.DataFrame:
    creds = service_account.Credentials.from_service_account_file(
        "credentials.json",
    )

    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()

    SHEET_ID = os.getenv("SHEET_ID")
    result = sheet.values().get(spreadsheetId=SHEET_ID, range=sheet_name).execute()
    values = result.get('values', [])

    df = pd.DataFrame(values[1:], columns=values[0])
    return df
