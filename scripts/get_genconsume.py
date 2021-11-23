import os
import io
from datetime import datetime
import requests

import pandas as pd
import click


REGION_DICT = {
    530000: 1,
    550000: 2,
    600000: 3,
    610000: 4,
    630000: 5,
    840000: 6
}


@click.command()
@click.option('--location', required=False,
              default="../data/gen/data.parquet", show_default=True,
              help="Data of generation/consumed")
def main(location):
    
    if os.path.exists(location):
        df = pd.read_parquet(location)
        date_from = datetime.utcfromtimestamp(
            df.M_DATE.values[-1].astype('O')/1e9).strftime("%Y.%m.%d")
    else:
        df = pd.DataFrame(columns=["INTERVAL", "M_DATE", "PRICE_ZONE_ID", "POWER_SYS_ID",
                                   "E_USE_FACT", "E_USE_PLAN", "GEN_FACT", "GEN_PLAN", "REGION"])
        date_from = datetime.now().strftime("%Y.01.01")

    date_to = datetime.now().strftime("%Y.%m.%d")

    uri_https = "https://br.so-ups.ru/"
    uri_service = "webapi/Public/Export/Csv/GenConsum.aspx?"
    terr_ids = "null:530000,null:550000,null:600000,null:610000,null:630000,null:840000"

    uri = f"{uri_https}{uri_service}startDate={date_from}&endDate={date_to}" + \
          f"&territoriesIds={terr_ids}&notCheckedColumnsNames="

    req = requests.get(uri)
    
    if req.status_code == 200:
        rawData = pd.read_csv(io.StringIO(req.content.decode('utf-8')), sep=";")
        rawData = rawData[rawData.POWER_SYS_ID.isin(REGION_DICT)]
        rawData['M_DATE'] = pd.to_datetime(rawData.M_DATE, format="%d.%m.%Y 0:00:00")
        rawData['REGION'] = rawData.POWER_SYS_ID.apply(REGION_DICT.get)
        rawData = rawData[rawData['E_USE_FACT'] > 1e-4]
        
        rawData = rawData[~rawData['M_DATE'].isin(df['M_DATE'].values)]
        
        df = df.append(rawData, ignore_index=True)
        df.drop_duplicates(inplace=True, ignore_index=True)
        df = df.sort_values(["M_DATE", "INTERVAL", "POWER_SYS_ID"])
        df = df.reset_index(drop=True)
        
        df.to_parquet(location)

    else:
        exit(1)


if __name__ == "__main__":
    main()