import os
import io
from datetime import datetime
import requests

import pandas as pd
import click

area_by_oes_dict = {
    1: ['Москва', 'Московская область', 'Белгородская область',
    'Владимирская область', 'Вологодская область', 'Воронежская область',
    'Ивановская область', 'Костромская область', 'Курская область',
    'Орловская область', 'Липецкая область', 'Рязанская область',
    'Брянская область', 'Калужская область', 'Смоленская область',
    'Тамбовская область', 'Тверская область', 'Тульская область',
    'Ярославская область'],
    2: ['Адыгея', 'Республика Дагестан', 'Ингушетия', 'Кабардино-Балкарская Республика',
    'Республика Калмыкия', 'Республика Карачаево-Черкессия', 'Крым',
    'Северная Осетия', 'Чечня',
    'Краснодарский край', 'Ставропольский край',
    'Астраханская область', 'Ростовская область',
    'Волгоградская область', 'Севастополь'],
    3: ['Пензенская область', 'Самарская область', 'Саратовская область',
    'Ульяновская область', 'Нижегородская область', 'Чувашская Республика',
    'Республика Марий Эл', 'Мордовия', 'Республика Татарстан'],
    4: ['Алтай', 'Республика Бурятия', 'Республика Тыва', 'Республика Хакасия',
    'Алтайский край', 'Забайкальский край',
    'Краснодарский край', 'Иркутская область',
    'Кемеровская область', 'Новосибирская область',
    'Омская область', 'Томская область'],
    5: ['Республика Башкортостан', 'Удмуртская Республика',
    'Пермский край', 'Кировская область',
    'Курганская область', 'Оренбургская область',
    'Свердловская область', 'Челябинская область',
    'Тюменская область', 'Ханты-Мансийский АО - Югра', 'Ямало-Ненецкий АО'],
    6: ['Санкт-Петербург', 'Мурманская область',
    'Калининградская область', 'Ленинградская область',
    'Новгородская область', 'Псковская область',
    'Архангельская область', 'Республика Карелия', 'Республика Коми', 'Ненецкий АО']
}
area_dict = {}
for k, v in area_by_oes_dict.items():
    area_dict.update({x:k for x in v})


@click.command()
@click.option('--location', required=False,
              default="../data/sber/sberindex_comsumer.parquet", show_default=True,
              help="Data of Sberindex comsumer")
def main(location):

    if os.path.exists(location):
        df = pd.read_parquet(location)
        date_from = df.DATE.values[-1]
    else:
        df = pd.DataFrame(columns=["DATE", "REGION", "AREA", "VALUE"])
        date_from = pd.to_datetime("2010-01-01", format="%Y-%m-%d")

    uri_https = "https://api.sberindex.ru"
    uri_service = "/source/ru/csv/indeks-potrebitelskoi-aktivnosti?representation=1&timeDivision=0"

    uri = f"{uri_https}{uri_service}"
    req = requests.get(uri)

    if req.status_code == 200:
        rawData = pd.read_csv(io.StringIO(req.content.decode('cp1251')), sep=";")

        rawData = rawData[rawData["Регион"].isin(area_dict)]
        rawData["AREA"] = rawData["Регион"].map(area_dict)
        rawData = rawData.rename(columns={"Регион":"REGION","Дата":"DATE", "Значение":"VALUE"})[["DATE", "REGION", "AREA", "VALUE"]]
        rawData['DATE'] = pd.to_datetime(rawData['DATE'], format="%Y-%m-%d")

        df = df.append(rawData[rawData.DATE > date_from], ignore_index=True)
        df = df.sort_values(["DATE", "REGION", "AREA"])
        df = df.reset_index(drop=True)

        df.to_parquet(location)
    else:
        exit(1)


if __name__ == "__main__":
    main()