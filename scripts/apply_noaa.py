import json
import tarfile
import warnings

import click
import pandas as pd


warnings.filterwarnings("ignore")

area_by_oes_dict = {
    1: ['Москва', 'Московская обл.', 'Белгородская обл.',
    'Владимирская обл.', 'Вологодская обл.', 'Воронежская обл.',
    'Ивановская обл.', 'Костромская обл.', 'Курская обл.',
    'Орловская обл.', 'Липецкая обл.', 'Рязанская обл.',
    'Брянская обл.', 'Калужская обл.', 'Смоленская обл.',
    'Тамбовская обл.', 'Тверская обл.', 'Тульская обл.',
    'Ярославская обл.'],
    2: ['Адыгея', 'Дагестан', 'Ингушетия', 'Кабардино-Балкария',
    'Калмыкия', 'Карачаево-Черкессия', 'Крым',
    'Северная Осетия', 'Чечня',
    'Краснодарский край', 'Ставропольский край',
    'Астраханская обл.', 'Ростовская обл.',
    'Волгоградская обл.', 'Севастополь'],
    3: ['Пензенская обл.', 'Самарская обл.', 'Саратовская обл.',
    'Ульяновская обл.', 'Нижегородская обл.', 'Чувашия',
    'Марий Эл', 'Мордовия', 'Татарстан'],
    4: ['Алтай', 'Бурятия', 'Тыва', 'Хакасия',
    'Алтайский край', 'Забайкальский край',
    'Краснодарский край', 'Иркутская обл.',
    'Кемеровская обл.', 'Новосибирская обл.',
    'Омская обл.', 'Томская обл.'],
    5: ['Башкортостан', 'Удмуртия',
    'Пермский край', 'Кировская обл.',
    'Курганская обл.', 'Оренбургская обл.',
    'Свердловская обл.', 'Челябинская обл.',
    'Тюменская обл.', 'ХМАО – Югра', 'Ямало-Ненецкий АО'],
    6: ['Санкт-Петербург', 'Мурманская обл.',
    'Калининградская обл.', 'Ленинградская обл.',
    'Новгородская обл.', 'Псковская обл.',
    'Архангельская обл.', 'Карелия', 'Коми', 'Ненецкий АО'],
    7: ['Саха (Якутия)', 'Красноярский край', 'Хабаровский край',
    'Чукотский АО', 'Амурская обл.', 'Магаданская обл.',
    'Камчатский край', 'Приморский край', 'Сахалинская обл.',
    'Еврейская АО'],
}

area_dict = {}
for k, v in area_by_oes_dict.items():
    area_dict.update({x:k for x in v})


@click.command()
@click.option('--isd-location', required=False,
              default="../data/isd-region-full.csv", show_default=True,
              help="Station to region mapping")
@click.option('--noaa-location', required=True, help="NOAA data in tar")
@click.option('--output-csv', '-o', required=True, help="Average temperature by region")
def main(isd_location, noaa_location, output_csv):
    
    isd = pd.read_csv(isd_location, sep=";").dropna()
    stations = {v["USAF"]: v["REGION"] for k, v in isd.iterrows()}
    
    data = pd.DataFrame(columns=["STN---", "YEARMODA", "TEMP", "VISIB", "REGION"])
    with tarfile.open(noaa_location, "r:*") as tar:
        for tar_name in tar.getnames():
            if len(tar_name) > 2 and tar_name.replace("./", "")[0] < 'A' and int(tar_name.replace("./", "").split("-")[0]) in stations:
                data_year = pd.read_fwf(tar.extractfile(tar_name), 
                                        widths=[6, 8, 8, 8, 3, 9, 3, 8, 3, 8, 2, 8], 
                                        compression='gzip')[["STN---", "YEARMODA", "TEMP"]].dropna()
                data = data.append(data_year)

    data["REGION"] = data["STN---"].apply(stations.get)
    data = data[data.REGION.isin(area_dict)]
    data["AREA"] = data["REGION"].apply(area_dict.get)
    data["TEMP"] = (data["TEMP"] - 32) * 5 / 9
    data['DATE'] = pd.to_datetime(data.YEARMODA, format="%Y%m%d")
    data_grp = data.groupby(["DATE", "REGION"], as_index=False).TEMP.agg(['median'])
    df = pd.merge(data, data_grp, how="left", left_on=["DATE", "REGION"], right_on=["DATE", "REGION"])
    df.sort_values(["DATE", "REGION", "STN---"], inplace=True)
    df = df[(df['median'] - 5. < df.TEMP ) & (df.TEMP < df['median'] + 5.)]
    df = df[['DATE', 'REGION', 'AREA', 'TEMP']]
    df = df.groupby(['DATE', 'REGION', 'AREA'], as_index=False).mean()
                
    df.to_csv(output_csv, sep=";", index=False)


if __name__ == "__main__":
    main()