from datetime import datetime
import base64

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import streamlit as st
import plotly.express as px

from model.model6 import Model, PREVIOUS, PREDICT, MODEL_NAME


class Auth():
    def __init__(self):
        self.token = ""
    def update(self, token):
        self.token = token
    def get(self):
        return self.token

@st.cache(allow_output_mutation=True)
def get_auth():
    return Auth()

st.set_page_config(layout='wide')
st.title('Прогноз электропотребления в среднесрочной перспективе для ОЭС Урала')

auth = get_auth()

if (auth.get() == "") and (not 'token' in st.session_state):
    st.write("Авторизуйтесь")
    token = st.text_input("Введите имя код доступа:", key="token", placeholder="код")
    if (not 'token' in st.session_state) or (st.session_state['token'] == ""):
        st.stop()

if (auth.get() == "") and (st.session_state['token'] != "1234"):
    st.write("Введен неверный токен доступа. Обратитесь к администратору системы.")
    st.stop()

if auth.get() == "":
    auth.update(st.session_state['token'])


@st.cache(allow_output_mutation=False)
def get_df():
    df_gen = pd.read_parquet("./data/gen/data.parquet")
    df_sber = pd.read_parquet("./data/sber/sberindex_comsumer.parquet")

    df_temp = None
    for i in range(2019, 2022):
        tmp = pd.read_csv(f"./data/temp/{str(i)}.csv", sep=";")
        if df_temp is None:
            df_temp = tmp.copy()
        else:
            df_temp = df_temp.append(tmp, ignore_index=True)
        del tmp
    df_temp['DATE'] = pd.to_datetime(df_temp.DATE, format="%Y-%m-%d")

    df_gen = df_gen[df_gen.REGION == 5].drop(columns=["INTERVAL", "REGION", "POWER_SYS_ID", "PRICE_ZONE_ID", "E_USE_PLAN", "GEN_PLAN"])
    df_gen = df_gen.rename(columns={"M_DATE":"DATE", "E_USE_FACT":"USE", "GEN_FACT":"GEN"})
    df_gen = df_gen.groupby("DATE", as_index=False).sum()

    df_sber = df_sber[df_sber.AREA == 5].drop(columns=["REGION", "AREA"]).rename(columns={"VALUE":"CONS_VALUE"})
    df_sber = df_sber.groupby("DATE", as_index=False).mean()

    df_temp = df_temp[df_temp.AREA == 5].drop(columns=["REGION", "AREA"])
    df_temp = df_temp.groupby("DATE", as_index=False).mean()

    consum = df_sber.CONS_VALUE.mean()
    df_from = df_sber.DATE.min().to_pydatetime()

    add_dates = pd.date_range(start=df_temp.DATE[0], end=df_from, freq='D')[:-1]
    df_sber = df_sber.append(
        pd.DataFrame(data=zip(add_dates, [consum] * len(add_dates)), columns=df_sber.columns), ignore_index=True)
    df_sber.sort_values("DATE", inplace=True)
    df_sber.reset_index(drop=True, inplace=True)

    df_temp['WEEKDAY'] = df_temp.DATE.apply(lambda x: x.weekday())
    df_temp['MONTH'] = df_temp.DATE.apply(lambda x: x.month)
    df_temp['DAY'] = df_temp.DATE.apply(lambda x: x.day)
    df_temp['WEEKOFYEAR'] = df_temp.DATE.apply(lambda x: x.weekofyear)

    df = pd.merge(df_gen, df_temp, left_on="DATE", right_on="DATE")
    df = pd.merge(df, df_sber, left_on="DATE", right_on="DATE")    
    return df

df = get_df()


st.markdown('''
Прогнозирование в среднесрочной перспективе объем ежесуточной потребляемой мощности ОЭС Урала, с возможностью ручной корректировки прогноза в зависимости от внешних факторов.
''')

col1 = st.sidebar
col1.header('Настройки')

col1.subheader("What-If прогноз")

temperature_delta = col1.slider(
    label="Изменение средней температуры, Δt°C",
    value=0,
    min_value=-10,
    max_value=10)

consumption_index_delta = col1.slider(
    label="Изменение потребительской активности",
    value=0,
    min_value=-10,
    max_value=20)

dates = [datetime.utcfromtimestamp(x.astype(float)/1e9) for x in df.DATE.values]

min_date = dates[PREVIOUS]
max_date = dates[-1]

period_from = col1.date_input(
    label='Расчетная дата',
    value=dates[-20],
    min_value=min_date,
    max_value=max_date)

pos_data = pd.to_datetime(period_from)
pos_idx = df[df['DATE'] <= pos_data].index[-1]

data_day = torch.tensor(df.DAY.values[pos_idx - PREVIOUS:pos_idx], dtype=torch.long) - 1
data_month = torch.tensor(df.MONTH.values[pos_idx - PREVIOUS:pos_idx], dtype=torch.long) - 1
data_weekofyear = torch.tensor(df.WEEKOFYEAR.values[pos_idx - PREVIOUS:pos_idx], dtype=torch.long) - 1
data_weekday = torch.tensor(df.WEEKDAY.values[pos_idx - PREVIOUS:pos_idx], dtype=torch.long)

data_temp = torch.tensor(df.TEMP.values[pos_idx - PREVIOUS:pos_idx], dtype=torch.float) + temperature_delta
data_use_fact = torch.tensor(df.USE.values[pos_idx - PREVIOUS:pos_idx], dtype=torch.float)
data_gen_fact = torch.tensor(df.GEN.values[pos_idx - PREVIOUS:pos_idx], dtype=torch.float)
data_consume = torch.tensor(df.CONS_VALUE.values[pos_idx - PREVIOUS:pos_idx], dtype=torch.float) + consumption_index_delta * 2

data_target = torch.tensor(df.USE.values[pos_idx:pos_idx + PREDICT], dtype=torch.float)

date = df.DATE.values[pos_idx - PREVIOUS:pos_idx]
date_target = df.DATE.values[pos_idx:pos_idx + PREDICT]

device = torch.device('cpu')

@st.cache(allow_output_mutation=False)
def get_model(device):
    model = Model(device)
    model.load_state_dict(torch.load(f'model/{MODEL_NAME}', map_location='cpu'))
    model.eval()
    return model

model = get_model(device)

with torch.no_grad():
    pred = model(data_day.unsqueeze(0), data_month.unsqueeze(0),
                 data_weekofyear.unsqueeze(0), data_weekday.unsqueeze(0),
                 data_temp.unsqueeze(0), data_use_fact.unsqueeze(0),
                 data_gen_fact.unsqueeze(0), data_consume.unsqueeze(0))
    
plot_df = pd.DataFrame()

plot_df['date'] = pd.date_range(start=pos_data, periods=PREDICT, freq='D')

bias = data_target.mean().item() - pred[0][:len(data_target)].mean().cpu().item()

plot_df['pred'] = pred[0].cpu().numpy() + bias

np_target = np.full(PREDICT, np.nan)
np_target[:len(data_target)] = data_target.numpy()
plot_df['use'] = np_target
plot_df['MAPE'] = (plot_df['pred'] - plot_df['use']) * 100 / plot_df['use']

df_pre = pd.DataFrame()
df_pre['date'] = date
df_pre['use'] = data_use_fact
plot_df = df_pre.append(plot_df, ignore_index=True)


period_change_power = col1.date_input(
    label="Дата изменения потребляемой мощности",
    value=period_from,
    min_value=min_date,
    max_value=max_date)

change_power = col1.number_input(
    label="Объем изменения, МВт/сут",
    value=0)


ch_pos_data = pd.to_datetime(period_change_power)
ch_pos_idx = plot_df[plot_df['date'] <= ch_pos_data].index[-1]

change_array = np.full(len(plot_df) - ch_pos_idx, np.float(change_power))

plot_df['use'][ch_pos_idx:] += change_array
plot_df['pred'][ch_pos_idx:] += change_array

# y_min = ((plot_df[['use', 'pred']].values.min() - 10_000) // 10_000) * 10_000
# y_max = ((plot_df[['use', 'pred']].values.max() + 10_000) // 10_000) * 10_000

fig = px.line(plot_df.rename(columns={'use':'фактическое', 'pred':'прогнозное'}),
                             x='date', y=['фактическое', 'прогнозное'],
              labels={'date':'Дата', 'value': 'Потребление, MW'},
              title='График ежесуточного потребления',
#               range_y = [y_min, y_max],
              color='variable')
fig.update_layout(xaxis=dict(tickformat='%d.%m'), legend_title_text="Потребление")

fig.update_yaxes(visible=True, fixedrange=True)

st.plotly_chart(fig, use_container_width=True)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="energy.csv">Download CSV File</a>'
    return href

data_df = plot_df[PREVIOUS:].copy()
mean_mape = data_df['MAPE'].mean()


data_df['date'] = data_df['date'].apply(lambda x: str(x)[:11])
data_df = data_df.rename(columns={'date':'Дата',
                                  'use':'Факт. потребление',
                                  'pred':'Прогн. потребление',
                                  'MAPE':'MAPE, %'})

st.markdown(filedownload(data_df.reset_index(drop=True)), unsafe_allow_html=True)
st.table(data_df.style.format(na_rep=' ',
                formatter={'Факт. потребление': "{:.0f}",
                           'Прогн. потребление': "{:.0f}",
                           'MAPE, %': lambda x: "" if np.isnan(x) else "{:.2f}".format(x)
                          })
        )

# st.write(f"Среднее значение MAPE: {mean_mape:.2f}")
