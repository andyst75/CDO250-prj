from datetime import datetime

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import streamlit as st
import plotly.express as px


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


df_gen = pd.read_parquet("./data/gen/data.parquet")
df_sber = pd.read_parquet("./data/sber/sberindex_comsumer.parquet")

df_temp = None
for i in range(2020, 2022):
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
    min_value=-5,
    max_value=20)

dates = [datetime.utcfromtimestamp(x.astype(float)/1e9) for x in df.DATE.values]

min_date = dates[256]
max_date = dates[-1]

period_from = col1.date_input(
    label='Расчетная дата',
    value=dates[-20],
    min_value=min_date,
    max_value=max_date)

pos_data = pd.to_datetime(period_from)
pos_idx = df[df['DATE'] <= pos_data].index[-1]

data_day = torch.tensor(df.DAY.values[pos_idx - 256:pos_idx], dtype=torch.long) - 1
data_month = torch.tensor(df.MONTH.values[pos_idx - 256:pos_idx], dtype=torch.long) - 1
data_weekofyear = torch.tensor(df.WEEKOFYEAR.values[pos_idx - 256:pos_idx], dtype=torch.long) - 1
data_weekday = torch.tensor(df.WEEKDAY.values[pos_idx - 256:pos_idx], dtype=torch.long)

data_temp = torch.tensor(df.TEMP.values[pos_idx - 256:pos_idx], dtype=torch.float)
data_use_fact = torch.tensor(df.USE.values[pos_idx - 256:pos_idx], dtype=torch.float)
data_gen_fact = torch.tensor(df.GEN.values[pos_idx - 256:pos_idx], dtype=torch.float)
data_consume = torch.tensor(df.CONS_VALUE.values[pos_idx - 256:pos_idx], dtype=torch.float)

data_target = torch.tensor(df.USE.values[pos_idx:pos_idx + 90], dtype=torch.float)

date = df.DATE.values[pos_idx - 256:pos_idx]
date_target = df.DATE.values[pos_idx:pos_idx + 90]


class Model(nn.Module):
    def __init__(self, predicts=90, v_dim=256, e_dim=16, ff_dim=512):
        super(Model, self).__init__()

        self.v_dim = v_dim
        self.e_dim = e_dim
        self.predicts = predicts
        
        self.embDay        = nn.Embedding(31, e_dim)
        self.embMonth      = nn.Embedding(12, e_dim)
        self.embWeekOfYear = nn.Embedding(53, e_dim)
        self.embWeekDay    = nn.Embedding(7, e_dim)
        self.embNorm       = nn.LayerNorm(e_dim)

        self.temp          = nn.Linear(v_dim, ff_dim)
        self.tempNorm      = nn.LayerNorm(ff_dim)
        self.use_fact      = nn.Linear(v_dim, ff_dim)
        self.useNorm       = nn.LayerNorm(ff_dim)
        self.gen_fact      = nn.Linear(v_dim, ff_dim)
        self.genNorm       = nn.LayerNorm(ff_dim)
        self.consume       = nn.Linear(v_dim, ff_dim)
        self.consNorm      = nn.LayerNorm(ff_dim)
        
        self.to_embs      = nn.Linear(ff_dim * 4, ff_dim)
        
        self.ff_1           = nn.Linear(v_dim  + ff_dim, v_dim * 8)
        self.ff_2           = nn.Linear(v_dim * 8, v_dim * 8)
        self.ff_3           = nn.Linear(v_dim * 8, v_dim * 8)
        self.ff_4           = nn.Linear(v_dim * 8, v_dim * 8)
        self.ff_5           = nn.Linear(v_dim * 8, v_dim * 8)
        self.ff_6           = nn.Linear(v_dim * 8, v_dim * 8)

        self.fc             = nn.Linear(v_dim * 8, predicts)

    def forward(self, day, month, weekofyear, weekday,
                temp, use_fact, gen_fact, consume, **kwargs):
        
        emb = self.embDay(day.transpose(0, 1))
        emb.add_(self.embMonth(month.transpose(0, 1)))
        emb.add_(self.embWeekOfYear(weekofyear.transpose(0, 1)))
        emb.add_(self.embWeekDay(weekday.transpose(0, 1)))
        emb = self.embNorm(emb).transpose(0, 1).mean(axis=2)
        
        mean_use_fact = use_fact.mean(axis=1).detach().view(-1, 1)
        
        values = torch.hstack([
            self.tempNorm(self.temp(temp)),
            self.useNorm(self.use_fact(use_fact)),
            self.genNorm(self.gen_fact(gen_fact)),
            self.consNorm(self.consume(consume))
        ])
        values = self.to_embs(F.leaky_relu(values))
        
        out = torch.hstack([values, emb])

        out = self.ff_1(F.leaky_relu(out))
        out = self.ff_2(F.leaky_relu(out))
        out = self.ff_3(F.leaky_relu(out))
        out = self.ff_4(F.leaky_relu(out))
        out = self.ff_5(F.leaky_relu(out))
        out = self.ff_6(F.leaky_relu(out))

        out = mean_use_fact * (1 + out)
        
        out = self.fc(F.leaky_relu(out))

        return out
        
model = Model()

model.load_state_dict(torch.load('model/model.pth', map_location='cpu'))
model.eval()

criterion = nn.MSELoss()

with torch.no_grad():
    pred = model(data_day.unsqueeze(0), data_month.unsqueeze(0),
                 data_weekofyear.unsqueeze(0), data_weekday.unsqueeze(0),
                 data_temp.unsqueeze(0), data_use_fact.unsqueeze(0),
                 data_gen_fact.unsqueeze(0), data_consume.unsqueeze(0))
    
plot_df = pd.DataFrame()

plot_df['date'] = pd.date_range(start=pos_data, periods=90, freq='D')
plot_df['pred'] = pred[0].cpu().numpy().tolist()
np_target = np.full(90, np.nan)
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


fig = px.line(plot_df, x='date', y=['use', 'pred'], labels={'value': 'Совокупное ежесуточное потребление, MW'}, color='variable')
fig.update_layout(xaxis=dict(tickformat='%d.%m'))

st.plotly_chart(fig, use_container_width=True)

data_df = plot_df[256:].copy()

mean_mape = data_df['MAPE'].mean()

data_df['date'] = data_df['date'].apply(lambda x: str(x)[:11])
data_df = data_df.rename(columns={'date':'Дата',
                                  'use':'Факт. потребление',
                                  'pred':'Прогн. потребление',
                                  'MAPE':'MAPE, %'})
st.table(data_df)

st.write(f"Среднее значение MAPE: {mean_mape:.2f}")
