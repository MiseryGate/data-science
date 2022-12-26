from numpy.core.fromnumeric import size
from numpy.lib.function_base import select
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import re
import matplotlib.pyplot as plt
import base64
import streamlit.components.v1 as stc
import os
import plotly.express as px
from matplotlib.figure import Figure
from plotly import graph_objs as go
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
#import pydea as dea
scaler = MinMaxScaler()
LOGO_IMAGE = "./bluspeedrun_trans.png"
#Baca Dataset awal
data = pd.read_excel('./dataset/data_dea.xlsx')
#Data untuk korelasi
data_ok = data.drop(['Tahun','Kode_satker'],axis=1)
#Data untuk Plotting (sum)
data_plot = data.groupby("Tahun", dropna=False).sum().reset_index()
#Data untuk Plotting (rata-rata)
data_plot_m = data.groupby("Tahun", dropna=False).mean().reset_index()
#Data Surplus Defisit
data['Surplus_Defisit'] = data['Pendapatan'] - data ['Belanja']
#Baca Data DEA (RS Khusus)
data_khusus = pd.read_excel('./dataset/dea_khusus.xlsx')
#Baca Data Clustering (2016)
data_2016 = pd.read_excel('./dataset/dea_2016.xlsx')
#Baca Data Clustering (2017)
data_2017 = pd.read_excel('./dataset/dea_2017.xlsx')
#Baca Data Clustering (2018)
data_2018 = pd.read_excel('./dataset/dea_2018.xlsx')
#Baca Data Clustering (2019)
data_2019 = pd.read_excel('./dataset/dea_2019.xlsx')
#Baca Data Clustering (2020)
data_2020 = pd.read_excel('./dataset/dea_2020.xlsx')
#Baca Data Rekap DEA
tempat_tidur = pd.read_excel('./dataset/tempat_tidur.xlsx')
sdm = pd.read_excel('./dataset/sdm.xlsx')
rawat_jalan= pd.read_excel('./dataset/rawat_jalan.xlsx')
rawat_inap = pd.read_excel('./dataset/rawat_inap.xlsx')
pobo = pd.read_excel('./dataset/pobo.xlsx')
#fungsi kelas rs umum
def tipe_umum (x):
    if (x >= 250):
        return 'A'
    elif (x >= 200):
        return 'B'
    elif (x >= 100):
        return 'C'
    elif (x >= '50'):
        return 'D'
    return 'error'
#fungsi kelas rs khusus
def tipe_khusus (x):
    if (x >= 100):
        return 'A'
    elif (x >= 75):
        return 'B'
    elif (x >= 25):
        return 'C'
    return 'error'
#Fungsi Plotting
def show_values_on_bars(axs, h_v="v", space=0.4):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
#Klaster Kelas
def klaster_kelas(data):
    umum = [415423,415432,415582,538815,415618,415624,548886,415520,415479,415661,415448,415536,415573,532214,258462,415630,415397]
    data['Jenis_rs'] = np.where(data.Kode_satker.isin(umum),'Umum','Khusus')

#Fungsi Input dan Output untuk pyDEA
def input(data):
    inputs = data[['SDM_medis_total','SDM_non_medis_total','Jumlah_tempat_tidur','Belanja']]
    return inputs
def output(data):
    outputs = data[['Pasien_rawat_inap_orang','Pasien_rawat_jalan_orang','BOR_persen','ALOS_hari','BTO_kali','TOI_hari','Pendapatan']]
    return outputs
#Disable Warning
st.set_option('deprecation.showPyplotGlobalUse', False)
#Set Size
sns.set(rc={'figure.figsize':(8,8)})
#Coloring
colors_1 = ['#66b3ff','#99ff99']
colors_2 = ['#66b3ff','#99ff99']
colors_3 = ['#79ff4d','#4d94ff']
colors_4 = ['#ff0000','#ff1aff']
st.markdown(
    f"""
    <div style="text-align: center;">
    <img class="logo-img" src="data:png;base64,{base64.b64encode(open(LOGO_IMAGE, 'rb').read()).decode()}">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align: center; color: #243A74; font-family:sans-serif'>DASHBOARD DEA BLU</h1>", unsafe_allow_html=True)

menu = st.sidebar.selectbox("Select Menu", ("Analisis Deskriptif","Analisis Per BLU","Analisis Komparatif", "DEA Processing","Unggah/Unduh Data","Prediction"))
if menu == "Analisis Deskriptif":
    st.write("""# Analisis Deskriptif""")
    st.write("Banyak Data", data.shape)
    st.write("Banyaknya Satuan Kerja RS BLU saat ini adalah ", len(data['Nama_satker'].unique()), " satuan kerja")
    st.write("Gambaran Data")
    desc = data.describe()
    st.write(desc)
    st.write("Total Pendapatan dan Belanja BLU")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.groupby("Tahun", dropna=False).sum().reset_index().Tahun, y=data.groupby("Tahun", dropna=False).sum().reset_index().Pendapatan, name="Pendapatan"))
    fig.add_trace(go.Scatter(x=data.groupby("Tahun", dropna=False).sum().reset_index().Tahun, y=data.groupby("Tahun", dropna=False).sum().reset_index().Belanja, name="Belanja"))
    fig.layout.update(title_text="Data Pendapatan dan Belanja BLU", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    # fig_3d = px.scatter_3d(
    # data_frame=data,
    # x="Tahun",
    # y="Pasien_rawat_inap_orang",
    # z="Jumlah_tempat_tidur",
    # color="Pasien_rawat_inap_orang",
    # title="3D Scatter Jumlah Tempat Tidur dan Pasien Rawat Inap")
    # st.plotly_chart(fig_3d)
    st.write("SDM Medis dan Non Medis")
    X_axis = np.arange(len(data.groupby('Tahun').sum()[['SDM_medis_total','SDM_non_medis_total']]))
    plt.bar(X_axis - 0.2, data.groupby('Tahun').sum()['SDM_medis_total'], 0.4, label = 'SDM Medis')
    plt.bar(X_axis + 0.2, data.groupby('Tahun').sum()['SDM_non_medis_total'], 0.4, label = 'SDM Non Medis')
    plt.xticks(X_axis, data.groupby('Tahun').sum().index)
    plt.xlabel("Tahun")
    plt.ylabel("Banyaknya SDM")
    plt.title("Total SDM Medis dan Non Medis")
    plt.legend()
    st.pyplot()
    col1, col2 = st.columns(2)
    with col1:
        st.write("Jumlah Pasien Rawat Inap (ribu orang)")
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(x=data_plot['Tahun'],
                    y=data_plot['Pasien_rawat_inap_orang']/1000, ax=ax)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Pasien Rawat Inap (ribu orang)')
        st.pyplot(fig)
    with col2:
        st.write("Jumlah Pasien Rawat Jalan (ribu orang)")
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(x=data_plot['Tahun'],
                    y=data_plot['Pasien_rawat_jalan_orang']/1000, ax=ax)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Pasien Rawat Jalan (ribu orang)')
        st.pyplot(fig)
    subcol1, subcol2, subcol3, subcol4 = st.columns(4)
    with subcol1:
        st.write("Rata-rata BOR (%)")
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(x=data_plot_m['Tahun'],
                    y=data_plot_m['BOR_persen'], ax=ax)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Bed Occupancy Ratio (%)')
        st.pyplot(fig)
    with subcol2:
        st.write("Rata-rata ALOS (hari)")
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(x=data_plot_m['Tahun'],
                    y=data_plot_m['ALOS_hari'], ax=ax)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Average Length of Stay (hari)')
        st.pyplot(fig)
    with subcol3:
        st.write("Rata-rata BTO (kali)")
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(x=data_plot_m['Tahun'],
                    y=data_plot_m['BTO_kali'], ax=ax)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Bed Turn Over (kali)')
        st.pyplot(fig)
    with subcol4:
        st.write("Rata-rata TOI (hari)")
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(x=data_plot_m['Tahun'],
                    y=data_plot_m['TOI_hari'], ax=ax)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Turn Over Interval (hari)')
        st.pyplot(fig)
    st.write("Korelasi Data")
    fig = plt.figure(figsize=(16, 8))
    matrix = np.triu(data_ok.corr())
    heatmap = sns.heatmap(data_ok.corr(), vmin=-1, vmax=1, annot=True,cmap='coolwarm',mask = matrix)
    heatmap.set_title('Correlation Heatmap BLU Rumah Sakit', fontdict={'fontsize':12}, pad=20)
    st.pyplot(fig)
if menu == 'Analisis Per BLU':
    st.write("""# Analisis Per BLU""")
    pilih_blu = st.selectbox('Pilih BLU Rumah Sakit',data['Nama_satker'].unique())
    st.sidebar.write("Pilih data yang akan ditampilkan : ")
    pilih_sdm = st.sidebar.checkbox('SDM Medis dan Non Medis')
    pilih_uang = st.sidebar.checkbox('Pendapatan dan Belanja')
    pilih_tidur = st.sidebar.checkbox('Jumlah Tempat Tidur')
    pilih_bor = st.sidebar.checkbox('BOR')
    pilih_toi = st.sidebar.checkbox('TOI')
    pilih_alos = st.sidebar.checkbox('ALOS')
    pilih_bto = st.sidebar.checkbox('BTO')
    pilih_pasien = st.sidebar.checkbox('Jumlah Pasien')
    for item in data['Nama_satker'].unique():
            if item == pilih_blu:
                st.write('### Data ' + pilih_blu)
                st.write(data[(data.Nama_satker == item)])
    agree = st.checkbox('Tampilkan Korelasi Data?')
    if agree:
        st.write("Korelasi Data")
        fig = plt.figure(figsize=(16, 8))
        matrix = np.triu(data[(data.Nama_satker == pilih_blu)].corr())
        heatmap = sns.heatmap(data[(data.Nama_satker == pilih_blu)].corr(), vmin=-1, vmax=1, annot=True,cmap='coolwarm',mask = matrix)
        heatmap.set_title('Correlation Heatmap Satker ' + str(pilih_blu), fontdict={'fontsize':12}, pad=20)
        st.pyplot(fig)
    if pilih_uang:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[(data.Nama_satker == pilih_blu)].Tahun, y=data[(data.Nama_satker == pilih_blu)].Pendapatan, name="Pendapatan"))
        fig.add_trace(go.Scatter(x=data[(data.Nama_satker == pilih_blu)].Tahun, y=data[(data.Nama_satker == pilih_blu)].Belanja, name="Belanja"))
        fig.layout.update(title_text="Data Pendapatan dan Belanja BLU " + pilih_blu, xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    if pilih_sdm:
        st.write("SDM Medis dan Non Medis")
        X_axis = np.arange(len(data[(data.Nama_satker == pilih_blu)][['SDM_medis_total','SDM_non_medis_total']]))
        plt.bar(X_axis - 0.2, data[(data.Nama_satker == pilih_blu)]['SDM_medis_total'], 0.4, label = 'SDM Medis')
        plt.bar(X_axis + 0.2, data[(data.Nama_satker == pilih_blu)]['SDM_non_medis_total'], 0.4, label = 'SDM Non Medis')
        plt.xticks(X_axis, data[(data.Nama_satker == pilih_blu)]['Tahun'])
        plt.xlabel("Tahun")
        plt.ylabel("Banyaknya SDM")
        plt.title("Total SDM Medis dan Non Medis")
        plt.legend()
        st.pyplot()
    if pilih_tidur:
        st.write('### Jumlah Tempat Tidur ' + pilih_blu)
        data_plot_tidur = data[(data.Nama_satker == pilih_blu)][['Tahun','Jumlah_tempat_tidur']]
        ax1 = sns.barplot(x="Tahun", y="Jumlah_tempat_tidur", data=data_plot_tidur)
        show_values_on_bars(ax1)
        st.pyplot()
    if pilih_bor:
        st.write('### Bed Occupancy Ratio (BOR) ' + pilih_blu + ' (dalam persen)')
        data_plot_bor = data[(data.Nama_satker == pilih_blu)][['Tahun','BOR_persen']]
        ax1 = sns.barplot(x="Tahun", y="BOR_persen", data=data_plot_bor)
        show_values_on_bars(ax1)
        st.pyplot()
    if pilih_toi:
        st.write('### Turn Over Interval (TOI) ' + pilih_blu + ' (kali/frekuensi) ')
        data_plot_toi = data[(data.Nama_satker == pilih_blu)][['Tahun','TOI_hari']]
        ax1 = sns.barplot(x="Tahun", y="TOI_hari", data=data_plot_toi)
        show_values_on_bars(ax1)
        st.pyplot()
    if pilih_alos:
        st.write('### Average Length of Stay (ALOS) ' + pilih_blu + ' (hari)')
        data_plot_alos = data[(data.Nama_satker == pilih_blu)][['Tahun','ALOS_hari']]
        ax1 = sns.barplot(x="Tahun", y="ALOS_hari", data=data_plot_alos)
        show_values_on_bars(ax1)
        st.pyplot()
    if pilih_bto:
        st.write('### Bed Turn Over (BTO) ' + pilih_blu + ' (kali)')
        data_plot_bto = data[(data.Nama_satker == pilih_blu)][['Tahun','BTO_kali']]
        ax1 = sns.barplot(x="Tahun", y="BTO_kali", data=data_plot_bto)
        show_values_on_bars(ax1)
        st.pyplot()
    if pilih_pasien:
        st.write('### Jumlah Pasien ' + pilih_blu)
        X_axis = np.arange(len(data[(data.Nama_satker == pilih_blu)][['Pasien_rawat_inap_orang','Pasien_rawat_jalan_orang']]))
        plt.bar(X_axis - 0.2, data[(data.Nama_satker == pilih_blu)]['Pasien_rawat_inap_orang'], 0.4, label = 'Jumlah Pasien Rawat Jalan')
        plt.bar(X_axis + 0.2, data[(data.Nama_satker == pilih_blu)]['Pasien_rawat_jalan_orang'], 0.4, label = 'Jumlah Pasien Rawat Inap')
        plt.xticks(X_axis, data[(data.Nama_satker == pilih_blu)]['Tahun'])
        plt.xlabel("Tahun")
        plt.ylabel("Jumlah Pasien")
        plt.title("Total Pasien Rawat Inap dan Rawat Jalan")
        plt.legend()
        st.pyplot() 
if menu == 'Analisis Komparatif':
    st.sidebar.write("Pilih data yang akan ditampilkan : ")
    pilih_1 = st.sidebar.checkbox('SDM Medis dan Non Medis')
    pilih_2 = st.sidebar.checkbox('Pendapatan dan Belanja')
    pilih_3 = st.sidebar.checkbox('Jumlah Tempat Tidur')
    pilih_4 = st.sidebar.checkbox('BOR')
    pilih_5 = st.sidebar.checkbox('TOI')
    pilih_6 = st.sidebar.checkbox('ALOS')
    pilih_7 = st.sidebar.checkbox('BTO')
    pilih_8 = st.sidebar.checkbox('Jumlah Pasien')
    col1, col2 = st.columns(2)
    with col1:
        st.write("Pilih Rumah Sakit")
        pilih_blu_banding = st.selectbox('Pilih BLU Rumah Sakit Pertama',data['Nama_satker'].unique())
        for item in data['Nama_satker'].unique():
            if item == pilih_blu_banding:
                st.write(data[(data.Nama_satker == item)])
        if pilih_1:
            st.write("SDM Medis dan Non Medis")
            X_axis = np.arange(len(data[(data.Nama_satker == pilih_blu_banding)][['SDM_medis_total','SDM_non_medis_total']]))
            plt.bar(X_axis - 0.2, data[(data.Nama_satker == pilih_blu_banding)]['SDM_medis_total'], 0.4, label = 'SDM Medis')
            plt.bar(X_axis + 0.2, data[(data.Nama_satker == pilih_blu_banding)]['SDM_non_medis_total'], 0.4, label = 'SDM Non Medis')
            plt.xticks(X_axis, data[(data.Nama_satker == pilih_blu_banding)]['Tahun'])
            plt.xlabel("Tahun")
            plt.ylabel("Banyaknya SDM")
            plt.title("Total SDM Medis dan Non Medis")
            plt.legend()
            st.pyplot()
        if pilih_8:
            st.write('Jumlah Pasien Rawat Inap dan Rawat Jalan (orang)')
            X_axis = np.arange(len(data[(data.Nama_satker == pilih_blu_banding)][['Pasien_rawat_inap_orang','Pasien_rawat_jalan_orang']]))
            plt.bar(X_axis - 0.2, data[(data.Nama_satker == pilih_blu_banding)]['Pasien_rawat_inap_orang'], 0.4, label = 'Jumlah Pasien Rawat Jalan')
            plt.bar(X_axis + 0.2, data[(data.Nama_satker == pilih_blu_banding)]['Pasien_rawat_jalan_orang'], 0.4, label = 'Jumlah Pasien Rawat Inap')
            plt.xticks(X_axis, data[(data.Nama_satker == pilih_blu_banding)]['Tahun'])
            plt.xlabel("Tahun")
            plt.ylabel("Jumlah Pasien")
            plt.title("Total Pasien Rawat Inap dan Rawat Jalan")
            plt.legend()
            st.pyplot()   
    with col2:
        st.write("Pilih Rumah Sakit")
        data_baru = data[data.Nama_satker != item]
        pilih_blu_banding_2 = st.selectbox('Pilih BLU Rumah Sakit Pembanding',data_baru['Nama_satker'].unique()) #tambahin unique value
        for item1 in data_baru['Nama_satker'].unique():
            if item1 == pilih_blu_banding_2:
                st.write(data_baru[(data_baru.Nama_satker == item1)])
        if pilih_1:
            st.write("SDM Medis dan Non Medis")
            X_axis = np.arange(len(data[(data.Nama_satker == pilih_blu_banding_2)][['SDM_medis_total','SDM_non_medis_total']]))
            plt.bar(X_axis - 0.2, data[(data.Nama_satker == pilih_blu_banding_2)]['SDM_medis_total'], 0.4, label = 'SDM Medis')
            plt.bar(X_axis + 0.2, data[(data.Nama_satker == pilih_blu_banding_2)]['SDM_non_medis_total'], 0.4, label = 'SDM Non Medis')
            plt.xticks(X_axis, data[(data.Nama_satker == pilih_blu_banding_2)]['Tahun'])
            plt.xlabel("Tahun")
            plt.ylabel("Banyaknya SDM")
            plt.title("Total SDM Medis dan Non Medis")
            plt.legend()
            st.pyplot()
        if pilih_8:
            st.write('Jumlah Pasien Rawat Inap dan Rawat Jalan (orang)')
            X_axis = np.arange(len(data[(data.Nama_satker == pilih_blu_banding_2)][['Pasien_rawat_inap_orang','Pasien_rawat_jalan_orang']]))
            plt.bar(X_axis - 0.2, data[(data.Nama_satker == pilih_blu_banding_2)]['Pasien_rawat_inap_orang'], 0.4, label = 'Jumlah Pasien Rawat Jalan')
            plt.bar(X_axis + 0.2, data[(data.Nama_satker == pilih_blu_banding_2)]['Pasien_rawat_jalan_orang'], 0.4, label = 'Jumlah Pasien Rawat Inap')
            plt.xticks(X_axis, data[(data.Nama_satker == pilih_blu_banding_2)]['Tahun'])
            plt.xlabel("Tahun")
            plt.ylabel("Jumlah Pasien")
            plt.title("Total Pasien Rawat Inap dan Rawat Jalan")
            plt.legend()
            st.pyplot()
    if pilih_2:
            st.write("Pendapatan dan Belanja " + str (pilih_blu_banding))
            fig = go.Figure()
            fig.add_trace(
            go.Scatter(
                x=data['Tahun'],
                y=data[(data.Nama_satker == pilih_blu_banding)].Pendapatan/1000000,
                name="Pendapatan",
                mode='lines+markers', 
            #         mode = 'lines'
                marker= dict(size=9,
                                symbol = 'diamond',
                                color ='RGB(251, 177, 36)',
                                line_width = 2
                            ),
                line = dict(color='firebrick', width=3)
            ))
            fig.add_trace(
            go.Bar(
                x=data['Tahun'],
                y=data[(data.Nama_satker == pilih_blu_banding)].Surplus_Defisit/1000000,
                name="Surplus/Defisit",
                text = data[(data.Nama_satker == pilih_blu_banding)].Surplus_Defisit,
                textposition='outside',
                textfont=dict(
                size=13,
                color='#1f77b4'),      
                marker_color=["#f3e5f5", '#e1bee7', '#ce93d8', '#ba68c8','#ab47bc',
                                '#9c27b0','#8e24aa','#7b1fa2','#6a1b9a','#4a148c','#3c0a99'],
                marker_line_color='rgb(17, 69, 126)',
                marker_line_width=1, 
                opacity=0.7
            ))# strip down the rest of the plot
            fig.update_layout(
            showlegend=True,
            plot_bgcolor="rgb(240,240,240)",
            margin=dict(t=50,l=10,b=10,r=10),
            title_text='2016-2020 Financial Report',
            title_font_family="Times New Roman",
            title_font_size = 25,
            title_font_color="darkblue",
            title_x=0.5,
            xaxis=dict(
                tickfont_size=14,
                tickangle = 270,
                showgrid = True,
                zeroline = True,
                showline = True,
                showticklabels = True,
                dtick=1
            ),
            yaxis=dict(
                title='dalam jutaan rupiah',
                titlefont_size=16,
                tickfont_size=14
            ),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)'
            ),
            bargap=0.15
            )
            fig.update_traces(texttemplate='%{text:.2s}')
            st.plotly_chart(fig)
            st.write("Pendapatan dan Belanja " + str(pilih_blu_banding_2))
            fig = go.Figure()
            fig.add_trace(
            go.Scatter(
                x=data['Tahun'],
                y=data[(data.Nama_satker == pilih_blu_banding_2)].Pendapatan/1000000,
                name="Pendapatan",
                mode='lines+markers', 
            #         mode = 'lines'
                marker= dict(size=9,
                                symbol = 'diamond',
                                color ='RGB(251, 177, 36)',
                                line_width = 2
                            ),
                line = dict(color='firebrick', width=3)
            ))
            fig.add_trace(
            go.Bar(
                x=data['Tahun'],
                y=data[(data.Nama_satker == pilih_blu_banding_2)].Surplus_Defisit/1000000,
                name="Surplus/Defisit",
                text = data[(data.Nama_satker == pilih_blu_banding_2)].Surplus_Defisit,
                textposition='outside',
                textfont=dict(
                size=13,
                color='#1f77b4'),      
                marker_color=["#f3e5f5", '#e1bee7', '#ce93d8', '#ba68c8','#ab47bc',
                                '#9c27b0','#8e24aa','#7b1fa2','#6a1b9a','#4a148c','#3c0a99'],
                marker_line_color='rgb(17, 69, 126)',
                marker_line_width=1, 
                opacity=0.7
            ))# strip down the rest of the plot
            fig.update_layout(
            showlegend=True,
            plot_bgcolor="rgb(240,240,240)",
            margin=dict(t=50,l=10,b=10,r=10),
            title_text='2016-2020 Financial Report',
            title_font_family="Times New Roman",
            title_font_size = 25,
            title_font_color="darkblue",
            title_x=0.5,
            xaxis=dict(
                tickfont_size=14,
                tickangle = 270,
                showgrid = True,
                zeroline = True,
                showline = True,
                showticklabels = True,
                dtick=1
            ),
            yaxis=dict(
                title='dalam jutaan rupiah',
                titlefont_size=16,
                tickfont_size=14
            ),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)'
            ),
            bargap=0.15
            )
            fig.update_traces(texttemplate='%{text:.2s}')
            st.plotly_chart(fig)
    if pilih_3 :
        st.write("Jumlah Tempat Tidur")
        X_axis = np.arange(len(data[(data.Nama_satker == pilih_blu_banding_2)][['Jumlah_tempat_tidur']]))
        plt.bar(X_axis - 0.2, data[(data.Nama_satker == pilih_blu_banding)]['Jumlah_tempat_tidur'], 0.4, label = 'Jumlah Tempat Tidur ' + str(pilih_blu_banding))
        plt.bar(X_axis + 0.2, data[(data.Nama_satker == pilih_blu_banding_2)]['Jumlah_tempat_tidur'], 0.4, label = 'Jumlah Tempat Tidur ' + str(pilih_blu_banding_2))
        plt.xticks(X_axis, data[(data.Nama_satker == pilih_blu_banding_2)]['Tahun'])
        plt.xlabel("Tahun")
        plt.ylabel("Jumlah Tempat Tidur")
        plt.title("Perbandingan Jumlah Tempat Tidur")
        plt.legend()
        st.pyplot()
    if pilih_4 :
        st.write("Bed Occupancy Ratio")
        X_axis = np.arange(len(data[(data.Nama_satker == pilih_blu_banding_2)][['BOR_persen']]))
        plt.bar(X_axis - 0.2, data[(data.Nama_satker == pilih_blu_banding)]['BOR_persen'], 0.4, label = 'BOR ' + str(pilih_blu_banding))
        plt.bar(X_axis + 0.2, data[(data.Nama_satker == pilih_blu_banding_2)]['BOR_persen'], 0.4, label = 'BOR ' + str(pilih_blu_banding_2))
        plt.xticks(X_axis, data[(data.Nama_satker == pilih_blu_banding_2)]['Tahun'])
        plt.xlabel("Tahun")
        plt.ylabel("Bed Occupancy Ratio (%)")
        plt.title("Perbandingan BOR (%)")
        plt.legend()
        st.pyplot()
    if pilih_5:
        st.write("Turn Over Interval (hari)")
        X_axis = np.arange(len(data[(data.Nama_satker == pilih_blu_banding_2)][['TOI_hari']]))
        plt.bar(X_axis - 0.2, data[(data.Nama_satker == pilih_blu_banding)]['TOI_hari'], 0.4, label = 'TOI ' + str(pilih_blu_banding))
        plt.bar(X_axis + 0.2, data[(data.Nama_satker == pilih_blu_banding_2)]['TOI_hari'], 0.4, label = 'TOI ' + str(pilih_blu_banding_2))
        plt.xticks(X_axis, data[(data.Nama_satker == pilih_blu_banding_2)]['Tahun'])
        plt.xlabel("Tahun")
        plt.ylabel("TOI (hari)")
        plt.title("Perbandingan Turn Over Interval (hari)")
        plt.legend()
        st.pyplot()
    if pilih_6 :
        st.write("Average Lenght of Stay (hari)")
        X_axis = np.arange(len(data[(data.Nama_satker == pilih_blu_banding_2)][['ALOS_hari']]))
        plt.bar(X_axis - 0.2, data[(data.Nama_satker == pilih_blu_banding)]['ALOS_hari'], 0.4, label = 'ALOS ' + str(pilih_blu_banding))
        plt.bar(X_axis + 0.2, data[(data.Nama_satker == pilih_blu_banding_2)]['ALOS_hari'], 0.4, label = 'ALOS ' + str(pilih_blu_banding_2))
        plt.xticks(X_axis, data[(data.Nama_satker == pilih_blu_banding_2)]['Tahun'])
        plt.xlabel("Tahun")
        plt.ylabel("Average Lenght of Stay (hari)")
        plt.title("Perbandingan ALOS (hari)")
        plt.legend()
        st.pyplot()
    if pilih_7 :
        st.write("Bed Turn Over")
        X_axis = np.arange(len(data[(data.Nama_satker == pilih_blu_banding_2)][['BTO_kali']]))
        plt.bar(X_axis - 0.2, data[(data.Nama_satker == pilih_blu_banding)]['BTO_kali'], 0.4, label = 'BTO ' + str(pilih_blu_banding))
        plt.bar(X_axis + 0.2, data[(data.Nama_satker == pilih_blu_banding_2)]['BTO_kali'], 0.4, label = 'BTO ' + str(pilih_blu_banding_2))
        plt.xticks(X_axis, data[(data.Nama_satker == pilih_blu_banding_2)]['Tahun'])
        plt.xlabel("Tahun")
        plt.ylabel("Bed Turn Over (kali)")
        plt.title("Perbandingan BTO (kali)")
        plt.legend()
        st.pyplot()
if menu == 'DEA Processing':
    #Mengklaster berdasarkan kelas (2016-2020)
    klaster_kelas(data_2016)
    klaster_kelas(data_2017)
    klaster_kelas(data_2018)
    klaster_kelas(data_2019)
    klaster_kelas(data_2020)
    #menambahkan kolom kelas pada rs umum
    def kelas_rs(data):
        data['Kelas_rs'] = data[(data.Jenis_rs=='Umum')]['Jumlah_tempat_tidur'].apply(tipe_umum)
        data.loc[(data.Jenis_rs=='Khusus'), ['Kelas_rs']] = data[(data.Jenis_rs=='Khusus')]['Jumlah_tempat_tidur'].apply(tipe_khusus)
        return data
    #plotting pie chart
    def plot_pie_dea (data):
        st.write('Jumlah RS berdasarkan Kelas Tahun', select_tahun_gam)
        labels = data['Kelas_rs'].value_counts().index
        values = data['Kelas_rs'].value_counts().values
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        st.plotly_chart(fig)
        st.write('Jumlah RS berdasarkan Jenis Tahun ', select_tahun_gam)
        labels = data['Jenis_rs'].value_counts().index
        values = data['Jenis_rs'].value_counts().values
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        return st.plotly_chart(fig)
    #Define Dataframe Baru dengan Menambahkan kelas dan jenis RS
    data_16_ok = kelas_rs(data_2016)
    data_17_ok = kelas_rs(data_2017)
    data_18_ok = kelas_rs(data_2018)
    data_19_ok = kelas_rs(data_2019)
    data_20_ok = kelas_rs(data_2020)
    #Plotting PCA
    def pca(data):
        selcol = ['SDM_medis_total', 'SDM_non_medis_total', 'Jumlah_tempat_tidur', 'Belanja', 'Pasien_rawat_inap_orang', 'Pasien_rawat_jalan_orang', 'BOR_persen', 'ALOS_hari','BTO_kali', 'TOI_hari','Pendapatan']
        scaled = data.drop(columns=['Nama_satker','Jenis_rs','Kelas_rs'], axis=1)
        scalers = []
        for tahun in scaled.Tahun.values:
            scaler.fit(scaled.loc[scaled['Tahun'] == tahun,selcol])
            scalers.append(scaler)
            scaled.loc[scaled['Tahun'] == tahun,selcol] = scaler.transform(scaled.loc[scaled['Tahun'] == tahun,selcol])
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
            kmeans.fit(scaled)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('Metode Elbow')
        plt.xlabel('Jumlah clusters')
        plt.ylabel('WCSS')
        return st.pyplot()
    def clustering(data):
        X = data.loc[:, ['Belanja','Pendapatan']].values
        kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 123)
        y_kmeans = kmeans.fit_predict(X)
        # Visualisasi hasil clusters
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
        #plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')
        plt.title('Cluster Rumah Sakit Umum')
        plt.xlabel('Belanja')
        plt.ylabel('Pendapatan')
        plt.legend()
        return st.pyplot()
    #Menu Selectbox DEA Analysis
    selectbox_dea = st.selectbox('Pilih Analisis',('Gambaran Data','Cek Outlier','Clustering','DEA Analysis (terminated)','DEA Solver'))
    #Menu Gambaran Data
    if selectbox_dea == 'Gambaran Data':
        select_tahun_gam = st.sidebar.selectbox('Pilih tahun : ',('2016','2017','2018','2019','2020'))
        if select_tahun_gam == '2016':
            plot_pie_dea(data_2016)
        if select_tahun_gam == '2017':
            plot_pie_dea(data_2017)
        if select_tahun_gam == '2018':
            plot_pie_dea(data_2018)
        if select_tahun_gam == '2019':
            plot_pie_dea(data_2019)
        if select_tahun_gam == '2020':
            plot_pie_dea(data_2020)
    #Menu PCA dan Clustering
    if selectbox_dea == 'Clustering':
        st.write ('Clustering dengan KMeans')
        select_tahun = st.sidebar.selectbox('Pilih tahun : ',('2016','2017','2018','2019','2020'))
        radio_dea = st.sidebar.radio('Pilih Data' , ('Mencari Elbow', 'Clustering Kmeans'))
        if radio_dea == 'Mencari Elbow':
            if select_tahun == '2016':
                pca(data_16_ok)
            if select_tahun == '2017':
                pca(data_17_ok)
            if select_tahun == '2018':
                pca(data_18_ok)
            if select_tahun == '2019':
                pca(data_19_ok)
            if select_tahun == '2020':
                pca(data_20_ok)
        if radio_dea == 'Clustering Kmeans':
            if select_tahun == '2016':
                clustering(data_16_ok)
            if select_tahun == '2017':
                clustering(data_17_ok)
            if select_tahun == '2018':
                clustering(data_18_ok)
            if select_tahun == '2019':
                clustering(data_19_ok)
            if select_tahun == '2020':
                clustering(data_20_ok)
    #Menu Outlier
    if selectbox_dea == 'Cek Outlier':
        select_tahun_outlier = st.sidebar.selectbox('Pilih tahun : ',('2016','2017','2018','2019','2020'))
        radio_out = st.sidebar.radio('Pilih Data',('TOI','BOR','ALOS','BTO'))
        if radio_out == 'BOR':
            data['BOR_persen'].plot.box()
            st.pyplot()
            bor = data['BOR_persen']
            q1_bor = bor.quantile(0.25)
            q3_bor = bor.quantile(0.75)

            iqr_bor = q3_bor - q1_bor
            iqr_lower_bor = q1_bor - 1.5 * iqr_bor
            iqr_upper_bor = q3_bor + 1.5 * iqr_bor
            st.write('### Satker BLU yang merupakan outlier pada variabel BOR')
            st.write('BOR (Bed Occupancy Ratio), idealnya 60-85% (Kemenkes, 2011)')
            st.write(data[(data.BOR_persen < iqr_lower_bor) | (data.BOR_persen > iqr_upper_bor)][['Kode_satker','Tahun','Nama_satker','BOR_persen']].sort_values('BOR_persen', ascending=False))
        if radio_out == 'ALOS':
            data['ALOS_hari'].plot.box()
            st.pyplot()
            alos = data['ALOS_hari']
            q1_alos = alos.quantile(0.25)
            q3_alos = alos.quantile(0.75)

            iqr_alos = q3_alos - q1_alos
            iqr_lower_alos = q1_alos - 1.5 * iqr_alos
            iqr_upper_alos = q3_alos + 1.5 * iqr_alos
            st.write('### Satker BLU yang merupakan outlier pada variabel ALOS')
            st.write('ALOS (Average Leng of Stay), idealnya 6-9 hari (Kemenkes, 2011)')
            st.write(data.loc[ (data.ALOS_hari < iqr_lower_alos) | (data.ALOS_hari > iqr_upper_alos),['Kode_satker','Tahun','Nama_satker','ALOS_hari']].sort_values(by='ALOS_hari', ascending=False))
        if radio_out == 'BTO':
            data.BTO_kali.plot.box()
            st.pyplot()
            bto = data['BTO_kali']
            q1_bto = bto.quantile(0.25)
            q3_bto = bto.quantile(0.75)

            iqr_bto = q3_bto - q1_bto
            iqr_lower_bto = q1_bto - 1.5 * iqr_bto
            iqr_upper_bto = q3_bto + 1.5 * iqr_bto
            st.write('### Satker BLU yang merupakan outlier pada variabel BTO')
            st.write('BTO (Bed Turn Over), idealnya 40-50 kali (Kemenkes, 2011)')
            st.write(data.loc[ (data.BTO_kali < iqr_lower_bto) | (data.BTO_kali> iqr_upper_bto),['Kode_satker','Tahun','Nama_satker','BTO_kali']].sort_values(by='BTO_kali', ascending=False))
        if radio_out == 'TOI':
            data.TOI_hari.plot.box()
            st.pyplot()
            toi = data['TOI_hari']
            q1_toi = toi.quantile(0.25)
            q3_toi = toi.quantile(0.75)

            iqr_toi = q3_toi - q1_toi
            iqr_lower_toi = q1_toi - 1.5 * iqr_toi
            iqr_upper_toi = q3_toi + 1.5 * iqr_toi
            st.write('### Satker BLU yang merupakan outlier pada variabel TOI')
            st.write('TOI (Turn Over Interval), idealnya 1-3 hari (Kemenkes, 2011)')
            st.write(data.loc[ (data.TOI_hari < iqr_lower_toi) | (data.TOI_hari> iqr_upper_toi),['Kode_satker','Tahun','Nama_satker','TOI_hari']].sort_values(by='TOI_hari', ascending=False))
    #Menu DEA Analysis
    # if selectbox_dea == 'DEA Analysis':
    #     dea_select = st.sidebar.radio('Pilih Variabel',('VRS','CRS'))
    #     tahun_select = st.sidebar.radio('Pilih Tahun',('2016','2017','2018','2019','2020'))
    #     if tahun_select == '2016':
    #         data_2016 = pd.read_excel('./dataset/dea_2016.xlsx')
    #         st.write(data_2016.head())
    #         if dea_select == 'VRS':
    #             inputs = input(data_2016)
    #             outputs = output(data_2016)
    #             uni_prob = dea.DEAProblem(inputs, outputs, returns='VRS')
    #             myresults = uni_prob.solve()
    #             st.write(myresults['Weights'])
    #         if dea_select == 'CRS':
    #             inputs = input(data_2016)
    #             outputs = output(data_2016)
    #             uni_prob = dea.DEAProblem(inputs, outputs, returns='CRS')
    #             myresults = uni_prob.solve()
    #             st.write(myresults['Weights'])
    #     if tahun_select == '2017':
    #         data_2017 = pd.read_excel('./dataset/dea_2017.xlsx')
    #         st.write(data_2017.head())
    #         if dea_select == 'VRS':
    #             inputs = input(data_2017)
    #             outputs = output(data_2017)
    #             uni_prob = dea.DEAProblem(inputs, outputs, returns='VRS')
    #             myresults = uni_prob.solve()
    #             st.write(myresults['Weights'])
    #         if dea_select == 'CRS':
    #             inputs = input(data_2017)
    #             outputs = output(data_2017)
    #             uni_prob = dea.DEAProblem(inputs, outputs, returns='CRS')
    #             myresults = uni_prob.solve()
    #             st.write(myresults['Weights'])
    #     if tahun_select == '2018':
    #         data_2018 = pd.read_excel('./dataset/dea_2018.xlsx')
    #         st.write(data_2018.head())
    #         if dea_select == 'VRS':
    #             inputs = input(data_2018)
    #             outputs = output(data_2018)
    #             uni_prob = dea.DEAProblem(inputs, outputs, returns='VRS')
    #             myresults = uni_prob.solve()
    #             st.write(myresults['Weights'])
    #         if dea_select == 'CRS':
    #             inputs = input(data_2018)
    #             outputs = output(data_2018)
    #             uni_prob = dea.DEAProblem(inputs, outputs, returns='CRS')
    #             myresults = uni_prob.solve()
    #             st.write(myresults['Weights'])
    #     if tahun_select == '2019':
    #         data_2019 = pd.read_excel('./dataset/dea_2019.xlsx')
    #         st.write(data_2019.head())
    #         if dea_select == 'VRS':
    #             inputs = input(data_2019)
    #             outputs = output(data_2019)
    #             uni_prob = dea.DEAProblem(inputs, outputs, returns='VRS')
    #             myresults = uni_prob.solve()
    #             st.write(myresults['Weights'])
    #         if dea_select == 'CRS':
    #             inputs = input(data_2019)
    #             outputs = output(data_2019)
    #             uni_prob = dea.DEAProblem(inputs, outputs, returns='CRS')
    #             myresults = uni_prob.solve()
    #             st.write(myresults['Weights'])
    #     if tahun_select == '2020':
    #         data_2020 = pd.read_excel('./dataset/dea_2020.xlsx')
    #         st.write(data_2020.head())
    #         if dea_select == 'VRS':
    #             inputs = input(data_2020)
    #             outputs = output(data_2020)
    #             uni_prob = dea.DEAProblem(inputs, outputs, returns='VRS')
    #             myresults = uni_prob.solve()
    #             st.write(myresults['Weights'])
    #         if dea_select == 'CRS':
    #             inputs = input(data_2020)
    #             outputs = output(data_2020)
    #             uni_prob = dea.DEAProblem(inputs, outputs, returns='CRS')
    #             myresults = uni_prob.solve()
    #             st.write(myresults['Weights'])
    if selectbox_dea == 'DEA Solver':
        st.write("## Hasil Analisis DEA Solver")
        pilih_tahun = st.sidebar.selectbox('Pilih tahun',('2016','2017','2018','2019','2020'))
        if pilih_tahun == '2016':
            st.write('### Data Tahun 2016')
            st.write('#### Rekap Data Envelopment Analysis')
            st.write(tempat_tidur.head())
if menu == 'Unggah/Unduh Data':
    radio_unggah = st.sidebar.radio('Pilih Menu',('Unggah Data','Unduh Data'))
    if radio_unggah == 'Unggah Data':
        st.write('## Unggah Data')
    if radio_unggah == 'Unduh Data':
        st.write('## Unduh Data')
if menu == 'Prediction':
    st.write('Hello Prediction')
    st.write('## On Progress..........')