import os
import sys
import glob
import numpy as np
import pygmt as gmt
import xarray as xr
from PIL import Image
from grace_time import doy2date
import matplotlib.pyplot as plt


def gmt_show_mascon(data, lon, lat, directory_path, image_name, time):
    fig = gmt.Figure()
    gmt.config(MAP_GRID_PEN_PRIMARY="0.01p,gray50")
    z_lim, z_step = 20, 5
    # 颜色映射
    gmt.makecpt(cmap="jet", series=[-z_lim, z_lim, z_step], continuous=True, background=True)
    # 地图框架设置
    with gmt.config(FONT="8p", MAP_TICK_LENGTH_PRIMARY="0c", MAP_FRAME_PEN='0.5p'):
        fig.basemap(region="d", projection="N10c", frame=["WsNe", "xa90", "ya90"])

    # Load data
    with gmt.config(FONT_TITLE='8p,Helvetica,gray9', MAP_TITLE_OFFSET="4p"):
        fig.plot(x=lon, y=lat, style="h0.07c", fill=data, cmap=True, frame=f"+tEWH (cm) in {time}")

    fig.basemap(frame="g60")
    # Add shorelines
    fig.coast(shorelines="1/0.5p,gray3")
    # Add colorbar
    with gmt.config(FONT="8p", MAP_FRAME_PEN="0.01c"):
        fig.colorbar(
            frame=[f"xa{z_step}f{z_step}"],
            position="JBC+o0c/0.4c+h+w5c/0.3c+ml"
        )
    # justify the existence of output directory
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if not image_name.endswith(".jpg"):
        image_name += '.jpg'
    fig_name = os.path.join(directory_path, image_name)
    fig.savefig(fname=fig_name, dpi=500)

    print(fig_name)


def show_mask(mask_path, image_path, image_name):
    RMS_mask_dataset = xr.open_dataset(mask_path)
    RMS_mask = RMS_mask_dataset['RMS_mask'].values
    lat = RMS_mask_dataset['lat'].values
    lon = RMS_mask_dataset['lon'].values
    # 绘制掩膜
    fig = gmt.Figure()
    gmt.config(MAP_GRID_PEN_PRIMARY="0.01p,gray50")
    z_step = 0.5
    # 颜色映射
    gmt.makecpt(cmap="jet", series=[-2, 2, z_step], continuous=True, background=True)
    # 地图框架设置
    with gmt.config(FONT="8p", MAP_TICK_LENGTH_PRIMARY="0c", MAP_FRAME_PEN='0.5p'):
        fig.basemap(region="d", projection="N10c", frame=["WsNe", "xa90", "ya90"])

    # Load data
    with gmt.config(FONT_TITLE='8p,Helvetica,gray9', MAP_TITLE_OFFSET="4p"):
        fig.plot(x=lon, y=lat, style="h0.07c", fill=RMS_mask, cmap=True, frame=f"+tRMS mask")

    fig.basemap(frame="g60")
    # Add shorelines
    fig.coast(shorelines="1/0.1p,gray30")
    # Add colorbar
    with gmt.config(FONT="8p", MAP_FRAME_PEN="0.01c"):
        fig.colorbar(
            frame=[f"xa{z_step}f{z_step}"],
            position="JBC+o0c/0.4c+h+w5c/0.3c+ml"
        )
    # justify the existence of output directory
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not image_name.endswith(".jpg"):
        image_name += '.jpg'
    fig_name = os.path.join(image_path, image_name)
    fig.savefig(fname=fig_name, dpi=500)

    print(fig_name)


def show_mask_Antarctica(mask_path, image_path, image_name):
    RMS_mask_dataset = xr.open_dataset(mask_path)
    RMS_mask = RMS_mask_dataset['RMS_mask'].values
    lat = RMS_mask_dataset['lat'].values
    lon = RMS_mask_dataset['lon'].values
    # 绘制掩膜
    fig = gmt.Figure()
    gmt.config(MAP_GRID_PEN_PRIMARY="0.01p,gray50")
    z_step = 0.5
    # 颜色映射
    gmt.makecpt(cmap="jet", series=[-2, 2, z_step], continuous=True, background=True)
    # 地图框架设置
    with gmt.config(FONT="8p", MAP_TICK_LENGTH_PRIMARY="0c", MAP_FRAME_PEN='0.5p'):
        fig.basemap(region='g', projection="E0.33/-89.6/60/8c", frame=True)

    z_lim, z_step = 2, 1
    # # 颜色映射
    gmt.makecpt(cmap="jet", series=[-z_lim, z_lim, z_step], continuous=True, background=True)
    # # 地图框架设置
    # # Load data
    fig.plot(x=lon, y=lat, style="h0.07c", fill=RMS_mask, cmap=True, frame=f"+tRMS mask")

    # Add shorelines
    fig.coast(shorelines="1/0.5p,grey30")
    # Add colorbar
    with gmt.config(FONT="8p", MAP_FRAME_PEN="0.01c"):
        fig.colorbar(
            frame=[f"xa{z_step}f{z_step}"],
            position="JBC+o0c/0.4c+h+w5c/0.3c+ml"
        )
    # justify the existence of output directory
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not image_name.endswith(".jpg"):
        image_name += '.jpg'
    fig_name = os.path.join(image_path, image_name)
    fig.savefig(fname=fig_name, dpi=500)

    print(fig_name)


def show_Rmatrix(file_path, directory_path, image_name):
    dataset = xr.open_dataset(file_path)
    Rmatrix = dataset['Rmatrix'].values
    lat = dataset['lat'].values
    lon = dataset['lon'].values
    fig = gmt.Figure()
    gmt.config(MAP_GRID_PEN_PRIMARY="0.01p,gray50")
    z_step = 2
    # 颜色映射
    gmt.makecpt(cmap="jet", series=[0, 15, z_step], continuous=True, background=True)
    # 地图框架设置
    with gmt.config(FONT="8p", MAP_TICK_LENGTH_PRIMARY="0c", MAP_FRAME_PEN='0.5p'):
        fig.basemap(region="d", projection="N10c", frame=["WsNe", "xa90", "ya90"])

    # Load data
    with gmt.config(FONT_TITLE='8p,Helvetica,gray9', MAP_TITLE_OFFSET="4p"):
        fig.plot(x=lon, y=lat, style="h0.07c", fill=1 / np.sqrt(Rmatrix), cmap=True, frame=f"+tregularization matrix")

    fig.basemap(frame="g60")
    # Add shorelines
    fig.coast(shorelines="1/0.5p,gray3")
    # Add colorbar
    with gmt.config(FONT="8p", MAP_FRAME_PEN="0.01c"):
        fig.colorbar(
            frame=[f"xa{z_step}f{z_step}"],
            position="JBC+o0c/0.4c+h+w5c/0.3c+ml"
        )
    # justify the existence of output directory
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if not image_name.endswith(".jpg"):
        image_name += '.jpg'
    fig_name = os.path.join(directory_path, image_name)
    fig.savefig(fname=fig_name, dpi=500)

    print(fig_name)


def check_mascon(mascon_path, location_path, image_path):
    if os.path.exists(mascon_path):
        files = os.listdir(mascon_path)
        file_list = [file for file in files if file.endswith(".nc")]
        file_path = os.path.join(mascon_path, file_list[0])
        mascon_dataset = xr.open_dataset(file_path)
        # 获取网格地理位置
        # grid_location = xr.open_dataset(location_path)
        # lon = grid_location['lon'].values
        # lat = grid_location['lat'].values
        mascon = mascon_dataset['lwe_thickness'].values
        lon = mascon_dataset['lon'].values
        lat = mascon_dataset['lat'].values
        time = mascon_dataset['time'].values
        dates = doy2date(time)
        num_month = len(mascon)
        for i in range(1):
            vdata = mascon[i, :]
            vdate = dates[i]
            image_name = f'TWSA_in_{vdate}.jpg'
            gmt_show_mascon(vdata.T, lon, lat, image_path, image_name, vdate)
    else:
        sys.exit('未生成mascon数据')


def outputGif(image_path, mascon_gif_path):
    '''
    将多个月的等效水高输出为gif
    path：输出结果的根目录
    lwe: 上一步的格网计算结果，可以是TWS，GWS和DSI
    region：展示区域
    name：与TWS，GWS和DSI一致的名称，用于gif中的标题
    '''

    # 获取临时文件夹中的所有图片
    jpeg_files = sorted(glob.glob(os.path.join(image_path, '*.jpg')))
    frames = []
    # 遍历每个 JPEG 图像并创建帧
    for jpeg_file in jpeg_files:
        img = Image.open(jpeg_file)
        frames.append(img)
    if not os.path.exists(mascon_gif_path):
        os.makedirs(mascon_gif_path)
    output_gif = os.path.join(mascon_gif_path, f'between 2003-03 and 2014-03.gif')
    # 保存 GIF 动画
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=500, loop=0)
    print(f'已将各月masccon结果输出至{output_gif}文件')


def mEWH_series_plot(data_rsds, seriesname, time, imgpath):
    print('绘制平均等效水高的时间序列')
    # 设置颜色
    colors = ['#5b9bd5', '#ed7d31', '#70ad47', '#ffc000', '#4472c4', '#91d024', '#b235e6', '#02ae75', '#009db2',
              '#22559C', '#0780cf', '#f8cb7f', '#63b2ee']
    # 绘制噪声序列
    fig, ax = plt.subplots(figsize=(10 / 2.5, 8 / 2.5))
    for t, data, name, color in zip(time, data_rsds, seriesname, colors):
        ax.plot(t, data, label=name, color=color, lw=1)
    # ax.plot(time, data_rsds, label=seriesname, color=colors[0], lw=1)
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    # 添加网格
    plt.grid(True, axis='both', c='black', alpha=0.2, linestyle='--')
    # 设置坐标轴和图例
    fontstyle = 'Times New Roman'
    ax.tick_params(axis='y', right=True, left=True, direction='in', length=2, width=0.5, labelsize=6)
    ax.tick_params(axis='x', bottom=True, top=True, direction='in', length=2, width=0.5, labelsize=6)
    plt.xlabel('Year', fontdict={'family': fontstyle, 'size': 6})
    plt.ylabel('wEWH[cm]', fontdict={'family': fontstyle, 'size': 6})
    plt.xticks(fontname=fontstyle)
    plt.yticks(fontname=fontstyle)
    # 设置图例
    plt.legend(loc='lower left',
               bbox_to_anchor=(0, 1.02, 1, 0.102),
               mode='expand',
               borderaxespad=0.1,
               ncols=3,
               frameon=True,
               prop={'family': 'Times New Roman', 'size': 6})
    plt.tight_layout(pad=0)
    # 设置坐标轴边框
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)

    # 设置图像
    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9, right=0.9)

    plt.tight_layout()
    # 判断存储图像的文件夹是否存在，不存在则创建一个
    imgpath, imgname = os.path.split(imgpath)
    if not os.path.exists(imgpath):
        print('图像存储文件夹不存在')
        os.makedirs(imgpath)
    if not imgname.endswith('.jpg'):
        imgname += '.jpg'
    imgpath = os.path.join(imgpath, imgname)
    plt.savefig(imgpath, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f'区域平均等效水高序列输出为{imgpath}')

def figure_one(vdata,lon,lat,image_path, image_name):
    # 进行绘制
    fig = gmt.Figure()
    gmt.config(MAP_GRID_PEN_PRIMARY="0.01p,gray50")
    z_step = 0.5
    # 颜色映射
    gmt.makecpt(cmap="jet", series=[-2, 2, z_step], continuous=True, background=True)
    # 地图框架设置
    with gmt.config(FONT="8p", MAP_TICK_LENGTH_PRIMARY="0c", MAP_FRAME_PEN='0.5p'):
        fig.basemap(region=[-78, -10, 59, 84], projection="L-45/72/70/78/8c", frame='afg')

    z_lim, z_step = 20, 5
    # # 颜色映射
    gmt.makecpt(cmap="jet", series=[-z_lim, z_lim, z_step], continuous=True, background=True)
    # # 地图框架设置
    # # Load data
    fig.plot(x=lon, y=lat, style="c0.28c", fill=vdata, cmap=True, frame=f"+t{image_name}")

    # Add shorelines
    fig.coast(shorelines="1/0.5p,grey30")
    # Add colorbar
    with gmt.config(FONT="8p", MAP_FRAME_PEN="0.01c"):
        fig.colorbar(
            frame=[f"xa{z_step}f{z_step}"],
            position="JBC+o0c/0.4c+h+w5c/0.3c+ml"
        )
    # justify the existence of output directory
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not image_name.endswith(".jpg"):
        image_name += '.jpg'
    fig_name = os.path.join(image_path, image_name)
    fig.savefig(fname=fig_name, dpi=500)

    print(fig_name)

def show_Greenland(mascon_dataset, image_path):
    # 获取网格地理位置
    mascon = mascon_dataset['lwe_thickness'].values
    lon = mascon_dataset['lon'].values
    lat = mascon_dataset['lat'].values
    time = mascon_dataset['time'].values
    dates = doy2date(time)
    num_month = len(mascon)
    if np.ndim(mascon)==3:
        for i in range(num_month):
            vdata = mascon[i, 9, :]
            vdate = dates[i]
            image_name = f'TWSA_in_{vdate}_Greenland'
            figure_one(vdata, lon, lat, image_path, image_name)

