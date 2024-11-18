def trend_plot_vector(data, lon, lat, title, vmin, vmax, step, savepath, blnpath=r'D:\ill_pose\CSRMASCON\data\bln',
                      basin_id=0):
    # lat += 3
    # lon += 3
    # 读取边界文件,对数据进行掩膜
    if basin_id != 0:
        blnname = select_region_from_file(basin_id, blnpath)
        s = 5
    else:
        blnname = None
        s = 0.5
    sns.set_context('paper')
    plt.style.use('seaborn-ticks')

    # 若是冰盖区域，单独设置投影
    proj = ccrs.Robinson()
    if basin_id == 3:
        proj = ccrs.SouthPolarStereo()
    elif basin_id == 19:
        proj = ccrs.Mercator()

    # 设置颜色条
    cmap0 = plt.get_cmap('jet_r')  # 设置网格点颜色
    cmap = cmap0.reversed()  # 反转颜色
    levels = LinearLocator(50).tick_values(vmin, vmax)  # 设置网格点颜色显示区间
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend='both')

    # 创建画布
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=500, subplot_kw=dict(projection=proj))

    # 设置绘图范围
    if basin_id == 3:
        leftlon, rightlon, lowerlat, upperlat = (-180, 180, -60, -90)  # -60， -90
        img_extent = [leftlon, rightlon, lowerlat, upperlat]
        for ax in axs.flat:
            ax.set_extent(img_extent, ccrs.PlateCarree())
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.ylocator = mticker.FixedLocator([-70, -80])
            #######以下为网格线的参数######
            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
    elif basin_id == 19:
        img_extent = [-85, -25, 55, 85]
    elif basin_id != 3 and basin_id != 19 and blnname is not None:
        max_longitude, min_longitude, max_latitude, min_latitude = utils.max_coordinate(blnname)
        max_x = int(max_longitude + 5)
        min_x = int(min_longitude - 5)
        max_y = int(max_latitude + 5)
        min_y = int(min_latitude - 5)
        img_extent = [min_x, max_x, min_y, max_y]
        # 设置坐标轴显示区间和间隔
        xticks = np.arange(min_x, max_x, 15)
        yticks = np.arange(min_y, max_y, 15)
    else:
        xticks = np.arange(-180, 181, 60)
        yticks = np.arange(-90, 91, 30)
    # 填充数据  s=s,
    i = 0
    for ax in axs.flat:
        sc = ax.scatter(lon, lat, c=data[0, :], marker='H', cmap=cmap, transform=ccrs.PlateCarree(),
                        norm=norm)
        # 设置标题
        ax.set_title(f'{title[i]}')
        i += 1
    # 设置绘图范围
    if basin_id != 3 and blnname is not None:
        for ax in axs.flat:
            ax.set_extent(img_extent, crs=ccrs.PlateCarree())
            ax.spines['geo'].set_linewidth(1)
            ax.set_xticks(xticks, crs=proj)
            ax.set_yticks(yticks, crs=proj)
    # 添加色标
    cbar = plt.colorbar(sc, ax=axs, orientation='horizontal', shrink=0.5, pad=0.15,
                        aspect=30, fraction=0.05, ticks=np.arange(vmin, vmax * 1.05, step), extend='both')
    cbar.ax.tick_params(which='minor', direction='in', length=2.5, width=0, color='k',
                        labelsize=11)
    cbar.ax.tick_params(which='major', direction='out', length=2.5, width=1, color='k',
                        labelsize=11)
    cbar.set_label('Equivalent Water Height (cm)', fontsize=11, labelpad=1)  # 色带图例

    # 利用已有shp文件绘制地理信息,如海岸线、河流
    # coastlines = gpd.read_file(r'D:\ill_pose\CSRMASCON\data\special_area\ne_110m_coastline.shp')
    # for ax in axs.flat:
    #     coastlines.plot(ax=ax,  color='k', transformers=ccrs.Geodetic())
    reader_coastline = shpreader.Reader(r'D:\ill_pose\CSRMASCON\data\special_area\ne_110m_coastline.shp')
    coastlines = reader_coastline.records()
    for coastline in coastlines:
        geometry = coastline.geometry
        if geometry.geom_type == "MultiLineString":
            for line in geometry:
                x, y = zip(*line.coords)
                for ax in axs.flat:
                    ax.plot(x, y, transform=ccrs.Geodetic(), color='k')  # Geodetic
        elif geometry.geom_type == "LineString":
            x, y = zip(*geometry.coords)
            for ax in axs.flat:
                ax.plot(x, y, transform=ccrs.Geodetic(), color='k')
    reader_rivers = shpreader.Reader(r'D:\ill_pose\CSRMASCON\data\special_area\ne_110m_rivers_lake_centerlines.shp')
    rivers = reader_rivers.records()
    for river in rivers:
        geometry = river.geometry
        if geometry.geom_type == "MultiLineString":
            for line in geometry:
                x, y = zip(*line.coords)
                for ax in axs.flat:
                    ax.plot(x, y, transform=ccrs.Geodetic(), color='k')
        elif geometry.geom_type == "LineString":
            x, y = zip(*geometry.coords)
            for ax in axs.flat:
                ax.plot(x, y, transform=ccrs.Geodetic(), color='k')

    # 保存至文件夹
    typename, _ = os.path.splitext(savepath)
    type = typename.split("\\")[-1]
    if blnname is not None:
        dirStr, _ = os.path.splitext(blnname)
        fname = dirStr.split("\\")[-1]
        IMG_NAME = f'{savepath}\\{fname}_{type}.jpg'
        fig.savefig(IMG_NAME, dpi=400, bbox_inches='tight')
    else:
        IMG_NAME = f'{savepath}\\worldwideEWH_{type}.jpg'
        fig.savefig(IMG_NAME, dpi=400, bbox_inches='tight')
    print(f'图片已保存为{IMG_NAME}')