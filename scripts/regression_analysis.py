import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import *
import scipy as sc
from scipy.stats import t,norm
import xarray as xr
from scipy import stats,signal

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mplc
import matplotlib as mpl

import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter



def detrend(time_series):
    time_series = time_series
    y            =    np.squeeze(np.asarray(time_series))
    if len(y.shape) !=1: 
        print("ERROR : The input must be 1D array")
        
    else:
        nobs         =    y.shape[0]
        x            =    np.arange(1,nobs+1)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        qhat         =    m*x+c
        detrended   =     y - qhat
           
    
    return detrended,m,c
    
from scipy.stats import norm


def mk_test(x, alpha=0.05):
    """
    This code is written by @author: Michael Schramm
    This function is derived from code originally posted by Sat Kumar Tomer
    (satkumartomer@gmail.com)
    See also: http://vsp.pnnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm
    The purpose of the Mann-Kendall (MK) test (Mann 1945, Kendall 1975, Gilbert
    1987) is to statistically assess if there is a monotonic upward or downward
    trend of the variable of interest over time. A monotonic upward (downward)
    trend means that the variable consistently increases (decreases) through
    time, but the trend may or may not be linear. The MK test can be used in
    place of a parametric linear regression analysis, which can be used to test
    if the slope of the estimated linear regression line is different from
    zero. The regression analysis requires that the residuals from the fitted
    regression line be normally distributed; an assumption not required by the
    MK test, that is, the MK test is a non-parametric (distribution-free) test.
    Hirsch, Slack and Smith (1982, page 107) indicate that the MK test is best
    viewed as an exploratory analysis and is most appropriately used to
    identify stations where changes are significant or of large magnitude and
    to quantify these findings.
    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)
    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics
    Examples
    --------
      >>> x = np.random.rand(100)
      >>> trend,h,p,z = mk_test(x,0.05)
    """
    n = len(x)

    # calculate S
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else:  # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(x == unique_x[i])
        var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    else: # s == 0:
        z = 0

    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1-alpha/2)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return trend, h, p, z


def linear_regress(xx,yy,alpha,opt_detrend,opt_mktest):
    xx                    =    np.squeeze(np.asarray(xx))
    yy                    =    np.squeeze(np.asarray(yy))

    if len(yy)!=len(xx):
        print("ERROR : Length of the arrays mismatch !")
    else:
        N                     =         len(xx)
        if opt_detrend:
            x,m1,c1               =         detrend(xx)
            y,m2,c2               =         detrend(yy)
        else:
            x=xx*1
            y=yy*1
        #print(x,y)
        corr                  =         np.corrcoef(x,y)[0,1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y,rcond=None)[0]
        qhat                  =          m*x+c
        error                 =         (y-qhat)**2
        var_residuals         =         (1/(N-2))*(np.sum(error))
        x_er                  =         np.sum((x-np.mean(x))**2)
        s                     =         var_residuals/x_er
        standard_error        =         (s)**0.5
        t_score               =         np.absolute(m)/standard_error
        # Student t-distribution Percent Point Function
        # define probability
        if opt_mktest:
            trend, h, p1, z  = mk_test(y, alpha)
            h1 =h*1.0  
        else:
            #p = 0.025
            df                    =         N-2
            # retrieve value <= probability
            t_critical            =         t.ppf(1-alpha/2, df)
            # confirm with cdf
            p1                    =        1.- t.cdf(t_score, df)
            if p1< alpha/2:
                h1 = 1.0
            else:
                h1= 0.0
                
    return m,c,p1,corr,h1



def linear_regress_scipy(xx,yy,alpha,opt_detrend,opt_mktest):
    xx                    =    np.squeeze(np.asarray(xx))
    yy                    =    np.squeeze(np.asarray(yy))

    if len(yy)!=len(xx):
        print("ERROR : Length of the arrays mismatch !")
    else:
        N                     =         len(xx)
        if opt_detrend:
            x              =        signal.detrend(xx)
            y               =       signal.detrend(yy)
        else:
            x=xx*1
            y=yy*1
        #print(x,y)
        m,c,corr,p1,serr = stats.linregress(x,y)
        
        if opt_mktest:
            trend, h, p1, z  = mk_test(yy, alpha)
            h1 =h*1.0  
        else:
            if p1< alpha:
                h1 = 1.0
            else:
                h1= 0.0
                
    return m,c,p1,corr,h1


def write_to_netcdf(data_map,output_filename,ex_filename='',varname_ex=''):
    ## for 3D file structure
    d         =   data_map
    ds1       =   xr.open_dataset(ex_filename)
    times     =   ds1.time
    lon       =   ds1.lon
    lat       =   ds1.lat
    dk2       =   ds1

    t         =   xr.DataArray(d,coords=[('time', times),('lat', lat),('lon', lon)])
    dk2[varname_ex]=t
    print (dk2.coords)
    print ('finished saving')
    dk2.to_netcdf(output_filename)

    return  print("Thank you")

def draw_map(data_name,varname,vmin,vmax,inc,titlestr,axiom,cmap='RdBu',hatch='/',draw_par=1):
    ds1           =     xr.open_dataset(data_name)
    data          =     ds1[varname].values
    lon           =     ds1.lon
    lat           =     ds1.lat

    # m = Basemap(projection='ortho',lat_0=0,lon_0=-180,resolution='l')
    #m = Basemap(projection='moll',lon_0=0,lat_0=0,resolution='l')  

    m              = Basemap(projection='mill',lat_ts=10,llcrnrlon=lon.min(),urcrnrlon
                             =lon.max(),llcrnrlat=lat.min(),urcrnrlat=lat.max(),resolution='c',ax=axiom)
    lon2, lat2     =    np.meshgrid(lon,lat)
    x, y           =    m(lon2, lat2)
    #m.fillcontinents(color='gray',lake_color='gray')

    m.drawcoastlines()
    
    if draw_par:
        m.drawparallels(np.arange(-80.,81.,20.),labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180.,181.,60.),labels=[0,0,0,1])
        
    norm      =    mpl.colors.Normalize(vmin,vmax)
    v         =    np.arange(vmin,vmax+inc,inc)
    cs        =    m.contourf(x,y,data[0,:,:],v,norm=norm,extend='both',cmap=plt.cm.get_cmap(cmap))
    axiom.set_title(titlestr)
    cbar0 = plt.colorbar(cs,ax=axiom,orientation='horizontal',fraction=0.05)

    
def draw_map_cartopy(data_name,varname,vmin,vmax,inc,titlestr,s1,s2,sub_no,cmap='RdBu',hatch='/',draw_par=1):
    ds1           =     xr.open_dataset(data_name)
    data          =     ds1[varname].values
    lon           =     ds1.lon
    lat           =     ds1.lat

    axiom = plt.subplot(s1,s2,sub_no,projection=ccrs.PlateCarree(central_longitude=180.0))
    v         =    np.arange(vmin,vmax+inc,inc)

    cs=axiom.contourf(lon, lat, data[0,:,:],v, cmap=cmap,extend='both',transform = ccrs.PlateCarree())
    axiom.coastlines()
    if draw_par:
        axiom.gridlines()

    cbar =  plt.colorbar(cs, shrink=0.5, orientation='horizontal',extend='both')
    axiom.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
    axiom.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    axiom.xaxis.set_major_formatter(lon_formatter)
    axiom.yaxis.set_major_formatter(lat_formatter)
    axiom.set_title(titlestr)



class reg_plot(): 
    
    """Plotting the regression map with significance """
    """Plotting the correlation map with significance"""
   
    
    def __init__(self,time_series=[],data_name='',varname_data='',alpha=0,opt_detrend=0,opt_mktest=0):
        self.time_series  = time_series 
        self.data_name    = data_name
        self.varname_data = varname_data
        self.alpha            = alpha
        self.opt_detrend  = opt_detrend
        self.opt_mktest  = opt_mktest

    def explain_to(self):
        print("Hello, users. These are inputs:")
        print("Your time series is {}.".format(self.time_series))
        print("Your data filename is {}.".format(self.data_name))
        print("Your variable name is {}.".format(self.varname_data))
        print("significance level is {}.".format(self.alpha))
        

    
    def regression_map_making(self):

        data_f     =      xr.open_dataset(self.data_name)
        data       =      data_f[self.varname_data].values
        shape      =      [1,data.shape[1],data.shape[2]]
        regress_map=      np.zeros((shape))
        cor_map    =      np.zeros((shape))
        significant_map=  np.zeros((shape))
     
        for i in range(data.shape[2]):
            for j in range(data.shape[1]): 
                temp              = data[:,j,i]
                if np.all(np.isnan(temp)):
                    regress_map[0,j,i]= np.nan
                    cor_map[0,j,i]    = np.nan
                else:
                    slope,intercept,p1,corr,h1=linear_regress_scipy(self.time_series,temp,self.alpha,self.opt_detrend,self.opt_mktest)
                    regress_map[0,j,i]= slope
                    cor_map[0,j,i]    = corr
                    significant_map[0,j,i] = h1
                       
        
        return regress_map,cor_map,significant_map    

    
    def draw_regression(self,vmin,vmax,inc,titlestr,axiom,cmap='RdBu',hatch='/',draw_par=1):
        regress_map,cor_map,significant_map= self.regression_map_making()
        ds1           =     xr.open_dataset(self.data_name)
        lon           =     ds1.lon
        lat           =     ds1.lat

        # m = Basemap(projection='ortho',lat_0=0,lon_0=-180,resolution='l')
        #m = Basemap(projection='moll',lon_0=0,lat_0=0,resolution='l')  

        m              = Basemap(projection='mill',lat_ts=10,llcrnrlon=lon.min(),urcrnrlon =lon.max(),llcrnrlat=lat.min(),urcrnrlat=lat.max(),resolution='c',ax=axiom)
        lon2, lat2     =    np.meshgrid(lon,lat)
        x, y           =    m(lon2, lat2)

        #fig = plt.figure()
        #m.fillcontinents(color='gray',lake_color='gray')

        m.drawcoastlines()
        if draw_par:
            m.drawparallels(np.arange(-80.,81.,20.),labels=[1,0,0,0])
            m.drawmeridians(np.arange(-180.,181.,60.),labels=[0,0,0,1])

        #m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0])
        #m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
        # m.drawmapboundary(fill_color='white')
        norm      =    mpl.colors.Normalize(vmin,vmax)
        v         =    np.arange(vmin,vmax+inc,inc)
        cs        =    m.contourf(x,y,regress_map[0,:,:],v,norm=norm,extend='both',cmap=plt.cm.get_cmap(cmap))
        #levels=[0,1]
        zm = np.ma.masked_equal(significant_map[0,:,:], 0)
        m.contourf(x,y,zm, hatches=hatch,alpha=0.)
        cbar0     = plt.colorbar(cs,ax=axiom,orientation='horizontal',fraction=0.05)
        axiom.set_title(titlestr)

    def draw_regression_cartopy(self,vmin,vmax,inc,titlestr,s1,s2,sub_no,cmap='RdBu',hatch='/',draw_par=1):
        regress_map,cor_map,significant_map= self.regression_map_making()
        ds1           =     xr.open_dataset(self.data_name)
        lon           =     ds1.lon
        lat           =     ds1.lat

        axiom = plt.subplot(s1,s2,sub_no,projection=ccrs.PlateCarree(central_longitude=180.0))
        axiom.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
        axiom.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        axiom.xaxis.set_major_formatter(lon_formatter)
        axiom.yaxis.set_major_formatter(lat_formatter)

        v         =    np.arange(vmin,vmax+inc,inc)
        
        cs=axiom.contourf(lon, lat, regress_map[0,:,:],v, cmap=cmap,extend='both',transform = ccrs.PlateCarree())
        axiom.contourf(lon,lat,significant_map[0,:,:], levels=[0,0.5,1],colors='none',hatches=[None,hatch,],transform = ccrs.PlateCarree())
        axiom.coastlines()
        
        if draw_par:
            axiom.gridlines()

        cbar =  plt.colorbar(cs, fraction=0.05, orientation='horizontal',extend='both')
        axiom.set_title(titlestr)


    
    
    def draw_correlation(self,significant_value,vmin,vmax,inc,titlestr,axiom,cmap='RdBu',hatch='/',draw_par=1):
        regress_map,cor_map,significant_map=self.regression_map_making()
        ds1=xr.open_dataset(self.data_name)
        lon=ds1.lon
        lat=ds1.lat
        # m = Basemap(projection='ortho',lat_0=0,lon_0=-180,resolution='l')
        #m = Basemap(projection='moll',lon_0=0,lat_0=0,resolution='l')  
        m = Basemap(projection='mill',lat_ts=10,llcrnrlon=lon.min(),         urcrnrlon =    lon.max(),llcrnrlat=lat.min(),urcrnrlat=lat.max(), resolution='c',ax=axiom)
        lon2, lat2 = np.meshgrid(lon,lat)
        x, y       = m(lon2, lat2)
        fig        = plt.figure(figsize=(10,7))
        #fig = plt.figure()
        #m.fillcontinents(color='gray',lake_color='gray')
        m.drawcoastlines()
        if draw_par:
            m.drawparallels(np.arange(-80.,81.,20.),labels=[1,0,0,0])
            m.drawmeridians(np.arange(-180.,181.,60.),labels=[0,0,0,1])
        #m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0])
        #m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
        # m.drawmapboundary(fill_color='white')
        norm = mpl.colors.Normalize(vmin,vmax)
        v=np.arange(vmin,vmax+inc,inc)
        cs = m.contourf(x,y,cor_map[0,:,:],v,norm=norm,extend='both',cmap=plt.cm.get_cmap(cmap))
        #levels=[0,1]
        #m.contourf(x,y,sig[0,:,:], 2, hatches=["", "/"],alpha=0)
        m.contour(x, y, cor_map[0,:,:], levels=[-1*significant_value,significant_value], linewidths=0.5, colors='black', antialiased=True)
        cbar0 = plt.colorbar(cs,ax=axiom,orientation='horizontal',fraction=0.05)
        axiom.set_title(titlestr) 
       
        
    def draw_correlation_cartopy(self,significant_value,vmin,vmax,inc,titlestr,s1,s2,sub_no,cmap='RdBu',draw_par=1):
        regress_map,cor_map,significant_map= self.regression_map_making()
        ds1           =     xr.open_dataset(self.data_name)
        lon           =     ds1.lon
        lat           =     ds1.lat

        axiom = plt.subplot(s1,s2,sub_no,projection=ccrs.PlateCarree(central_longitude=180.0))
        axiom.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
        axiom.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        axiom.xaxis.set_major_formatter(lon_formatter)
        axiom.yaxis.set_major_formatter(lat_formatter)

        v         =    np.arange(vmin,vmax+inc,inc)
        
        cs=axiom.contourf(lon, lat, cor_map[0,:,:],v, cmap=cmap,extend='both',transform = ccrs.PlateCarree())
        axiom.contour(lon,lat,cor_map[0,:,:],levels=[-1*significant_value,significant_value], linewidths=0.5, colors='black', antialiased=True,transform = ccrs.PlateCarree())
        axiom.coastlines()
        
        if draw_par:
            axiom.gridlines()

        cbar =  plt.colorbar(cs, fraction=0.05, orientation='horizontal',extend='both')
        axiom.set_title(titlestr)
        
   
      
