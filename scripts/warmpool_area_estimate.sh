# This shell script uses CDO, NCO (operator ncwa) and GMT (operator grdvolume) 
# and displays the the warmpool area for each month during Nov–April, 1980–2018.
for i in $(seq 1900 2018)
do
        for j in $(seq 7 12)                                                    #Month numbers 7-12 here are November–April (shifted)
        do
        rm sst.nc sst_ncwa.nc
        cdo -s selyear,${i} -selmon,${j} sst_1900_2018_mjo_mon_remap.nc sst.nc  #CDO command to extract each month to sst.nc
        ncwa -A -v lon,lat,SST -a TIME,TIME_bnds,lev sst.nc sst_ncwa.nc         #NCO command to strip off extra dimensions from sst.nc
        gmt grdvolume sst_ncwa.nc -Vquiet -C28 -R40/220/-25/25 -Sk              #GMT command which estimates the area for SST>28°C
        done
done
