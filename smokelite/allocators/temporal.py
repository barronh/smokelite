__all__ = ['Temporal']

import os
import numpy as np
import pandas as pd
import PseudoNetCDF as pnc


known_sectors = [
    'AG_WASTE_BURN', 'AGRICULTURE', 'AIR', 'AIR_CDS', 'AIR_CRS', 'AIR_LTO',
    'ENERGY', 'INDUSTRY', 'LS_BIOBURN', 'RESIDENTIAL', 'SHIPS', 'TRANSPORT'
]

_monnames = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]


def getlabel(cmt):
    label = cmt
    label = label.replace('AGRICULTURAL', 'AG')
    label = label.replace('LARGE_SCALE', 'LS')
    label = label.replace('BIOMASS_', 'BIO')
    label = label.replace('BURNING', 'BURN')
    return label


def hourfactor(hrdf, idx, tzf):
    """
    Return UTC hourfactor [0, inf) to increase or decrease default rate

    Arguments
    ---------
    idx : scalar
        index for the hourly

    Returns
    -------
    griddedfactor : array-like
    """
    lsthourwgt = hrdf.loc[
        idx, [f'hour{h}' for h in [24] + list(range(1, 25)) + [1]]
    ].values
    lsthourinstantwgt = np.convolve([.5, .5], lsthourwgt, mode='valid')
    lstwgt = lsthourinstantwgt / lsthourinstantwgt.mean()
    # print(lstwgt, flush=True)
    utcoff = tzf.variables['UTCOFFSET'][:]
    utcwgts = np.zeros((25,) + utcoff.shape[2:], dtype='f')
    for j in range(tzf.NROWS):
        for i in range(tzf.NCOLS):
            myoff = utcoff[0, 0, j, i].astype('i')
            # print(myoff, flush=True)
            utcwgt = np.roll(lstwgt, -myoff)
            # print(utcwgt, flush=True)
            utcwgts[:, j, i] = utcwgt
    return utcwgts


def monfactor(mondf, idx, tzf):
    """
    Return monthly factors [0, inf) to increase or decrease default rate

    Arguments
    ---------
    idx : scalar
        index for the hourly
    mon : str
        Month three letter name %b

    Returns
    -------
    griddedfactor : array-like
    """
    out = np.zeros((12,) + tzf.variables['UTCOFFSET'].shape[2:], dtype='f')
    monthfact = mondf.loc[idx, _monnames].values
    monthwgt = monthfact / monthfact.mean()
    out[:] = monthwgt[:, None, None]
    return out


def weekdayfactor(wkdf, idx, tzf):
    """
    Return UTC weekday [0, inf) to increase or decrease default rate

    Arguments
    ---------
    idx : scalar
        index for the hourly
    utcstartdate : date-like
        utc starting day for the 24-hour period

    Returns
    -------
    griddedfactor : array-like
    """

    _weekdays = 'Mon Tue Wed Thu Fri Sat Sun'.split()

    rawdayfacts = wkdf.loc[idx, _weekdays].values.repeat(24, 0)
    dayfacts = np.append(rawdayfacts, rawdayfacts[0])
    daywgts = dayfacts / dayfacts.mean()
    # print(utcdays.dt.strftime('%a'))

    utcoff = tzf.variables['UTCOFFSET'][:]
    utcwgts = np.zeros((25, 7) + utcoff.shape[2:], dtype='f')
    for j in range(tzf.NROWS):
        for i in range(tzf.NCOLS):
            myoff = utcoff[0, 0, j, i].astype('i')
            mydayfacts = np.roll(daywgts, -myoff)
            for dayi in range(7):
                utcwgts[:, dayi, j, i] = mydayfacts[dayi*24:(dayi+1)*24+1]

    return utcwgts


class Temporal:
    def __init__(
        self, root='.', diurnalpath=None, dayofweekpath=None, monthlypath=None,
        timezonepath=None, gdnam='108NHEMI2', griddescpath=None
    ):
        """
        Initial Temporal Allocation object

        Any paths that do not exist will need to be made using the associated
        make_<>file method (e.g., make_monthlyfile), which are separately
        documented.

        Arguments
        ---------
        root : str
            root for data for paths that are not provided
        diurnalpath : str
            path to IOAPI file with hourly allocations  for sectors (as
            variables) with shape TSTEP=25, LAY=7, ROW=NROWS, COL=NCOLS
        dayofweekpath : str
            path to IOAPI file with day of week allocations for sectors (as
            variables) with shape  TSTEP=7, LAY=1, ROW=NROWS, COL=NCOLS
        monthlypath : str
            path to IOAPI file with month of year allocations for sectors (as
            variables) with shape TSTEP=12, LAY=1, ROW=NROWS, COL=NCOLS
        timezonepath : str
            path to IOAPI file with variable UTCOFFSET with shape TSTEP=1,
            LAY=1, ROW=NROWS, COL=NCOLS
        gdnam : str
            only used if timezonepath does not exists
        griddescpath : str
            path to IOAPI GRIDDESC file

        """
        if timezonepath is None:
            timezonepath = os.path.join(root, f'tz.{gdnam}.IOAPI.nc')
        if diurnalpath is None:
            diurnalpath = os.path.join(root, f'diurnal.{gdnam}.IOAPI.nc')
        if dayofweekpath is None:
            dayofweekpath = os.path.join(root, f'dayofweek.{gdnam}.IOAPI.nc')
        if monthlypath is None:
            monthlypath = os.path.join(root, f'monthly.{gdnam}.IOAPI.nc')
        if griddescpath is None:
            griddescpath = os.path.join(root, 'GRIDDESC')

        self.gdnam = gdnam
        self.diurnalpath = diurnalpath
        self.dayofweekpath = dayofweekpath
        self.monthlypath = monthlypath
        self.timezonepath = timezonepath
        self.griddescpath = griddescpath
        self._timezonefile = None
        self._monthlyfile = None
        self._dayofweekfile = None
        self._diurnalfile = None

    @property
    def monthlyfile(self):
        if self._monthlyfile is None:
            self.make_monthlyfile()
        return self._monthlyfile

    @monthlyfile.setter
    def monthlyfile(self, monthlyfile):
        self._monthlyfile = monthlyfile

    @property
    def diurnalfile(self):
        if self._diurnalfile is None:
            self.make_diurnalfile()
        return self._diurnalfile

    @diurnalfile.setter
    def diurnalfile(self, diurnalfile):
        self._diurnalfile = diurnalfile

    @property
    def dayofweekfile(self):
        if self._dayofweekfile is None:
            self.make_dayofweekfile()
        return self._dayofweekfile

    @dayofweekfile.setter
    def dayofweekfile(self, dayofweekfile):
        self._dayofweekfile = dayofweekfile

    @property
    def timezonefile(self):
        if self._timezonefile is None:
            self.make_timezonefile()
        return self._timezonefile

    @timezonefile.setter
    def timezonefile(self, timezonefile):
        self._timezonefile = timezonefile

    def make_diurnalfile(self, propath=None, read_kwds=None):
        """
        Arguments
        ---------
        propath : str
            path to tpro file ATPRO_HOURLY file. Must have 26 columns:
            profile_id, hour1, ... hour24, comment
        read_kwds : dict or None
            If None, default read_kwds are (comment='#', index_col=0)

        Returns
        -------
            df : PseudoNetCDFFile
                IOAPI-like file with hourly allocations  for sectors (as
                variables) with shape TSTEP=25, LAY=7, ROW=NROWS, COL=NCOLS
        """
        if self._diurnalfile is not None:
            return self._diurnalfile
        elif os.path.exists(self.diurnalpath):
            self._diurnalfile = pnc.pncopen(self.diurnalpath, format='ioapi')
            return self._diurnalfile

        if propath is None:
            raise KeyError(
                f'propath required because {self.diurnalpath} not found'
            )

        print(
            f'{self.diurnalpath} not available; calculating from {propath}'
        )
        if read_kwds is None:
            read_kwds = dict(comment='#', index_col=0)

        hrdf = pd.read_csv(propath, **read_kwds)
        if 'comment' not in hrdf.columns:
            hrdf['comment'] = hrdf.index.values
        tzf = self.timezonefile

        hr_f = tzf.subset([])
        hr_f.createDimension('TSTEP', 25).setunlimited(True)
        hr_f.SDATE = 2020001
        hr_f.STIME = 0
        hr_f.TSTEP = 10000

        for hridx, hrrow in hrdf.iterrows():
            cmt = hrrow['comment']
            label = getlabel(cmt)
            print(label, cmt)
            hourvals = hourfactor(hrdf, hridx, tzf)
            hrvar = hr_f.createVariable(
                label, 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
                long_name=label, units='1',
                var_desc=(
                    f'Scale a {label} daily rate 25 instantaneous hourly rates'
                )
            )
            hrvar[:] = hourvals[:, None]

        hr_f.updatemeta()
        hr_f.updatetflag(overwrite=True)

        hr_f.FILEDESC = """
## NASA-like metadata
1, 2310
Unknown, Unknown
Unknown
EPA sector-based hourly profiles
Not Applicable
1, 1
2020, 01, 01, 2020, 01, 01
0
...
PI_CONTACT_INFO: unknown@unknown.com
PLATFORM: CMAQ Emission processing input
DATA_INFO:  All data in time fractions
UNCERTAINTY:  large, preliminary data based on US averages.
DM_CONTACT_INFO: Unknown, Unknown, Unknown, unknown@unknown.com
PROJECT_INFO: For easy processing processing of emissions.
STIPULATIONS_ON_USE: Use at your own risk.
OTHER_COMMENTS: None.
REVISION: R0
R0: Preliminary data
"""
        hr_f.save(
            self.diurnalpath, format='NETCDF4_CLASSIC', complevel=1, verbose=0
        ).close()
        return self.make_diurnalfile()

    def make_dayofweekfile(self, propath=None, read_kwds=None):
        """
        Arguments
        ---------
        propath : str
            path to tpro file ATPRO_WEEKLY file. Requires columns
            Mon, Tue, Wed, Thu, Fri, Sat, Sun, and comment.
        read_kwds : dict or None
            If None, default read_kwds are dict(comment='#', index_col=0,
            names=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'comment'])
            which is equivalent to ignoring comments, indexing the first column
            and having no header row.

        Returns:
            df : PseudoNetCDFFile
                IOAPI-like file with day of week allocations for sectors (as
                variables) with shape  TSTEP=7, LAY=1, ROW=NROWS, COL=NCOLS
        """

        if self._dayofweekfile is not None:
            return self._dayofweekfile
        elif os.path.exists(self.dayofweekpath):
            self._dayofweekfile = pnc.pncopen(
                self.dayofweekpath, format='ioapi'
            )
            return self.dayofweekfile

        if propath is None:
            raise KeyError(
                f'propath required because {self.dayofweekpath} not found'
            )

        print(
            f'{self.dayofweekpath} not available; calculating from {propath}'
        )
        if read_kwds is None:
            read_kwds = dict(
                comment='#', index_col=0,
                names='Mon Tue Wed Thu Fri Sat Sun comment'.split()
            )
        wkdf = pd.read_csv(propath, **read_kwds)
        if 'comment' not in wkdf.columns:
            wkdf['comment'] = wkdf.index.values

        wkdf.index.name = 'profile_id'
        tzf = self.timezonefile

        day_f = tzf.subset([])
        day_f.createDimension('TSTEP', 25).setunlimited(True)
        day_f.createDimension('LAY', 7)
        day_f.VGLVLS = np.arange(8)
        day_f.VGTYP = 6
        day_f.SDATE = 2020001
        day_f.STIME = 0
        day_f.TSTEP = 10000

        for wkidx, wkrow in wkdf.iterrows():
            cmt = wkrow['comment']
            label = getlabel(cmt)
            print(label, cmt)
            wkvals = weekdayfactor(wkdf, wkidx, tzf)
            wkvar = day_f.createVariable(
                label, 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
                long_name=label, units='1',
                var_desc=f'Scale a {label} monthly rate to a 7 weekday-specific rates'
            )
            wkvar[:] = wkvals

        day_f.updatemeta()
        day_f.updatetflag(overwrite=True)
        day_f.FILEDESC = (
            """
## NASA-like metadata
1, 2310
Unknown, Unknown
Unknown
EPA sector-based hourly profiles
Not Applicable
1, 1
2020, 01, 01, 2020, 01, 01
0
...
PI_CONTACT_INFO: unknown@unknown.com
PLATFORM: CMAQ Emission processing input
DATA_INFO:  All data in time fractions
UNCERTAINTY:  large, preliminary data based on US averages.
DM_CONTACT_INFO: Unknown, Unknown, Unknown, unknown@unknown.com
PROJECT_INFO: For easy processing processing of emissions.
STIPULATIONS_ON_USE: Use at your own risk.
OTHER_COMMENTS: The LAY dimension is day of the week (Mon, Tue, ..., Sun)."""
            + "Time is UTC, but the profiles are based on LST days. So, "
            + "UTC_Mon will include hours from Sun and Tue as appropriate "
            + """given the hour offset.
REVISION: R0
R0: Preliminary data
"""
        )
        day_f.save(
            self.dayofweekpath, format='NETCDF4_CLASSIC', complevel=1,
            verbose=0
        ).close()
        return self.dayofweekfile

    def make_monthlyfile(self, propath=None, read_kwds=None):
        """
        Arguments
        ---------
        propath : str
            path to tpro file ATPRO_MONTHLY file. Requires columns:
            Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Nov, Dec, comment
        read_kwds : dict or None
            If None, default read_kwds are dict(comment='#', index_col=0,
            names=['Jan', ..., 'Dec', 'comment'])
            which is equivalent to ignoring comments, indexing the first column
            and having no header row.

        Returns:
            df : PseudoNetCDFFile
                IOAPI-like file with month of year allocations for sectors (as
                variables) with shape TSTEP=12, LAY=1, ROW=NROWS, COL=NCOLS
        """

        if self._monthlyfile is not None:
            return self._monthlyfile
        elif os.path.exists(self.monthlypath):
            self._monthlyfile = pnc.pncopen(self.monthlypath, format='ioapi')
            return self.monthlyfile
        if propath is None:
            raise KeyError(
                f'propath required because {self.monthlypath} not found'
            )

        print(
            f'{self.monthlypath} not available; calculating from {propath}'
        )
        names = _monnames + ['comment']

        if read_kwds is None:
            read_kwds = dict(comment='#', index_col=0, names=names)

        mondf = pd.read_csv(propath, **read_kwds)
        if 'comment' not in hrdf.columns:
            hrdf['comment'] = hrdf.index.values

        tzf = self.timezonefile

        mon_f = tzf.subset([])
        mon_f.createDimension('TSTEP', 12).setunlimited(True)
        mon_f.SDATE = 2020001
        mon_f.STIME = 0
        mon_f.TSTEP = 24 * 30.5 * 10000

        for monidx, monrow in mondf.iterrows():
            cmt = monrow['comment']
            label = getlabel(cmt)
            print(label, cmt)
            monvals = monfactor(mondf, monidx, tzf)
            monvar = mon_f.createVariable(
                label, 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
                long_name=label, units='1',
                var_desc=f'Scale a {label} annual rate to 12 monthly rate'
            )
            monvar[:] = monvals[:, None]

        mon_f.updatemeta()
        mon_f.updatetflag(overwrite=True)
        mon_f.FILEDESC = """
## NASA-like metadata
1, 2310
Unknown, Unknown
Unknown
EPA sector-based hourly profiles
Not Applicable
1, 1
2020, 01, 01, 2020, 01, 01
0
...
PI_CONTACT_INFO: unknown@unknown.com
PLATFORM: CMAQ Emission processing input
DATA_INFO:  All data in time fractions
UNCERTAINTY:  large, preliminary data based on US averages.
DM_CONTACT_INFO: Unknown, Unknown, Unknown, unknown@unknown.com
PROJECT_INFO: For easy processing processing of emissions.
STIPULATIONS_ON_USE: Use at your own risk.
OTHER_COMMENTS: None.
REVISION: R0
R0: Preliminary data
"""
        mon_f.save(
            self.monthlypath, format='NETCDF4_CLASSIC', complevel=1, verbose=0
        ).close()
        return self.monthlyfile

    def make_timezonefile(self):
        if self._timezonefile is not None:
            return self._timezonefile
        elif os.path.exists(self.timezonepath):
            self._timezonefile = pnc.pncopen(self.timezonepath, format='ioapi')
            return self._timezonefile
        print(
            f'{self.timezonepath} not available;'
            ' calculating UTCOFFSET in hours from longitude...', end=''
        )
        gf = pnc.pncopen(
            self.griddescpath, format='griddesc', GDNAM=self.gdnam
        )
        del gf.variables['TFLAG']
        gf.SDATE = 1970001
        I, J = np.meshgrid(np.arange(gf.NCOLS), np.arange(gf.NROWS))
        lon, lat = gf.ij2ll(I, J)
        utcoffset = (lon / 15)
        tzf = gf.subset([])
        tzvar = tzf.createVariable(
            'UTCOFFSET', 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
            long_name='UTCOFFSET', var_desc='UTCOFFSET', units='hours'
        )
        tzvar[:] = utcoffset
        mthdvar = tzf.createVariable(
            'METHOD', 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
            long_name='METHOD', units='none',
            var_desc='METHOD: 0=tz_world.geojson; 1=lon/15'
        )
        mthdvar[:] = 1
        tzf.updatetflag(overwrite=True)
        tzf.updatemeta()
        tzf.FILEDESC = """Calculated TZ from longitude"""
        tzf.HISTORY = """Calculated TZ from longitude"""
        tzf.save(
            self.timezonepath, format='NETCDF4_CLASSIC',
            complevel=1, verbose=0
        ).close()
        print('done')

        return self.timezonefile

    def get_factor(
        self, sector, refdate,
        diurnal=True, dayofweek=True, monthly=True, verbose=0
    ):
        if diurnal:
            if verbose > 0:
                print('Opening diurnal', flush=True)
            diurnalf = self.diurnalfile
            hfac = diurnalf.variables[sector][:]
        else:
            hfac = 1  # no need because data is already hourly

        if dayofweek:
            if verbose > 0:
                print('Opening dayofweek', flush=True)
            dayofweekf = self.dayofweekfile
            # Mon = 0
            dayidx = refdate.dayofweek
            dayfac = dayofweekf.variables[sector][:, dayidx][:, None]
        else:
            dayfac = 1  # no need because data is already monthly

        if monthly:
            if verbose > 0:
                print('Opening monthly', flush=True)
            monthf = self.monthlyfile
            # Jan = 0
            monidx = refdate.month - 1
            monfac = monthf.variables[sector][monidx]
        else:
            monfac = 1  # no need because data is already monthly

        # create an output factor
        factor = monfac * dayfac * hfac
        return factor

    def allocate(
        self, infile, outdate, alloc_keys, outpath=None,
        monthly=True, dayofweek=True, diurnal=True,
        time=None, format=None,
        overwrite=False, verbose=0
    ):
        """
        Arguments
        ---------
        infile : str or PseudoNetCDF File
            path to netcdf file (or file) to use as input (format keyword used
            as a modifier)
        outdate : datetime
            date to destination
        outpath : str or None
            path for output to be saved. If None, outf will be returned and not
            saved
        alloc_keys : mappable  or str
            alloc_keys key/value pairs map allocation variables (e.g., ENERGY)
            to variables in infile to allocate temporally. Each key should
            be in monthlyfile/dayofweekfile/diurnalfile variables. And each
            value is a list of variables in infile. One allocation variable can
            be assigned None instead of a list, which results in all unassigned
            variables being used. If alloc_keys is a str, this is equivalent to
            `alloc_keys={alloc_keys: None}`
        monthly : bool
            apply monthly scaling. If file already has months, use month=False
            and time=m to apply other scaling to time m.
        dayofweek : bool
            apply day of week  scaling. If file already has day of week, use
            dayofweek=False and time=d to apply other scaling to time d.
        diurnal : bool
            apply hour of day  scaling. If file already has hour of day, use
            diurnal=False and time=h to apply other scaling to time h.
        time : int or None
            if None, checks to ensure that file has only 1 time and uses first
            (i.e., 0)
        format : str
            format of file or meta data (e.g., netcdf or ioapi; see
            PseudoNetCDF pncopen)

        Returns
        -------
        outf : PseudoNetCDFFile
            file with temporal variation

        Notes
        -----

        1. month, dayofweek, and diurnal can be combined to exlude one or many
           scalings

        """
        remove = False

        if outpath is not None and os.path.exists(outpath):
            if not overwrite:
                raise IOError(f'{outpath} exists')
            else:
                remove = True

        refdate = outdate

        if verbose > 0:
            print('Opening input', flush=True)

        if isinstance(infile, str):
            ef = pnc.pncopen(infile, format=format)
        else:
            ef = infile

        if isinstance(alloc_keys, str):
            alloc_keys = {alloc_keys: None}

        all_keys = []
        for k, v in ef.variables.items():
            if 'LAY' in v.dimensions:
                all_keys.append(k)

        assigned_keys = []

        isnone = []
        for sector, varkeys in alloc_keys.items():
            if varkeys is None:
                isnone.append(sector)
            else:
                assigned_keys.extend(varkeys)

        unassigned_keys = list(set(all_keys).difference(assigned_keys))
        if len(isnone) > 1:
            raise ValueError(f'Can only have 1 None sector; got {isnone}')
        if len(isnone) == 1:
            alloc_keys[isnone[0]] = unassigned_keys

        if time is None:
            if len(ef.dimensions['TSTEP']) > 1:
                print('Time dimension is not 1, so you must choose a time')
            else:
                time = 0

        if format == 'ioapi':
            if verbose > 0:
                print('Appending TFLAG to exclude', flush=True)

        if verbose > 0:
            print('Creating output template', flush=True)

        outf = ef.subset([])
        if 'TFLAG' in outf.variables:
            del outf.variables['TFLAG']

        nsteps = 1
        if monthly:
            nsteps = nsteps * 1
            tstep = 30*240000
        if dayofweek:
            nsteps = nsteps * 1
            tstep = 240000
        if diurnal:
            nsteps = nsteps * 25
            tstep = 10000

        outf.createDimension('TSTEP', nsteps).setunlimited(True)

        if verbose > 0:
            print('Calculating composite factor', flush=True)

        for sectorkey, varkeys in alloc_keys.items():
            factor = self.get_factor(
                sectorkey, refdate,
                diurnal=diurnal, dayofweek=dayofweek, monthly=monthly
            )
            for varkey in varkeys:
                invar = ef.variables[varkey]
                if verbose > 0:
                    print(f'Scaling {varkey}...', flush=True)
                outvar = outf.copyVariable(invar, key=varkey, withdata=False)
                outvar.setncatts(
                    {pk: getattr(invar, pk) for pk in invar.ncattrs()}
                )
                outvar[:] = invar[time] * factor

        outf.SDATE = int(refdate.strftime('%Y%j'))
        outf.STIME = int(refdate.strftime('%H%M%S'))
        outf.TSTEP = tstep
        if format == 'ioapi':
            outf.updatemeta()
            outf.updatetflag(overwrite=True)

        history = getattr(outf, 'HISTORY', '')
        history += f'apply_temporal({locals})'
        setattr(outf, 'HISTORY', history)
        if outpath is not None and remove:
            os.remove(outpath)

        if outpath is None:
            return outf
        else:
            outf.save(outpath, verbose=0).close()
            return pnc.pncopen(outpath, format='ioapi')

    def get_diurnalfile(self, *args, **kwds):
        import warnings
        warnings.warn('Use make_diurnalfile instead', DeprecationWarning)
        return self.make_diurnalfile(*args, **kwds)

    def get_dayofweekfile(self, *args, **kwds):
        import warnings
        warnings.warn('Use make_dayofweekfile instead', DeprecationWarning)
        return self.make_dayofweekfile(*args, **kwds)

    def get_monthlyfile(self, *args, **kwds):
        import warnings
        warnings.warn('Use make_monthlyfile instead', DeprecationWarning)
        return self.make_monthlyfile(*args, **kwds)

    def get_timezonefile(self, *args, **kwds):
        import warnings
        warnings.warn('Use make_timezonefile instead', DeprecationWarning)
        return self.make_timezonefile(*args, **kwds)


def run():
    import argparse
    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    aa('-v', '--verbose', action='count', default=0)
    aa('--no-month', dest='month', default=True, action='store_false')
    aa('--no-dayofweek', dest='dayofweek', default=True, action='store_false')
    aa('--no-diurnal', dest='diurnal', default=True, action='store_false')
    aa('-O', '--overwrite', default=False, action='store_true')
    aa('-i', '--include', default=[], action='append')
    aa('-x', '--exclude', default=[], action='append')
    aa('-f', '--format', help='input format')
    aa('-t', '--time', type=int, default=None)
    aa(
        '-r', '--root', type=str, default='.',
        help='Path to existing gridded allocation files'
    )
    aa('sector', choices=known_sectors)
    aa('outdate', type=pd.to_datetime)
    aa('infile')
    aa('outpath')

    args = parser.parse_args()
    opts = vars(args)
    root = opts.pop('root')
    include = opts.pop('include', None)
    sector = opts.pop('sector')
    if include is None:
        opts['alloc_keys'] = sector
    else:
        opts['alloc_keys'] = {sector: include}

    return Temporal(root=root).allocate(**opts)


if __name__ == '__main__':
    import sys
    if len(sys.argv[:]) > 1:
        run()
    else:
        print('To run as a script, add an argument (e.g., -h)')
