import os
import numpy as np
import pandas as pd
import PseudoNetCDF as pnc


def add_pressure(
    va_df, ptop=5000., psfc=101325., sigmakey='Sigma', pressurekey='Pressure',
    inplace=True
):
    """
    Add a pressure variable based on a known top of atmosphere and surface

    Arguments
    ---------
    va_df : pd.DataFrame
        file must contain sigmakey
    sigmakey : str
        name of Sigma variable
    ptop : float
        top of atmosphere in pascals (e.g., 5000. or 10000.)
    psfc : float
        surface of the earth for approximation (e.g., 101325.)
    pressurekey : str
        Key to add pressure as
    inplace : bool
        add to va_df (or copy)

    Returns
    -------
    outf : pd.DataFrame
        same as va_df, but with Pressure

    Notes
    -----

    Older files likely use ptop = 10000., while newer ones may use 5000.
    """
    if inplace:
        outdf = va_df
    else:
        outdf = va_df.copy()
    outdf.loc[:, pressurekey] = outdf.loc[:, sigmakey] * (psfc - ptop) + ptop
    return outdf


def interp_va(
    va_df, vglvls, vgtop=5000., psfc=101325., metakeys=None, verbose=False
):
    """
    Interpolate vertical allcoation dataframe to new vglvls

    Arguments
    ---------
    va_df : pandas.DataFrame
        must contain Pressure values that are top of the level values
    vglvls : array-like
        VGLVLS values (edges) of layers. First value will be ignored
    vgtop : scalar
        VGTOP from IOAPI, which is top of atmosphere in Pascals
    psfc : scalar
        Pressure at the surface for calculation
    metakeys : list or None
        list of keys that should *not* be renormalized to 1 (default
        ['Sigma', 'Alt', 'L'])
    verbose : bool
        show warnings

    Returns
    out_df : pandas.DataFrame
        vertical allocation data consistent with vglvls
    """
    from collections import OrderedDict
    x = vglvls[:]
    if metakeys is None:
        metakeys = ['Sigma', 'Alt', 'L']
    xp = np.append(1, (va_df.Pressure.values - vgtop) / (psfc - vgtop))

    if xp[-1] < xp[0]:
        xp = xp[::-1]
        invert = True
    else:
        invert = False

    if verbose > 0:
        print(x)
        print(xp)

    out = OrderedDict()

    for key in va_df.columns:
        if key in metakeys:
            # Pressure and sigma should not have a 0 surface value
            # Layer heights should, but all metakeys are treated the same.
            fp = np.append(va_df[key].values[0], va_df[key].values)
        else:
            # The fraction at the surface (i.e., below layer 1)
            # is 0
            fp = np.cumsum(np.append(0, va_df[key].values))

        if invert:
            fp = fp[::-1]

        leftval = None
        rightval = None
        # Left 1 assumes the cumulative sum
        interpvals = np.interp(x, xp, fp, left=leftval, right=rightval)
        if verbose > 0:
            print(key)
            print(fp)
        if verbose > 1:
            print(interpvals)
        if key in metakeys:
            out[key] = interpvals[1:]
        else:
            out[key] = np.diff(interpvals)
            out[key] /= out[key].sum()
        if verbose:
            print(out[key])

    outdf = pd.DataFrame.from_dict(out)
    outdf['Sigma'] = x[1:]
    return outdf


def dfmake3d(
    infile, va_df, frackey, sigmakey='Sigma',
    outpath=None, overwrite=False, **save_kwds
):
    """
    Thin wrapper on make3d

    Arguments
    ---------
    infile : PseudoNetCDFFile
        file with IOAPI structure
    va_df : pd.DataFrame
        has sigmakey and frackey
    frackey : str
        column with layer fractions, should sum to 1
    sigmakey : str
        column with top edge sigma values to be used for VGLVLS
    outpath : str or None
        if str, saves to disk and returns result of save
        if None, returns in-memory file
    overwrite : bool
        If True, remove outpath if it exists
    save_kwds : dict
        if outpath is str, then save_kwds are used for save call

    Returns
    -------
    outfile : PseudoNetCDFFile or handle
        Result of PseudoNetCDFFile.save
    """
    if outpath is not None and os.path.exists(outpath):
        if overwrite:
            os.remove(outpath)

    vglvls = np.append(1, va_df[sigmakey].values)
    outfile = make3d(infile, va_df[frackey].values, vglvls)
    if outpath is None:
        return outfile
    else:
        return outfile.save(outpath, **save_kwds)


def make3d(infile, layerfractions, vglvls):
    """
    Interpolate vertical allocation dataframe to new vglvls

    Arguments
    ---------
    infile : PseudoNetCDFFile
        file with IOAPI structure
    layerfractions : array
        layer fractions should have one dimension and sum to 1
    vglvls : array
        sigmatops are the edge values to be used for VGLVLS, extra values
        (i.e., more than layerfractions.shape[0] + 1) will be pruned

    Returns
    -------
    """
    nz = layerfractions.shape[0]
    outfile = infile.slice(LAY=[0]*nz)

    for key, var in outfile.variables.items():
        if key != 'TFLAG':
            var[:] *= layerfractions[None, :, None, None]
    outfile.VGLVLS = vglvls[:nz + 1].astype('f')
    outfile.NLAYS = nz
    return outfile


class Vertical:
    def __init__(
        self, csvpath, outvglvls, outvgtop, read_kwds=None,
        pressurekey='Pressure', sigmakey='Sigma', csvvgtop=5000.,
        metakeys=None, psfc=101325., prune=True, verbose=0
    ):
        """
        Arguments
        ---------
        csvpath : str or pd.DataFrame
            path to vertical allocation file or DataFrame already opened. Each
            key represents a value to use as a fractional allocation to this
            level.
        outvglvls : array
            Has nz+1 values, which will match the output file. The top edges
            outvglvls[1:] will be use for interpolation
        outvgtop : float
            top of the model atmosphere for output
        pressurekey : str
            key in csv that holds or will hold Pressure
        sigmakey : str
            key in csv that holds sigma (sigma should be layer tops)
        csvvgtop : float
            top of the model atmosphere when deriving csv
        metakeys : list or None
            If None, defaults to ['Sigma', 'Alt', 'L', 'Pressure']
        psfc : float
            pressure in Pascals at the surface. Used for interpolation between
            sigma grids
        prune : bool
            remove unnecessary levels
        verbose : int
            count of verbosity level

        Returns
        -------
        """
        if read_kwds is None:
            read_kwds = dict(comment='#')
        self.pressurekey = pressurekey
        self.sigmakey = sigmakey
        self.verbose = verbose
        self.outvglvls = outvglvls
        self.outvgtop = outvgtop
        self.psfc = psfc

        if isinstance(csvpath, pd.DataFrame):
            self.indf = csvpath
        else:
            self.indf = pd.read_csv(csvpath, **read_kwds)

        if self.pressurekey not in self.indf.columns:
            add_pressure(
                self.indf, ptop=csvvgtop, psfc=psfc, sigmakey=sigmakey,
                pressurekey=pressurekey, inplace=True
            )

        if metakeys is None:
            metakeys = [
                key
                for key in
                [self.sigmakey, self.pressurekey, 'Alt', 'L']
                if key in self.indf.columns
            ]

        self.metakeys = metakeys

        # Interp_va ignores first outvglvls level automatically
        outdf = interp_va(
            self.indf, outvglvls, vgtop=outvgtop, psfc=psfc,
            metakeys=metakeys, verbose=self.verbose
        )
        outdf.loc[:, 'LAYER1'] = 0
        outdf.loc[0, 'LAYER1'] = 1
        layerused = outdf.drop(self.metakeys, axis=1).values.sum(1) != 0
        layerused[0] = True
        if not prune:
            layerused[:] = True
        usedupto = np.cumsum(layerused[::-1])[::-1] != 0
        self.outdf = outdf.loc[usedupto]

    def allocate(self, infile, alloc_keys, outpath=None, save_kwds=None):
        """
        Arguments
        ---------
        infile : str or PseudoNetCDFFile
            file to allocate vertically
        alloc_keys : mappable  or str
            alloc_keys are mappings of vertical allocation variables to the
            variables they should be used to allocate. Each key should exist in
            the vertical allocation file (csvpath), and values should
            correspond to variables in the infile. One key may be mapped to
            None, which will apply this key to all unmapped infile variables.
            If alloc_keys is a str, this is the same as
            `alloc_keys={alloc_keys: None}`.
        outpath : str or None
            path to save output
        save_kwds : mappable
            keywords for outf.save method

        Returns
        -------
        outf : PseudoNetCDFFile or PseudoNetCDFFile.save output
            Has each variable associated with a csv key in alloc_keys as a
            variable, where the old file has 1 layer, the new file has nz
            layers.

        Examples
        --------
        1. The following example would vertically allocate only NOX_ENERGY
        based on the ENERGY vertical profile.

        import pandas as pd
        import numpy as np
        import PseudoNetCDF as pnc
        import smokelite

        vadf = pd.DataFrame.from_dict({
            'Sigma': [0.99, .98, .97, .95, 0.9],
            'Pressure': [100361.75, 99398.5 , 98435.25, 96508.75, 91692.5],
            'ENERGY': [.12, .13, .26, .28, .21]
        })
        infile = pnc.pncopen(
            '../../../Downloads/GRIDDESC', format='griddesc', GDNAM='36US3'
        ).subset([])
        enevar = infile.createVariable(
            'NOX_ENERGY', 'f', ('TSTEP', 'LAY', 'ROW', 'COL')
            long_name='NOX_ENERGY', var_desc='Energy NOx', units='TgNO2/yr'
        )
        enevar[:] = 1
        othvar = infile.createVariable(
            'NOX_OTH', 'f', ('TSTEP', 'LAY', 'ROW', 'COL')
            long_name='NOX_OTH', var_desc='Other NOx', units='TgNO2/yr'

        )
        othvar[:] = 2
        infile.SDATE = 2017001
        infile.STIME = 0
        infile.TSTEP = 0
        infile.updatetflag()

        va = smokelite.Vertical(
            vadf, csvvgtop=5000,
            outvglvls=np.linspace(1, .9, 14), outvgtop=5000.
        )


        # Allocate NOX_ENERGY using ENERGY
        enefile = va.allocate(
            infile, alloc_keys=dict(
                ENERGY=['NOX_ENERGY',]
            )
        )

        # Allocate all variables using LAYER1
        lay1file = va.allocate(infile, alloc_keys=dict(LAYER1=None))

        # Simultaneously NOX_ENERGY using ENERGY and all other variables using
        # LAYER1
        bothfile = va.allocate(
            infile, alloc_keys=dict(LAYER1=None, ENERGY=['NOX_ENERGY',])
        )
        """
        if save_kwds is None:
            save_kwds = dict(
                format='NETCDF4_CLASSIC', complevel=1,
                verbose=self.verbose
            )

        if isinstance(alloc_keys, str):
            alloc_keys = {alloc_keys: None}

        if isinstance(infile, str):
            infile = pnc.pncopen(infile, format='ioapi')

        all_keys = []
        for k, v in infile.variables.items():
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

        sectorfiles = []
        for sector, varkeys in alloc_keys.items():
            layerfractions = self.outdf.loc[:, sector].values
            sectorfile = make3d(
                infile.subset(varkeys), layerfractions, self.outvglvls
            )
            sectorfiles.append(sectorfile)

        outfile = sectorfiles[0]
        for sectorfile in sectorfiles[1:]:
            for key, var in sectorfile.variables.items():
                if key != 'TFLAG':
                    outfile.copyVariable(var, key=key)

        if outpath is not None:
            return outfile.save(outpath, **save_kwds)
        else:
            return outfile
