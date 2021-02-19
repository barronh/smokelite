import PseudoNetCDF as pnc
import pandas as pd
import numpy as np


class Spatial:
    def __init__(
        self, gridpath, nominaldate='1970-01-01', format='griddesc', **kwds
    ):
        """
        Arguments
        ---------
        gridpath : str
            path to a GRIDDESC file
        nominaldate : str
            Date for spatial and regional files (default: '1970-01-01')
        format : str
            griddesc, by default, but can be any ioapi_base class
        kwds : mappable
            Keywords for opening GRIDDESC. For example, GDNAM if there are
            multiple domains.

        Returns
        -------
        """
        nominaldate = pd.to_datetime(nominaldate)
        gf = pnc.pncopen(gridpath, format=format, **kwds)
        gf.SDATE = int(nominaldate.strftime('%Y%j'))
        gf.STIME = int(nominaldate.strftime('%H%M%S'))
        gf.TSTEP = 10000
        self.spatialfile = gf.subset([])
        uv = self.spatialfile.createVariable(
            'UNIFORM', 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
            long_name='UNIFORM', var_desc='UNIFORM', units='none'
        )
        uv[:] = 1 / uv.size
        self.regionfile = gf.subset([])
        dw = self.regionfile.createVariable(
            'DOMAINWIDE', 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
            long_name='DOMAINWIDE', var_desc='DOMAINWIDE', units='fraction'
        )
        dw[:] = 1.
        self.regions = ['DOMAINWIDE']

    def plotmap(self, key, label=None, ax=None, infile='either', gridspec_kw=None, **kwds):
        from ..util import plotmap
        if infile == 'either':
            try_spatial = try_region = True
        elif infile == 'region':
            try_spatial = False
            try_region = True
        elif infile == 'spatial':
            try_spatial = True
            try_region = False
        else:
            raise ValueError(
                f'infile must be either, region or spatial; got {infile}'
            )
        if try_region and key in self.regionfile.variables:
            plotf = self.regionfile
        elif try_spatial and key in self.spatialfile.variables:
            plotf = self.spatialfile
        else:
            raise ValueError(
                f'Could not find variable {key} in {infile}'
            )
        return plotmap(
            plotf, plotf.variables[key][0, 0], label=label, ax=ax,
            gridspec_kw=gridspec_kw, **kwds
        )


    def regions_fromindex(self, idx_var, idx2name):
        """
        Quick function to conver cell codes to fraction-like (0 or 1) variables

        Arguments
        ---------
        idx_var : PseudoNetCDFVariable
            variable (TSTEP, LAY, ROW, COL) with indices in idx2name where
            cells that have each idx will be identified as 100% name
        idx2name : mappable
            key/value pairs where the key is the value in idx_var and name is
            the name for a variable with those assignments.

        Returns
        -------
        None
        """
        for idx, name in idx2name.items():
            ivar = self.regionfile.createVariable(
                name, 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
                long_name=name.ljust(16), units='fraction',
                var_desc=f'{name} (0: not mine; 1: all mine)'.ljust(80),
            )
            dimidx = list(np.where(idx_var[:] == idx))
            while len(dimidx) < 4:
                dimidx.insert(0, slice(None))
            dimidx = tuple(dimidx)
            ivar[dimidx] = 1

    def region_fromshapefile(
        self, shapepath, queryfield, query, key, fractional=True,
        simplify=None, buffer=0, use_tree=None, verbose=0
    ):
        "Thin wrapper on smokelite.util.fractional_overlap; see its docs"
        from ..util import fractional_overlap
        return fractional_overlap(
            ifile=self.regionfile, shapepath=shapepath, queryfield=queryfield,
            query=query, key=key, fractional=fractional, simplify=simplify,
            buffer=buffer, use_tree=use_tree, verbose=verbose
        )

    def add_spatialvariables(self, spatialdf=None, spatialf=None, keys=None):
        """
        Wrapper to add_variables
            self.add_variables(self.spatialfile, ...)
        """
        return self.add_variables(
            destf=self.spatialfile, df=spatialdf, nf=spatialf, keys=keys
        )

    def add_regionvariables(self, regiondf=None, regionf=None, keys=None):
        """
        Wrapper to add_variables
            self.add_variables(self.spatialfile, ...)
        """
        return self.add_variables(
            destf=self.regionfile, df=regiondf, nf=regionf, keys=keys
        )

    def add_variables(self, destf, df=None, nf=None, keys=None):
        """
        Quick function to conver cell codes to fraction-like (0 or 1) variables

        Arguments
        ---------
        destf : PseudoNetCDFFile or None
            Destination file (usually, regionfile or spatialfile
        df : pandas.DataFrame or None
            If provided, must have columns matching a subset of dimensions
            (TSTEP, LAY, ROW, COL)
        nf : PseudoNetCDFFile or None
            If provided, must match dimensions of destf and keys
        keys : list or None
            If None, all columns or variables will be used. If a list, then
            only these keys will be used.

        Returns
        -------
        None
        """
        if nf is not None and df is not None:
            raise KeyError('supply either spatialdf or spatialvar; got both')
        if nf is not None:
            if keys is None:
                keys = [
                    k for k in nf.variables.items()
                    if v.dimensions == ('TSTEP', 'LAY', 'ROW', 'COL')
                ]

            for key in keys:
                destf.copyVariable(nf.variables[key], key=key)
        if df is not None:
            from ..util import load_dataframe
            load_dataframe(destf, df, keys=keys)

    def allocate(self, infile, alloc_keys, outpath=None, **save_kwds):
        """
        Arguments
        ---------
        infile : str or PseudoNetCDF File
            path to netcdf file (or file) to use as input (format keyword used
            as a modifier)
        alloc_keys : mappable  or str
            each key should exist in the vertical allocation file, and values
            should correspond to variables in the infile. If is a str, then
            all allocatable variables will be asisgned to that csv key.
        outpath : str or None
            path for output to be saved. If None, outf will be returned and not
            saved

        Returns
        -------
        outf : PseudoNetCDFFile
            file with spatial variation

        Notes
        -----

        """
        if isinstance(infile, str):
            infile = pnc.pncopen(infile, format=format)

        if isinstance(alloc_keys, str):
            alloc_keys = {alloc_keys: None}

        all_keys = []
        for k, v in infile.variables.items():
            if 'LAY' in v.dimensions:
                all_keys.append(k)

        assigned_keys = []

        isnone = []
        for (region, srgkey), varkeys in alloc_keys.items():
            if varkeys is None:
                isnone.append((region, srgkey))
            else:
                assigned_keys.extend(varkeys)

        unassigned_keys = list(set(all_keys).difference(assigned_keys))
        if len(isnone) > 1:
            raise ValueError(f'Can only have 1 None sector; got {isnone}')
        if len(isnone) == 1:
            alloc_keys[isnone[0]] = unassigned_keys

        outf = self.spatialfile.subset([])
        for (regionkey, allockey), varkeys in alloc_keys.items():
            regionvar = self.regionfile.variables[regionkey]
            allocvar = self.spatialfile.variables[allockey]
            factor = regionvar[:] * allocvar[:]
            factor /= factor.sum()
            for varkey in varkeys:
                invar = infile.variables[varkey]
                outvar = outf.createVariable(
                    varkey, 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
                    long_name=varkeys, vardesc=varkey,
                    units=getattr(invar, 'units', 'unknown')
                )
                outvar[:] = invar[:] * factor

        if outpath is None:
            return outf
        else:
            return outf.save(outpath, **save_kwds)
