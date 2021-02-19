import PseudoNetCDF as pnc
import pandas as pd


class Spatial:
    def __init__(
        self, griddesc, gdnam, nominaldate='1970-01-01', **kwds
    ):
        nominaldate = pd.to_datetime(nominaldate)
        gf = pnc.pncopen(griddesc, format='griddesc', GDNAM=gdnam)
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

    def region_fromshapefile(
        self, shapepath, queryfield, query, key, fractional=True, simplify=None, buffer=0, use_tree=None, verbose=0
    ):
        "Thin wrapper on smokelite.util.fractional_overlap; see its docs"
        from ..util import fractional_overlap
        return fractional_overlap(
            ifile=self.regionfile, shapepath=shapepath, queryfield=queryfield,
            query=query, key=key, fractional=fractional, simplify=simplify,
            buffer=buffer, use_tree=use_tree, verbose=verbose
        )

    def add_spatialvariables(self, spatialdf=None, spatialf=None, keys=None):
        sptlf = self.spatialfile
        if spatialf is not None and spatialdf is not None:
            raise KeyError('supply either spatialdf or spatialvar; got both')
        if spatialf is not None:
            for key in keys:
                sptlf.copyVariable(spatialf.variables[key], key=key)
        if spatialdf is not None:
            from ..util import load_dataframe
            load_dataframe(sptlf, spatialdf)

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
