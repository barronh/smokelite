__all__ = ['load_dataframe', 'fractional_overlap']


def plotmap(gf, plotvar, label=None, ax=None, gridspec_kw=None, **kwds):
    """
    Arguments
    ---------
    gf : PseudoNetCDFFile
        used for getproj command
    plotvar : PseudoNetCDFVariable
        used for values and units
    label : str
        Override units for label
    ax : matplotlib.axes.Axes
        used for map; if none, subplots(1, 1) is used with gridspec_kw
    gridspec_kw : mappable
        keywords for generating new axes
    kwds : mappable
        keywords for pcolormesh

    Returns
    -------
    ax : maptlotlib.axes.Axes
        axes with plot and map outline
    """
    import matplotlib.pyplot as plt
    try:
        import pycno
        has_pycno = True
    except ImportError:
        has_pycno = False

    if ax is None:
        fig, ax = plt.subplots(1, 1, gridspec_kw=gridspec_kw)
    p = ax.pcolormesh(plotvar, **kwds)
    if has_pycno:
        proj = gf.getproj(withgrid=True)
        cno = pycno.cno(proj=proj)
        cno.draw()
    if label is None:
        label = getattr(plotvar, 'units', 'unknown')
    fig.colorbar(p, label=label)
    return ax


def fractional_overlap(
    ifile, shapepath, queryfield, query, key='MINE', fractional=True,
    simplify=None, buffer=0, use_tree=None, verbose=0
):
    """
    Arguments
    ---------
    ifile : PseudoNetCDFFile ioapi_base
        file that supports ij2ll, ll2xy, and xy2ll and has dimensions
        ('TSTEP', 'LAY', 'ROW', 'COL')
    shapepath : str
        path to shapefile
    queryfield : str
        name of attribute to query for query
    query : str or function
        if str, any attribute with that value will be selected
    key : str
        key for output variable in ifile
    fractional : bool
        If True, then use fractional area overlap otherwise centroid
    simplify : float or None
        Simplify to speed-up processing.
    buffer : float or None
        If None, do not simplify. If float, buffer. This can be useful to
        simplify self-intersecting polygons.
    use_tree : bool or none
        If None, use_tree will be set to true if there are more than 100
        polygons.
    verbose : int
        Counter for verbosity.

    Returns
    -------
    mine : PseudoNetCDFVariable
        variable with fractional area overlap
    """
    from types import FunctionType
    import numpy as np
    import shapefile as shp
    from shapely.geometry import Polygon, Point, shape
    from shapely.prepared import prep
    from shapely.ops import cascaded_union

    shpf = shp.Reader(shapepath)
    fldnames = [n for n, t, l, d in shpf.fields][1:]
    if verbose > 0:
        print('Field names', fldnames)
    if verbose > 1:
        print('Example fields', shpf.record(0))

    if isinstance(query, FunctionType):
        queryfunc = query
    else:
        def queryfunc(x):
            return x == query

    attridx = fldnames.index(queryfield)
    if verbose > 0:
        print('Query features...', flush=True, end='')

    recordnums = [
        ri for ri, rec in enumerate(shpf.iterRecords())
        if queryfunc(rec[attridx])
    ]
    if verbose > 0:
        print(f' found {len(recordnums)}')
    if verbose == 2:
        print([shpf.record(n)[attridx] for n in recordnums])
    elif verbose > 2:
        print(fldnames)
        print([shpf.record(n) for n in recordnums])

    # Centroids
    I, J = np.meshgrid(np.arange(ifile.NCOLS), np.arange(ifile.NROWS))
    LON, LAT = ifile.ij2ll(I, J)
    if fractional:
        # Centroids + 1
        Id, Jd = np.meshgrid(
            np.arange(ifile.NCOLS + 1),
            np.arange(ifile.NROWS + 1)
        )
        LONd, LATd = ifile.ij2ll(Id, Jd)
        # As X coordinates
        Xd, Yd = ifile.ll2xy(LONd, LATd)
        # As edge coordinates
        XD = Xd - ifile.XCELL / 2
        YD = Yd - ifile.YCELL / 2
        LOND, LATD = ifile.xy2ll(XD, YD)

    mine = ifile.createVariable(
        key, 'f', ('TSTEP', 'LAY', 'ROW', 'COL'),
        units='fraction', long_name=key, var_desc=key
    )
    mine.var_desc = '0 means not mine, 1 means all mine, inbetween is partial'
    if verbose > 0:
        print('Making polygons...', flush=True)
    shapes = [shpf.shape(rn) for rn in recordnums]
    polygons = []
    for s in shapes:
        p = shape(s)
        polygons.append(p)

    if buffer is not None:
        if verbose > 0:
            print(f'Buffer shapes with buffer {buffer}...', flush=True)
        polygons = [p.buffer(buffer) for p in polygons]
    if simplify is not None:
        if verbose > 0:
            print(f'Simplifying shapes: simplify({simplify})...', flush=True)
        polygons = [p.simplify(simplify) for p in polygons]
    # The optimal method for calculation of overlap is a complex problem that
    # will likely depend on the number of polygons and the structure/complexity
    # of the polygons. For this work, I implemented three possible solutions
    # and have empirically determined which one will be used.
    # 1. If there are many, the cascaded_union becomes prohibitive.
    #    In this case, use a STRtree to quickly query close ppolygons
    # 2. If there are multiple (not many), then create an puberpoly.  If there
    #    are none, the uberpoly is empty so skip and this is also fast.
    # 3. If there is just one, don't waste the computation.
    if use_tree is None:
        use_tree = len(polygons) > 100

    if use_tree:
        from shapely.strtree import STRtree
        tree = STRtree(polygons)
    else:
        if len(polygons) > 1 or len(polygons) == 0:
            if verbose > 0:
                print('Cascading union...', flush=True)
            uberpoly = cascaded_union(polygons)
        else:
            uberpoly = polygons[0]
        if verbose > 0:
            print('Making envelope...', flush=True)
        envelope = uberpoly.envelope
        if verbose > 0:
            print('Making prepared uber polygon...', flush=True)
        puberpoly = prep(uberpoly)
    if verbose:
        onepercent = int(ifile.NROWS * ifile.NCOLS / 100)
        print('Processing cells', flush=True)

    if verbose > 0:
        print('1234567890' * 10, flush=True)

    for j, i in np.ndindex(ifile.NROWS, ifile.NCOLS):
        if verbose > 0 and (verbose < 3 or not use_tree):
            n = j * ifile.NCOLS + i
            if n % onepercent == 0:
                print('.', end='', flush=True)

        if fractional:
            gpoly = Polygon([
                [LOND[j + 0, i + 0], LATD[j + 0, i + 0]],
                [LOND[j + 0, i + 1], LATD[j + 0, i + 1]],
                [LOND[j + 1, i + 1], LATD[j + 1, i + 1]],
                [LOND[j + 1, i + 0], LATD[j + 1, i + 0]],
                [LOND[j + 0, i + 0], LATD[j + 0, i + 0]],
            ])
            if use_tree:
                hits = tree.query(gpoly)
                if verbose > 2:
                    print(len(hits), end='.', flush='True')
                for hit in hits:
                    if hit.intersects(gpoly):
                        intx = gpoly.intersection(hit)
                        farea = intx.area / gpoly.area
                        mine[0, 0, j, i] += farea
            else:
                if envelope.intersects(gpoly):
                    if puberpoly.intersects(gpoly):
                        intx = gpoly.intersection(uberpoly)
                        farea = intx.area / gpoly.area
                        mine[0, 0, j, i] = farea
        else:
            gp = Point(LON[j, i], LAT[j, i])
            if use_tree:
                hits = tree.query(gp)
                if verbose > 2:
                    print(len(hits), flush='True')

                for hit in hits:
                    if hit.contains(gp):
                        mine[0, 0, j, i] = 1
            else:
                if envelope.intersects(gpoly):
                    if puberpoly.contains(gp):
                        mine[0, 0, j, i] = 1

    if verbose > 0:
        print()
    return mine


def load_dataframe(
    pfile, df, func='mean', units='unknown',
    dims=None, coords=False, keys=None
):
    """
    Arguments
    ---------
    pfile : PseudoNetCDFFile
        must have dimensions ('TSTEP', 'LAY', 'ROW', 'COL')
    df : pandas.DataFrame
        must have columns matching dimensions
    func : str
        Used to aggregate values within a unique coordinate
    units : str
        Default value for units
    dims : tuple or None
        If None, defaults to ('TSTEP', 'LAY', 'ROW', 'COL'), otherwise
        tuple of strings indicating dimensions for new variables
    coords : bool
        If True, copy dimension variables as IN_{dim}
    keys : list or None
        If None, all columns will be exported. If a list, only those
        keys will be output.

    Returns
    -------
    pfile : PseudoNetCDFFile
    """
    incols = df.columns
    # dims = pfile.dimensions
    dims = ('TSTEP', 'LAY', 'ROW', 'COL')
    gkeys = [k for k in incols if k in pfile.dimensions]
    dfgb = df.groupby(gkeys, as_index=False)
    gdf = getattr(dfgb, func)()
    vals = {k: gdf[k].values for k in gkeys}
    slices = {}
    for k, v in vals.items():
        if k in pfile.variables:
            slices[k] = pfile.val2idx(k, v)
        else:
            slices[k] = v.astype('i')

    slicer = tuple([
        slices.get(k, slice(None))
        for k in dims
    ])

    if keys is None:
        keys = incols

    for outkey in keys:
        outval = gdf[outkey]
        if outkey in gkeys:
            if not coords:
                continue

            outkey = f'IN_{outkey}'
        var = pfile.createVariable(
            outkey, outval.dtype.char, dims,
            units=units, long_name=outkey.ljust(16), var_desc=outkey.ljust(80)
        )
        var[slicer] = outval

    return pfile
