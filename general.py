from definitions import *


#################################################################################
#						Reading in fits files and the like						#
#################################################################################


# Reads a .fits img and returns the data array + WCS information
# If trim=True, we trim the image from 4D (if it is 4D) to 2D
def read_fits_image(imagename, read_header=False, trim=False):
	
	idx = 0
	hdu = fits.open(imagename)
	
	try: hdu[idx].data.shape
	except: idx = 1
	
	header = hdu[idx].header
	w = wcs.WCS(hdu[idx].header)
	data = hdu[idx].data
	hdu.close( )
	
	if data.ndim == 4 and trim:
		data = data[0,0]
		w = w.dropaxis(3)
		w = w.dropaxis(2)
	
	if not read_header:
		return data, w
	else:
		return data, w, header


# Saves a (modified) fits image (2D array) with header to <outname>
# I always forget this function name: here's some Ctrl+F keywords: save image imgdata
def write_fits_image(data, header, outname, overwrite=True, verbose=True):
	
	hdu = fits.PrimaryHDU(data=data, header=header)
	hdu.writeto(outname, overwrite=overwrite)
	if verbose:	print("Wrote to fits file", outname)




#################################################################################
#						General Function Definitions							#
#################################################################################


# Quick check if x is a list or a np.array
def check_if_list(x):
	lst = isinstance(x, list)
	arr = isinstance(x, np.ndarray)
	tup = isinstance(x, tuple)
	if (lst == True) or (arr == True) or (tup == True):
		return True
	else: return False


# Check if directory dirname exists - if not, create it
def check_and_create_dir(dirname, verbose=True):
	
	if not dirname.endswith('/'):
		dirname += '/'
	
	if not os.path.isdir(dirname):
		if verbose: print("\nCreated directory", dirname)
		os.mkdir(dirname)
	
	return dirname


def check_fits_dim(img, wcs=None):
	
	if img.ndim == 4:
		newimg = img[0,0]
	if not wcs is None:
		if wcs.naxis == 4:
			newwcs = wcs.dropaxis(3)
			newwcs = newwcs.dropaxis(2)
		return newimg, newwcs
	return newimg
	

# Human sorting of strings with numbers (taken from stackoverflow)
def human_sorting(mylist):
	
	convert = lambda text: int(text) if text.isdigit( ) else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(mylist, key=alphanum_key)


# Shortcut for formatting a float <val> to <dec> decimal places.
def form(val,dec):
	return '{:.{prec}f}'.format(val, prec=dec)


# Always forget how to do this... get the pixel scale of an image (via WCS object)
def get_pixel_scale(wcs):
	from astropy.wcs.utils import proj_plane_pixel_scales
	return np.array(proj_plane_pixel_scales(wcs)) * u.deg

# See, e.g., 1st eqn in https://casa.nrao.edu/casadocs/casa-5.1.0/global-task-list/task_imfit/about
def beam_area_in_pixels(bmaj, bmin, scale):
	return np.pi * bmaj * bmin / scale**2 / (4 * np.log(2))


# Determine the RMS in a (flattened) image after clipping all <sigma> pixels
def sigma_clipped_rms(data, sigma=3.0, axis=None):

	from astropy.stats import sigma_clipped_stats

	_, _, stddev = sigma_clipped_stats(data, sigma=sigma, axis=axis)
	return stddev


def has_unit_initialized(data):
	"""
	Check if <data> has units initialized, in which case True is returned. Else False.
	"""
	try:
		unit = data.unit
		return True if not unit is None else False
	except: return False


#################################################################################
#				Functions to be used with Error Propgation						#
#################################################################################


def addition(x, y):
	return x + y

def subtraction(x, y):
	return x - y

def multiplication(x, y):
	return x*y

def multiplication_N(*args):
	val = 1.0
	for arg in args:
		val = val * arg
	return val
	
def division(x, y):
	return x/y

def reciprocal(x):
	return 1.0/x

def tenth_power(x):
	return 10**x


#################################################################################


# Rounds down to nearest logarithmic decade
def rounddown(x):
	return 10**math.floor(math.log10(x))

# Rounds up to nearest logarithmic decade
def roundup(x):
	return 10**math.ceil(math.log10(x))

def round_by_base(x, base=10):
	return base * round(x/base)

def minmax(x):
	return np.array([np.min(x), np.max(x)])

def nanminmax(x):
	return np.array([np.nanmin(x), np.nanmax(x)])

def stddev_to_fwhm(stddev):
	return 2 * np.sqrt(2*np.log(2)) * stddev 

def fwhm_to_stddev(fwhm):
	return 1 / (2 * np.sqrt(2*np.log(2))) * fwhm

def area_under_gaussian(amp, stddev):
	return np.sqrt(2 * np.pi) * amp * stddev

def area_under_gaussian_fwhm(Nfwhm=1.0):
	"""
	Find the area under a 1D Gaussian function, integrated from -0.5 x Nfwhm x FWHM
	to +0.5 x Nfwhm x FWHM, divided by the total area from -infty to +infty.
	-------
	Parameters:
	-	Nfwhm (float)
		Multiple of the Gaussian FWHM across which we integrate. For Nfwhm = 1.0 (default), 
		we integrate across -0.5 x FWHM to +0.5 x FWHM, hence over 1x the total FWHM.
	-------
	Returns:
	-	Area(< Nfwhm) / Area(total)
	"""

	def mygauss(x, sigma=1.0):
		return np.exp(-x**2 / (2 * sigma**2))

	fwhm = stddev_to_fwhm(1.0)

	area_Nfwhm = quad(mygauss, -0.5 * Nfwhm * fwhm, 0.5 * Nfwhm * fwhm)[0]
	area_tot   = area_under_gaussian(1.0, 1.0)

	return area_Nfwhm / area_tot


def decode_list(mylist):
	return [myval.decode('utf-8') for myval in mylist]


def circular_mask(centre, arrsize, radius):
	""" Returns True when the array element is in the circular aperture
		and False otherwise. Note that centre should be a tuple. """
	
	if not check_if_list(arrsize):
		arrsize = [arrsize, arrsize]
		
	y,x = np.ogrid[-centre[0]:arrsize[0]-centre[0], -centre[1]:arrsize[1]-centre[1]]
	mask = x*x + y*y <= radius**2
	
	return mask


def circular_mask_center(data, radius):
	""" Assume data is a square 2x2 array"""
	assert len(data) == len(data[0])

	if len(data) % 2 == 0:
		print("\nWarning, cannot define central pixel for array which is even!")

	size = len(data)
	centre = int((len(data) - 1)/2)

	return circular_mask((centre, centre), size, radius)


def find_common_indices(x, y):
	"""
	Given two arrays x, y, we find the values they have in common.
	"""
	
	index = np.argsort(x)
	sorted_x = x[index]
	sorted_index = np.searchsorted(sorted_x, y)

	yindex = np.take(index, sorted_index, mode="clip")
	mask = x[yindex] != y

	result = np.ma.array(yindex, mask=mask)
	
	return result


# magic
def apply_argsort(arr, sortarr, axis=-1):
	
	i = list(np.ogrid[[slice(x) for x in arr.shape]])
	i[axis] = sortarr
	return arr[i]



	
	
def get_bin_centre(bins, data=None, inds=None, ctype='centre', wtype='confint', **kwargs):
	"""
	Description:
	What defines the centre of a bin? Is it simply the halfway point between the lower and upper
	edge, or is it the median value of the sources that fall into the bin? Well - with this function
	you can get either.
	
	WARNING: since 28 Aug 2019, we now use bootstrapping when <wtype>='confint'.
	-------
	Parameters:
	-	bins (array_like)
		Despite its name, this should be the *edges* of your bins, i.e. its length should be one
		more than the number of bins (get it?)
	-	data (array_like)
		The data that you have binned. Only required if not <ctype> equals centre.
	-	inds (array_like)
		Has the same length as <data>. For each element in <data>, corresponds to the bin in which
		this element belongs. You probably obtained this through np.digitize. If you don't supply
		it, but you do supply <data> and have <ctype> unequal to centre, we run np.digitize in here.
	-	ctype (string)
		Method of computing the bin centre. Currently we support
			-> centre: find the midway point between lower and upper bin edge
			-> median: find the median value of the sources in the given bin
			-> mean: take the mean value of sources in the bin.
			-> logcentre: find the midway point between the lower and upper bin edge in logspace
	-	wtype (string)
		Method of computing the bin width. Currently we support
			-> confint: find the 16th and 84th percentile of sources in the bin. Not used if 
				<ctype>==centre. WARNING: using bootstrapping now!
			-> full: find the width as (centre-lower edge) and (upper edge-centre). Default if
				<ctype>==centre.
	Kwargs: none currently exist.
	-------
	Returns:
	-	bin_centre (array_like)
		Centre of <bins>, through the method <ctype> you desire.
	-	bin_width (array_like)
		The width of <bins>, placed in a 2D array indicating for each bin the distance between the
		lower edge and centre as well as the centre and the upper edge. For <ctype>=centre, these
		two are of course equal.
	"""
	
	if (data is None) and not (ctype in ['centre','logcentre']):
		raise ValueError("\nYou need to supply the binned quantity for ctype!=centre!")
	
	if ctype == 'centre':
		bincentre = 0.5*(bins[1:] + bins[:-1])
		binwidth = np.diff(bins)
		
		if has_unit_initialized(bincentre):
			binwidth = np.asarray([binwidth.value, binwidth.value]) * binwidth.unit
		else: binwidth = 0.5*np.asarray([binwidth, binwidth])
		
		return bincentre, binwidth
	
	elif ctype == 'logcentre':
		
		logbins = np.log10(bins)
		logbincentre = 0.5*(logbins[1:] + logbins[:-1])
		binwidth = np.asarray([10**logbincentre - 10**logbins[:-1], 10**logbins[1:] - 10**logbincentre])
		
		return 10**(logbincentre), binwidth
		
	
	elif (ctype == 'median') or (ctype == 'mean'):
		
		my_avgfunc = np.median if ctype == 'median' else np.mean

		if inds is None:
			inds = np.digitize(data, bins)
		
		# If we have units, remove them to add them again in the final step
		if has_unit_initialized(data): data, unit = data.value, data.unit
		else: data, unit = data, 1.0
		# Ensure unit of <bins> is equal to that of <data>
		if has_unit_initialized(bins): bins = bins.to(unit).value
		else: pass
		
		
		bincentre, binwidth = np.zeros(len(bins)-1), np.zeros((2,len(bins)-1))
		for i in range(len(bins)-1):
			
			mask = (inds == i+1)
			#centre = np.median(data[mask])
			centre = my_avgfunc(data[mask])
			
			if wtype=='confint':
				if sum(mask) > 5:
					_, lo, hi = bootstrap(data[mask], ctype)
				else:
					lo, hi = centre-bins[i], bins[i+1]-centre
			elif wtype=='full':
				lo, hi = centre-bins[i], bins[i+1]-centre
			
			bincentre[i] = centre
			binwidth[:,i] = [lo,hi]
		
		return bincentre*unit, binwidth*unit


def combined_auto_binning(data1, data2):
	"""
	Description:
	Given two datasets <data1> and <data2>, combine them and bin them using the bins='auto' setting
	in np.histogram( ).
	"""
	
	fulldata = np.append(data1, data2)
	_, bins = np.histogram(fulldata, bins='auto')
	return bins


def combined_auto_binning_N(*args):

	fulldata = np.hstack(args)
	_, bins = np.histogram(fulldata, bins='auto')
	return bins


def combined_binning(data1, data2, Nbins=50):
	
	fulldata = np.append(data1, data2)
	_, bins = np.histogram(fulldata, bins=Nbins)
	return bins


def combined_binning_N(*args, Nbins=50):
	
	fulldata = np.hstack(args)
	_, bins = np.histogram(fulldata, bins=Nbins)
	return bins



#################################################
# General functions for plotting

def equal_axes(ax, lim=None, log=False, one2one=True, **kwargs):
	"""
	Given an axes object, ensure the x- and y-values have the same plotting limits, and plot a 
	one-to-one line if <one2one> is True.
	"""

	if log:
		ax.set_xscale('log')
		ax.set_yscale('log')

	if not lim is None:
		ax.set_xlim(lim[0], lim[1])
		ax.set_ylim(ax.get_xlim( ))

	else:
		(xmin, xmax), (ymin, ymax) = ax.get_xlim( ), ax.get_ylim( )
		axmin = min(xmin,ymin)
		axmax = max(xmax,ymax)
		ax.set_xlim(axmin,axmax)
		ax.set_ylim(ax.get_xlim( ))

	if one2one:
		ax.plot(ax.get_xlim( ), ax.get_ylim( ), color='black', zorder=-100, **kwargs)



# Ensure the axis is symmetric around zero, i.e. -100 to 100, where the extent is determined
# as the max offset from 0 in either the negative or positive direction
def symmetric_axis(ax, which='y'):
	
	if which.lower( ) in ['x','both']:
		xlim = ax.get_xlim( )
		xmax = max(np.abs(xlim[0]), xlim[1])
		ax.set_xlim(-xmax, xmax)
	if which.lower( ) in ['y','both']:
		ylim = ax.get_ylim( )
		ymax = max(np.abs(ylim[0]), ylim[1])
		ax.set_ylim(-ymax, ymax)


def share_axis_limit(axes, which_axis='x'):
	"""
	Given a number of axes, find the minimum and maximum plotting ranges, and set all axes to
	this plotting range.
	"""

	axflat = axes.flatten( )

	extrema_x = np.zeros((2, len(axflat)))
	extrema_y = np.zeros((2, len(axflat)))

	for ii, ax in enumerate(axflat):     
		extrema_x[:,ii] = np.asarray(ax.get_xlim( ))
		extrema_y[:,ii] = np.asarray(ax.get_ylim( ))

	minx, maxx = np.min(extrema_x[0]), np.max(extrema_x[1])
	miny, maxy = np.min(extrema_y[0]), np.max(extrema_y[1])

	for ii, ax in enumerate(axflat):
		if which_axis.lower( ) in ['x', 'both']:
			ax.set_xlim(minx, maxx)
		if which_axis.lower( ) in ['y', 'both']:
			ax.set_ylim(miny, maxy)



def diagonal_lines(axes, factor=2, add=False, **kwargs):
	"""
	Given a plot with equal axes limits in x and y, plot diagonal lines a factor of <factor>
	away from the one-to-one line. Parameter <add> determines whether we have a multiplicative
	effect, as you want in logspace (i.e. we have y=x as the one-to-one line, and also y=2x, y=0.5x), 
	or an additive one, where we plot y=x+add, y=x-add
	"""

	axes = [axes] if not check_if_list(axes) else axes

	if add:
		factor = np.append(factor, -1*float(factor))
	else:
		factor = np.append(factor, 1/float(factor))

	for ax in axes:
		lim = np.asarray([0, ax.get_xlim( )[1]])
		for fac in factor:
			if not add: ax.plot(lim, fac*lim, **kwargs)
			else: ax.plot(lim, lim+fac, **kwargs)
	return None


def axvhline(ax, value, **kwargs):
	ax.axhline(value, **kwargs)
	ax.axvline(value, **kwargs)
	return None
	
def axmedian(ax, data, which='y', **kwargs):
	if which.lower( ) == 'y':
		ax.axvline(np.median(data), **kwargs)
	else:
		ax.axhline(np.median(data), **kwargs)

def set_labels(ax, xlabel, ylabel, fontsize=24):
	
	ax.set_xlabel(xlabel, size=fontsize)
	ax.set_ylabel(ylabel, size=fontsize)


def square_subplots(Nplots):
	"""
	Given a number of subplots <Nplots>, create an (approximately) square grid of
	nrows, ncols.
	"""

	nrows = int(Nplots**0.5)
	ncols = Nplots//nrows
	if nrows*ncols < Nplots:
		nrows += 1

	return int(nrows), int(ncols)


def set_ticklabels(axes, location, values, axis='x'):
	
	is_list = check_if_list(axes)
	axes = [axes] if not is_list else axes
	
	newaxes = []
	for i, ax in enumerate(axes):
	
		if axis.lower( ) == 'x' or axis.lower( ) == 'both':
			ax.set_xticks(location)
			ax.set_xticklabels(values)
		if axis.lower( ) == 'y' or axis.lower( ) == 'both':
			ax.set_yticks(location)
			ax.set_yticklabels(values)
		newaxes.append(ax)
	if is_list: return newaxes
	else: return newaxes[0]


def set_tickparams(ax, which='both', width=[2,1], length=[8,4], direction=['out','inout']):

	if not check_if_list(width): width = [width, width]
	if not check_if_list(length): length = [length, length]
	if not check_if_list(direction): direction = [direction, direction]    

	if which in ['major','both']:
		ax.tick_params(which='major', width=width[0], length=length[0], \
		               direction=direction[0])
	if which in ['minor','both']:
		ax.tick_params(which='minor', width=width[1], length=length[1], \
		               direction=direction[1])


def increase_ticklabels(axes, size=20, which='both'):
	
	if which.lower( ) == 'both':
		which = ['major', 'minor']
	else: which = [which]
	
	is_list = check_if_list(axes)
	axes = [axes] if not is_list else axes
	
	newaxes = []
	for i, ax in enumerate(axes):
		
		if 'major' in which: ax.tick_params(axis='both', which='major', labelsize=size)
		if 'minor' in which: ax.tick_params(axis='both', which='minor', labelsize=size)
	
		newaxes.append(ax)
	if is_list: return newaxes
	else: return newaxes[0]


def do_minorticks(axes, axis='both', direction='out'):
    
	if not check_if_list(axes):
		axes = [axes]

	for ax in axes:
		if axis in ['x', 'both']:
			ax.tick_params(axis='x', which='minor', direction=direction, length=2, width=1)
		if axis in ['y', 'both']:
			ax.tick_params(axis='y', which='minor', direction=direction)
		ax.minorticks_on( )


# Because I always forget how to increase the cbar tick params...
def set_colorbar_labels(cbar, label, fontsize=24, labelsize=20, labelpad=0):
	
	cbar.set_label(label, fontsize=fontsize, labelpad=labelpad)
	cbar.ax.tick_params(labelsize=labelsize)


def get_vminvmax(ax):

	imgs = ax.get_images( )

	if len(imgs) > 0:
		vmin, vmax = imgs[0].get_clim( )
		return vmin, vmax 
	else: return None


# Delete the axes labels for an astropy wcs-projection plot
def delete_labels(ax):

	lon = ax.coords[0]
	lat = ax.coords[1]

	lon.set_axislabel('', size=0, color='white')
	lat.set_axislabel('', size=0, color='white')

	lon.set_ticklabel(size=0, color='white')
	lat.set_ticklabel(size=0, color='white')

	lon.set_ticks_visible(False)
	lat.set_ticks_visible(False)


# Given a set of levels, create also the negative levels and linestyles (dashed for negative, solid positive)
def set_contour_levels(levels):

	neg_levels  = list(np.sort(-1.0 * np.asarray(levels)))
	full_levels = neg_levels + list(levels)

	linestyles      = len(levels) * ['-']
	neg_linestyles  = len(levels) * ['--']
	full_linestyles = neg_linestyles + linestyles

	return full_levels, full_linestyles


# Adapted from stackoverflow (by user tmdavison)
# https://stackoverflow.com/questions/40929467/how-to-use-and-plot-only-a-part-of-a-colorbar-in-matplotlib
# Specify minColor & maxColor to truncate the cmap-range
def get_truncated_colormap(cmap, minval=0.0, maxval=1.0, minColor=0.0, maxColor=1.0, n=-1):
	
	cmap = plt.get_cmap(cmap)
	
	def truncate_colormap(cmap, minval=0.0, maxval=1.0, minColor=0.0, maxColor=1.0, n=-1):
		if n == -1:
			n = cmap.N
	
		new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
			 'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
			 cmap(np.linspace(minval, maxval, n)))
		return new_cmap
	
	cmap_t = truncate_colormap(cmap, minColor, maxColor)
	
	return cmap_t
	

def do_log(axes, which='both'):
	set_log(axes, which)


def set_log(axes, which='both'):
	
	axes = [axes] if not check_if_list(axes) else axes
	
	for ax in axes:
		if which == 'x' or which == 'both':
			ax.set_xscale('log')
		if which == 'y' or which == 'both':
			ax.set_yscale('log')


def multiple_lines(axes, vals, which='hline', **kwargs):
	
	axes = [axes] if not check_if_list(axes) else axes
	
	for ax in axes:
		for v in vals:
			if which == 'hline':	
				ax.axhline(v, **kwargs)
			else:
				ax.axvline(v, **kwargs)
	

def sample_colormap(N, low=0.20):
	return np.linspace(low, 1.0-low, N)


##########################################################################################################
# Working with confidence intervals and the like


def value_and_error_to_string(value, error_low=0.0, error_high=0.0, dec=2):
    
	# No errors are passed
	if (error_low == 0.0) and (error_high == 0.0):
		return r'$'+form(value,dec)+r'$'

	# Check whether only a lower error is passed
	if error_high > 0:
		return r'$'+form(value,dec)+'_{-'+form(error_low,dec)+'}^{+'+form(error_high,dec)+r'}$'
	else:
		return r'$'+form(value,dec)+r'\pm'+form(error_low,dec)+r'$'


def confidence_interval_2D(data, percent=[16,84]):
	"""
	Description:
	Typically, we use this function when plotting a shaded confidence area after MCMC-sampling.
	Given an NxM array, where N is the range for plotting (xrng) and M the samples, we determine
	the median for each x in xrng, and the corresponding percentile.
	
	NB: we do not calculate the ConfInt around the median, but order the full data array!
	-------
	Returns:
	Confidence "line" <myvals>. For percent=[16,84], you can use:
	>>> low, high = confidence_interval_2D(data, percent=[16,84])
	"""
	
	data = np.sort(data, axis=1)
	Nrows = len(data)
	
	myvals = np.zeros( (len(percent), Nrows) )
	for i, p in enumerate(percent):
		
		ind = int(len(data[0])*float(p)/100. + 0.5)
		myvals[i] = data[:,ind]
		
	return myvals


def mcmc_confidence_interval_to_uncertainties(confint):
	"""
	Uses the output of <mcmc_confidence_interval( )> and gives the uncertainties, assuming the
	percentiles are [p1, 50, p2].
	"""
	
	assert np.shape(confint)[1] == 3
	
	err_lo = confint[:,1] - confint[:,0]
	err_hi = confint[:,2] - confint[:,1]
	return np.asarray([err_lo, err_hi])


def print_confidence_interval(median, error, dec=2, text=None, delimiter='\n'):
	if text is None:
		text = "Median + error:\n"
	print(f"{text}{np.round(median,dec)}"+delimiter+f"{np.round(error,dec)}")


def name_and_save(plotfile, **kwargs):
	plt.savefig(plotfile, dpi=kwargs.get('dpi', 600), \
						  facecolor=kwargs.get('facecolor', 'white'), \
						  transparent=kwargs.get('transparent', False))
	print("Saved figure", plotfile)	

def name_and_write(tbdata, outfile, **kwargs):
	tbdata.write(outfile, overwrite=kwargs.get('overwrite', False))
	print("Wrote table", outfile)	

