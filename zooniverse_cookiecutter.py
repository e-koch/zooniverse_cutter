# -*- coding: utf-8 -*-
"""
@author: Elizabeth Jayne Watkins

TODOS:
    [ ] - Test all the methods work;
    [ ] - More functionality for setting filenames for saves;
    [ ] - Fill in doc strings;
    [ ] - Functionality for flipping and rotating images;
    [ ] - If only 1 band is used, need functionality for using colormaps
          (i.e., viridis etc)
    [ ] - Change method variable names so that users know ra and dec
          will not be ra and dec if the header uses a differen co-ordinate
          system. For example, if the header is in Glon glat, functions
          that are called "get_ra_dec" will be in glon glat. No behaviour
          needs changing, just the variable names to make them more accurate.
          However, the function `get_box_ds9_region` does need to be fixed
          to accept different wcs. Currently assumes all coordinates given
          are in RA and DEC.
"""

import functools
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs
from astropy.io import fits
from astropy.table import Table, QTable
from astropy.nddata import Cutout2D
from regions import RectangleSkyRegion, Regions
import reproject as rpj

import numpy as np
import matplotlib.pyplot as plt

import multicolorfits as mcf #Slightly modified version of this package

from astropy.nddata.utils import NoOverlapError

import copy


def get_fits_file_info(fits_file_name, extention=0, *header_info) :
    """Gets the image and header (or just some values from the header)

    Parameters
    ----------
    fits_file_name : Str
        Filename and path of the fits file
    extention : int
        Which HDU extention contains the data and header
    *header_info : str
        Which header information to pull and output

    Returns:
    --------
    data : numpy.ndarray
        The data contained within the fits file
    header_info_array : astropy.io.Header.header or list
        If no header_info requested, entire header is returned else
        a list containing the header info requested is returned

    """
    with fits.open(fits_file_name) as hdulist:     # importing fits file

        data = hdulist[extention].data
        #print(repr(hdulist[0].header))
        if header_info == ():
            header_info_array = hdulist[extention].header

        else:
            header_info_array = []
            for i in range(len(header_info)):
                header_info_array.append( hdulist[extention].header[header_info[i]])
        hdulist.close()

    return data, header_info_array

def get_perc_nan(image):
    """Finds out how much of am array is filled with NaNs

    Parameters
    ----------
    image : numpy.ndarray
        An array (2d expected buy not limited to 2d) that might contain NaNs

    Returns
    -------
    perc : float
        Percentage of NaNs in `image`

    """
    nans = np.isnan(image)
    perc = len(image[nans])/len(image.flatten()) * 100

    return perc


def zooniverse_cutout_filename_generator(center_ra_dec,
                                         boxsize_arcmin,
                                         band=None):
    """
    The filename string template used when saving data, and outputs that
    contain a centre, size and additional string information in band


    Parameters
    ----------
    center_ra_dec : 2-list-like floats
        Two coordinate values in RA and DEC decimals.
    boxsize_arcmin : astropy.units
        Size of the object in arcmin units.
    band : str, optional
        Additional information to add to the string, such as the band,
        or the datatype. The default is None.

    Returns
    -------
    filename : string
        A sting combining the parameters into a filename

    """
    template = '%.4f_%.4f_%.1farcmin_%.1farcmin'

    filename = template % (center_ra_dec[0], center_ra_dec[1], boxsize_arcmin[0].value, boxsize_arcmin[1].value)

    if band is not None:
        filename = band + '_' + filename

    return filename

def update_hdu(hdu, data, header):
    """Updates the data and header in a hdu.

    WARNING. Memory points at the same reference so
    if you want to preserve the origional hdu, enter
    hdu.copy() (or at some point this can be added to this function)?


    Parameters
    ----------
    hdu : astropy.io.fits.hdu.image.PrimaryHDU
        hdu object that needs the data and header updating
    data : numpy.ndarray
        Data to replace in `hdu`.
    header : astropy.io.Header.header
        Header object to replace header in `hdu`

    Returns
    -------
    hdu : astropy.io.fits.hdu.image.PrimaryHDU
        hdu object that has its data and header updated

    """
    hdu = hdu.copy()
    hdu.data = data
    hdu.header.update(header)

    return hdu

def save_cutout_as_fits(name_of_file, cutout_obj, hdu_obj_orig, file_type='.fits'):
    """This function saves a astropy.nddata.Cutout2d object to a fits file.

    Parameters
    ----------
    name_of_file : string
        The filename that will be used to save the cutout.
    cutout_obj : astropy.nddata.Cutout2d
        Cutout object to be saved
    hdu_obj_orig : astropy.io.fits.hdu.image.PrimaryHDU
        Acts as a template to save the cutout object.
    file_type : string, optional
        The file format. The default is '.fits'.

    Returns
    -------
    None.

    """
    # cutout_data, cutout_header = get_cutout_data_header(cutout)

    hdu_obj_orig = update_hdu(hdu_obj_orig, *get_cutout_data_header(cutout_obj))
    hdu_obj_orig.header.update(cutout_header)

    if name_of_file[-5:] != file_type:
        name_of_file += file_type

    save_fitsfile(name_of_file, hdu_obj_orig)

def save_fitsfile(name_of_file, hdu, end='.fits'):
    """Using the filepath+filename and hdu object, saves a fits file.

    Parameters
    ----------
    name_of_file : string
        The filename that will be used to save the hdu object
    hdu : astropy.io.fits.hdu.image.PrimaryHDU
        hdu that will be saved.
    end : string, optional
        The file format. The default is '.fits'.

    Returns
    -------
    None.

    """
    if name_of_file[-5:] != end:
        name_of_file = name_of_file + end
    hdu.writeto(name_of_file, overwrite=True)


def get_cutout_data_header(cutout):
    """Gets the data and header from an astropy.nddata.Cutout2d object

    Parameters
    ----------
    cutout : astropy.nddata.Cutout2d
        Cutout object containing the data and its header.

    Returns
    -------
    cutout_data : numpy.ndarray
        Array containing the cutout data.
    cutout_header : astropy.io.Header.header
        Header of the cutout data.

    """
    cutout_data = cutout.data
    cutout_header = cutout.wcs.to_header()

    return cutout_data, cutout_header


def half_step_overlap_grid(image_shape, initial_grid_pixel):
    """Generates a grid for making cutouts over a larger image in pixel
    coordinates. Each cutout is made in a grid that overlaps by 50%.
    The output is the central point needed to make the cutout in indices.

    Parameters
    ----------
    image_shape : 2-list-like ints
        The number of rows (y) and columns (x) in a 2d array.
    initial_grid_pixel : 2-list-like ints
        The size of the cutout [rows (y) and columns (x)] used to make the
        grid. The x and y shape of each cutout can be different (i.e.,
        rectangle cutouts are posible)

    Returns
    -------
    x_inds : numpy.1darray
        List containing the central x (column) indices of the cutout grid
    y_inds : numpy.1darray
        List containing the central y (row) indices of the cutout grid.

    """
    # if image_shape[0] % initial_grid[0] !=0 or image_shape[1] % initial_grid[1] !=0:
    #     print('Grid does not divide equally into one of the axis')
    # else:

    # indicies = []
    y_step= initial_grid_pixel[0]/2

    y_inds = np.arange(y_step, image_shape[0], y_step, dtype=int)

    x_step= initial_grid_pixel[1]/2
    x_inds = np.arange(x_step, image_shape[1], x_step, dtype=int)

    return x_inds, y_inds


def convert_pixel2world(paired_pixel_coordinates, header):
    """Load the WCS information from a fits header, and use it
    to convert pixel coordinates to world coordinates.

    Parameters
    ----------
    paired_pixel_coordinates : list/array of 2-list-like floats
        2d array where column 1 are the x coordinates/indices and
        column 2 are the y coordinates/indices that need converting to
        world coordinates
    header : astropy.io.Header.header
        Header containing the world information to convert the pixel
        coordinates into world coordinates.

    Returns
    -------
    paired_world_coordinates : 2n array
        2n array containing world coordinates

    """
    w = wcs.WCS(header)
    paired_world_coordinates = w.all_pix2world(paired_pixel_coordinates, 0)
    return paired_world_coordinates

def convert_world2pixel(paired_world_coordinates, header):
    """Load the WCS information from a fits header, and use it
    to convert world coordinates to pixel coordinates.

    Parameters
    ----------
    paired_world_coordinates : list/array of 2-list-like floats
        2d array where column 1 are the x world coordinates and
        column 2 is the y world coordinates that need converting to
        pixel coordinates
    header : astropy.io.Header.header
        Header containing the world information to convert the pixel
        coordinates into world coordinates.

    Returns
    -------
    paired_pixel_coordinates : 2n array
        2n array containing pixel coordinates

    """
    w = wcs.WCS(header)
    paired_pixel_coordinates= w.all_world2pix(paired_world_coordinates, 0)
    return paired_pixel_coordinates

def get_pixelsize_hdu(hdu):
    """Gets the pixel size (in degrees) from a hdu

    Parameters
    ----------
    hdu : astropy.io.fits.hdu.image.PrimaryHDU
        Hdu ibject that contains a header needed to find pixel size
        of the data.

    Returns
    -------
    pixelsize : float
        The pixel size (in degrees) of the data.

    """
    pixelsize = get_pixelsize_header(hdu.header)
    return pixelsize

def get_pixelsize_header(header):
    """Gets the pixel size (in degrees) from a header

    Parameters
    ----------
    header : astropy.io.Header.header
        Header object that contains the pixel size information

    Returns
    -------
    pixelsize : float
        The pixel size (in degrees) of the data.

    """
    try:
        pixelsize = header['CD2_2']
    except KeyError:
        pixelsize = header['CDELT2']
    return pixelsize

def where_highest_res_list(hdus):
    """    Finds which hdu/fits image from a 2d array of hdus
    (per row) is at the highest pixel resolution. Used to find which
    present image to use for reprojection

    Parameters
    ----------
    hdus : list of astropy.io.fits.hdu.image.PrimaryHDU
        List of hdus.

    Returns
    -------
    highest_res_pos : int
        Index of the list that contain the hdu that has the highest
        pixel resolution.

    """
    row, col = np.shape(hdus)
    res = np.zeros([row, col])
    for i in range(row):
        for j in range(col):
            res[i,j] = get_pixelsize_hdu(hdus[i,j])
    highest_res_pos = np.nanargmin(res, axis=1)
    return highest_res_pos


def make_header_3d(header, third_axis_length=3):
    """Makes a 2d header 3d.

    Parameters
    ----------
    header : astropy.io.Header.header
        2d header to be made 3d.
    third_axis_length : int, optional
        The number of images in the third dimension

    Returns
    -------
    header : astropy.io.Header.header
        Header now with the third dimension added.

    """
    header['NAXIS'] = 3
    header['NAXIS3'] = third_axis_length
    header['CDELT3'] = 1
    header['CRPIX3'] = 1
    header['CRVAL3'] = 1
    return header

def get_box_ds9_region(center_ra_dec_deg, size_wid_hei_arcmin, angle_deg=0):
    """Turns the cutout gridding into box ds9 regions.

    Parameters
    ----------
    center_ra_dec_deg : 2-list of float
        Centre of a cutout in ra and dec in degrees.
    size_wid_hei_arcmin : 2-list of float
        Size of the cutout box in width and height.
    angle_deg : float, optional
        The rotation of the box in degrees. Positive angles are anticlockwise.
        The default is 0.

    Returns
    -------
    region_sky : regions.RectangleSkyRegion
        Rectangle region object with the coordinates of the cutout.

    """
    center_ra_dec_deg = SkyCoord(center_ra_dec_deg[0], center_ra_dec_deg[1], unit='deg', frame='fk5')

    if not isinstance(size_wid_hei_arcmin, u.quantity.Quantity):
        size_wid_hei_arcmin *= u.arcmin

    if not isinstance(angle_deg, u.quantity.Quantity):
        angle_deg *= u.deg

    # center_sky = SkyCoord(42, 43, unit='deg', frame='fk5')
    region_sky = RectangleSkyRegion(center=center_ra_dec_deg,
                                    width=size_wid_hei_arcmin[0],
                                    height=size_wid_hei_arcmin[0],
                                    angle=angle_deg
                                    )
    return region_sky

def stack_hdus(prime_cuts_hdu, secondary_cuts_hdu, where_seconary_overlap):
    """Takes lists of hdu objects and combines them into a 2d array where
   columns represent the hdu lists


    Parameters
    ----------
    prime_cuts_hdu : list of astropy.io.fits.hdu.image.PrimaryHDU
        List containing all the hdu of cutouts for the primary data.
        Primary data is what is used to define the inital cutout grid.
    secondary_cuts_hdu : list of astropy.io.fits.hdu.image.PrimaryHDU
        List containing list of all the hdu of cutouts for additional data.
        The primary cutout boundaries are used for these
    where_seconary_overlap : array of int
        Listing indices where the data arrays of the secondaries overlap with
        the primary. Might be some cutouts where the array does not overlap.

    Raises
    ------
    IndexError
        If `where_seconary_overlap` is not provided and one of the
        `secondary_cuts_hdu` lists are a different size to `prime_cuts_hdu`
        IndexError is raised. For this case, the two lists must equal in
        lenght

    Returns
    -------
    hdus : nxm numpy array
        Array where columns are each image cutout hdus and rows are the same
        cutout location for different images.

    """
    if secondary_cuts_hdu is None:
        num_sec = 0
    else:
        secondary_cuts_hdu = np.atleast_1d(secondary_cuts_hdu)
        num_sec = len(secondary_cuts_hdu)

    hdus = np.zeros([len(prime_cuts_hdu),1+num_sec], dtype=object) * np.nan
    hdus[:,0] = prime_cuts_hdu
    for i in range(num_sec):
        if where_seconary_overlap is None:
            if len(prime_cuts_hdu) == len(secondary_cuts_hdu[i]):
                hdus[:,i+1] = secondary_cuts_hdu[i]

            else:
                raise IndexError('`secondary_cuts_hdu` is not the '\
                                 'same length as `prime_cuts_hdu` and '\
                                 '`where_seconary_overlap` is `None`')
        else:
            if len(where_seconary_overlap[i]) == len(secondary_cuts_hdu[i]):

                hdus[:,i+1] = secondary_cuts_hdu[i]
            elif len(prime_cuts_hdu) == len(secondary_cuts_hdu[i]):
                hdus[where_seconary_overlap[i],i+1] = secondary_cuts_hdu[i]
            else:
                raise IndexError('Where overlaps between primary and secondary '\
                                 'are incompatible')
    return hdus

def save_image(filename, arr, format='png', **kwargs):
    """wrapper for plt.imsave since the variable `format` doesn't
    automatically work. Function takes an array and saves it as an image
    using the given `format`

    Parameters
    ----------
    filename : str
        Name (including filepath) of the data to save.
    arr : numpy.array
        Data to be saved. 2d accpeted and 3d RGB also accepted
        accepted.
    format : str, optional
        The image format method. The default is 'png'.
    **kwargs : kwargs
        kwarg parameters for `plt.imsave`.

    Returns
    -------
    None.

    """
    format = kwargs.pop('format', format)
    if '.' in filename[-5:]:
        pass
    else:
        filename += '.' + format

    plt.imsave(filename, arr, format=format, **kwargs)

def min_max_of_images(images):
    """Given a 3d image (or a list of 2d images), finds the minimum and
    maximum values

    Parameters
    ----------
    images : List or 3darray
        The images (as cubes or as a list) that are considered together
        to find their minimum and maximum values.

    Returns
    -------
    min_max : 2-list
        The minimum and maximum values for all the images provided.

    """
    try:
        images = np.atleast_3d(images)
        min_all = np.nanmin(images)
        max_all = np.nanmax(images)
    except ValueError: #Some images might not have the same 2d shape
        number_of_images = len(images)
        mins = np.zeros(number_of_images)
        maxs = np.zeros(number_of_images)
        for i in range(len(images)):
            mins[i] = np.nanmin(images[i])
            maxs[i] = np.nanmax(images[i])
        min_all = np.nanmin(mins)
        max_all = np.nanmax(maxs)

    min_max = (min_all, max_all)

    return min_max

#Decorator than removes extra redundent dimensions
def correct_dimension(func):
    def wrapper(self, *args, **kwargs):
        output = func(self, *args, **kwargs)
        output_shape = np.shape(output)
        if len(output_shape) >= 3 and output_shape[0] == 1:
            return output[0]
        else:
            return output
    return wrapper

#Some dictionaries for colours
colors_rgb = {
    'red':[255,0,0],
    'yellow':[255,255,0],
    'green':[0,255,0],
    'cyan':[0,255,255],
    'blue':[0,0,255],
    'magenta':[255,0,255],
}

colors_hex = {
    'red':'#FF0000',
    'yellow':'#FFFF00',
    'green':'#00FF00',
    'cyan':'#00FFFF',
    'blue':'#0000FF',
    'magenta':'FF00FF',
}

# colors = {'hex':colors_hex,
#           'rgb':colors_rgb}



class ColorImages(object):
    """Class for colorizing an image (or list of images). Can output specific
    colors, and combine multiple colored images together/
    TODO: option for global percentile if list of images given
    TODO: option for either individual min max values or global if `min_max`
          is `None`
    TODO: make th above todos work via `scaletype`. So 'perc-global' etc?
    """

    def __init__(self, images, rescalefn=None, min_max=None, color=None, scaletype='abs'):
        """

        Parameters
        ----------
        images : list of 2d images, or a 3d datacube where each slice in
                 an image. Single 2d images are also accepted. These are
                 the image/s that will be colorised.
        rescalefn : str, optional
            This is the scaling/stretch to colorise to. Stretches come from
            astropy.visualization. Options include:
                - 'linear':astropy.visualization.LinearStretch,
                - 'sqrt':astropy.visualization.SqrtStretch,
                - 'squared':astropy.visualization.SquaredStretch,
                - 'log':astropy.visualization.LogStretch,
                - 'power':astropy.visualization.PowerDistStretch,
                - 'sinh':astropy.visualization.SinhStretch,
                - 'asinh':astropy.visualization.AsinhStretch
            If no value is given, None is used which results in a simple
            linear stretch being used. The default is None.

        min_max : 2-list of floats, optional
            List containing the minimum (vmin) and maximum (vmax) intensity
            values which impacts the contrast. If None, and `scaletype` is
            'abs', the global minimum and maximum of the images are used.
            If None and `scaletype` is 'perc', the relative 0th and 100th
            percentiles are used per image. The default is None.
        color : str, optional
            String of common colors or any hex/rgb color. Uf no color is
            provided, the colored image is set to grey. This
            allows you to check that the scaling looks alright before
            adding color The default is None.
        scaletype : str, optional
            The unit scaling of `min_max`. For percentile values (between
            0 and 100) set to `perc`. In image units, set to `abs`. By
            setting `scaletype` to perc, you make each individual image
            scale relative to itself, rather than a global value for all
            images given. This behaviour might be preferable in some cases.
            The default is 'abs'.

        Attributes:
        ----------
        self.method_colors : list of str
            The primary additive and subtrative colors have set methods
            one can use. The colors corresponding to them are given in this
            attribute
        self.grey : rgb cube
            grayscale representation of the parameters given
        self.is_greyscale : bool
            True if no color is given, meaning only a greyscale image is
            produced.
        self._rescalefn_options : List of str
            List containing the stretches that are avalible to use

        Returns
        -------
        None.

        """
        #primary additive and subtractive colors, and greyscale
        self.method_colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'grey', 'gray']
        self._rescalefn_options = ['linear', 'sqrt', 'squared', 'log', 'power', 'sinh', 'asinh']

        if len(np.shape(images)) !=3:
            images = [images]
        self.images = images

        self.rescalefn = self._set_rescalefn(rescalefn)

        self.scaletype = scaletype
        self.min_max = self._set_min_max(min_max)

        self.colortype = self._set_colortype(color)

        #if a color is provided, the colorization of it is run automatically
        #maybe re-write so gray is preserved and diff colors used from the grey?
        #if the grey step is slow should do this.
        self.grey = self.grey()
        self.colored = self.get_color(color, self.colortype)
        if color is None:
            self.is_greyscale = True
        else:
            self.is_greyscale = False

    def grey(self):
        """Makes a greyscale version of the image/s using the stretch, and
           minimum-maximum instensity scaling provided. `abs` forces all
           images to have the same minimum and maximum value, while `perc`
           scales the percentile intensity to each induvidual image.

        Returns
        -------
        grey_image : list
            List containing all the greyscale images.

        """
        grey_image = [mcf.greyRGBize_image(image, self.rescalefn, self.scaletype, self.min_max, gamma=2.2, checkscale=False) for image in self.images]

        return grey_image

    def gray(self):
        """Makes a grayscale version of the image/s using the stretch, and
           minimum-maximum instensity scaling provided. `abs` forces all
           images to have the same minimum and maximum value, while `perc`
           scales the percentile intensity to each induvidual image.

        Returns
        -------
            List containing all the grayscale images.

        """
        return self.grey()

    def red(self):
        """Makes a pure red version of the image/s using the stretch, and
           minimum-maximum instensity scaling provided. `abs` forces all
           images to have the same minimum and maximum value, while `perc`
           scales the percentile intensity to each induvidual image.

        Returns
        -------
            List containing all the red images.

        """
        return self._color(colors_hex['red'], 'hex')

    def yellow(self):
        """Makes a pure yellow version of the image/s using the stretch, and
           minimum-maximum instensity scaling provided. `abs` forces all
           images to have the same minimum and maximum value, while `perc`
           scales the percentile intensity to each induvidual image.

        Returns
        -------
            List containing all the yellow images.

        """
        return self._color(colors_hex['yellow'], 'hex')

    def green(self):
        """Makes a pure green version of the image/s using the stretch, and
           minimum-maximum instensity scaling provided. `abs` forces all
           images to have the same minimum and maximum value, while `perc`
           scales the percentile intensity to each induvidual image.

        Returns
        -------
            List containing all the green images.

        """
        return self._color(colors_hex['green'], 'hex')

    def cyan(self):
        """Makes a pure cyan version of the image/s using the stretch, and
           minimum-maximum instensity scaling provided. `abs` forces all
           images to have the same minimum and maximum value, while `perc`
           scales the percentile intensity to each induvidual image.

        Returns
        -------
            List containing all the cyan images.

        """
        return self._color(colors_hex['cyan'], 'hex')

    def blue(self):
        """Makes a pure blue version of the image/s using the stretch, and
           minimum-maximum instensity scaling provided. `abs` forces all
           images to have the same minimum and maximum value, while `perc`
           scales the percentile intensity to each induvidual image.

        Returns
        -------
            List containing all the blue images.

        """
        return self._color(colors_hex['blue'], 'hex')

    def magenta(self):
        """Makes a pure magenta version of the image/s using the stretch, and
           minimum-maximum instensity scaling provided. `abs` forces all
           images to have the same minimum and maximum value, while `perc`
           scales the percentile intensity to each induvidual image.

        Returns
        -------
            List containing all the magenta images.

        """
        return self._color(colors_hex['magenta'], 'hex')

    @correct_dimension
    def get_color(self, color, colortype=None):
        """Makes a colorized image using the color provided

        Parameters
        ----------
        color : str or 3-list
            The color that will be applied. Can be a hex code, rgb code given as
            a 3-length list-like object or can be one of the colors listed in
            `self.method_colors`, which contain the primary additive and
            subtractive colors
        colortype : str or None, optional
            Tells the code if the `color` is a hex code etc. NB: Not needed
            anymore, but have kept in for compatibility with previous code
            version. The default is None.

        Returns
        -------
        List
            List containing the colorized images.

        """
        if colortype is None:
            colortype = self._set_colortype(color)
        return self._get_color(color, colortype)

    #If a single cutout is entered, decorator removes the redundent dimension
    @correct_dimension
    def colorized_images(self, *img_list):
        """After you have colorized images of different observations, this
        convienence method will combine them into a final RGB images.
        Each additional parameter to `img_list` should represent a new
        color. Each `img_list` can either be a 2darray, or can be a
        list of 2darrays

        Parameters
        ----------
        *img_list : list or 2darray

            Each new color is entered as a new parameter. Can either provide
            one panel each which a different color, or a list of images as
            an arg. For example, one can enter:
            colorized_images(2darray_red, 2darray_yellow, 2darray_purple)
            and the result is one final image that combines the colors from
            the three arrays. Or one can enter:
            colorized_images(
                list_of_2darrays_brown
                list_of_2darrays_orange,
                list_of_2darrays_teal,
                list_of_2darrays_lilac
            )
            and as long as each list is the same size, this routine will loop
            through the lists and combine the 0th elements (the 0th brown
            orange, teal and lilac, the 1st brown, orange, teal and lilac etc),
            then the output will be a single list containing the color
            combined images.

        Returns
        -------
        final_images : list/rgbarray
            Outputs the colorized image/s as rgb cubes.

        """
        # Getting the dimension correct
        if not isinstance(img_list[0], list):
            for i in range(len(img_list)):
                img_list[i] = [img_list[i]]

        final_images = []
        for i in range(len(img_list[0])):

            final_image = []
            for j in range(len(img_list)):
                final_image.append(img_list[j][i])
            final_images.append(mcf.combine_multicolor(final_image, gamma=2.2))

        return final_images

    def _get_color(self, color, colortype=None):

        if colortype is None:
            colortype = self._set_colortype(color)

        if color in self.method_colors:
            return getattr(self, color)()
        else:
            return self._color(color, colortype)

    def _color(self, color, colortype=None):

        if colortype is None:
            colortype = self._set_colortype(color)
        # grey_images =
        if color is None:
            cols = self.grey
        else:
            cols = [mcf.colorize_image(grey_image, color, colorintype=colortype, gammacorr_color=2.2) for grey_image in self.grey]

        return cols

    def _set_min_max(self, min_max):

        if min_max is None:
            if 'perc' in self.scaletype:
                min_max = [0,100]
            elif 'abs' in self.scaletype:
                min_max = min_max_of_images(self.images)
            else:
                raise TypeError('`scaletype` must contain `perc` for'\
                                ' percentiles or `abs` for absolute'\
                                'intensity scaling.')
        return min_max

    def _set_colortype(self, color):

        if color is None:
            colortype = 'hex'
        elif color[0] == '#':
            colortype = 'hex'
        elif color in self.method_colors:
            colortype = 'hex'
        elif len(color) == 3 and not isinstance(color, str):
            colortype = 'rgb'
        else:
            raise TypeError('`color` much be a hex code, rgb color as a 3-list-like array, one'\
                            'of the colors in `self.method_colors`, or `None`,'\
                            'which defaults to grey.')

        return colortype

    def _set_rescalefn(self, rescalefn):

        if rescalefn is None:
            rescalefn = 'linear'
        else:
            if rescalefn not in self._rescalefn_options:
                raise TypeError('Stretch provided in `rescalefn` not one of'\
                                'the accepted options', self._rescalefn_options)
        return rescalefn

class ReprojectCutouts(object):
    """Takes cutout hdus and reprojects them to the highest resolution
    image.
    Might replace this step with a reprojection of the whole image
    in the begining (once and save the output), which might speed things up.

    """

    def __init__(self, prime_cuts_hdu, secondary_cuts_hdu, where_secondary_overlap=None):
        """

        Parameters
        ----------
        prime_cuts_hdu : list of hdus
            List containing the hdus of the primary image. The primary image
            is the reference image used to generate the inital cutouts.
        secondary_cuts_hdu : list of list of hdus
            List containing the hdu lists of any other images.
        where_secondary_overlap : ndarray, optional
            Array/list containing the index values where the
            secondary images have data at the position of the primary cutout
            locations. The footprints of different observations might
            not overlap, and some cutouts will not exist as a result. This
            array provides the positions where this occurs. The default
            is None.

        Attributes
        ----------
        self.highest_res : numpy.ndarray
            Index showing which present hdu of a cutout location has the
            highest pixel resolution
        self.hdus_repr : nxm array of hdus
            column stacked hdus all with the image data and header
            reprojected to the highest pixel resolution

        Returns
        -------
        None.

        """
        hdus = stack_hdus(prime_cuts_hdu, secondary_cuts_hdu, where_secondary_overlap)
        self.highest_res = where_highest_res_list(hdus)
        self.hdus_repr = self.reproject(hdus)

    @classmethod
    def from_FitsCutter(cls, FitsCutter_obj):
        """Initalise the class using a FitsCutter object

        Parameters
        ----------
        FitsCutter_obj : zooiverse_cutter.FitsCutter
            FitsCutter object which should contain all the required
            information to initalise this class

        Returns
        -------
        None.

        """
        prime_hdus, secondaries_hdus = [FitsCutter_obj.get_cutout_hdus()]
        where_seconary_overlaps = FitsCutter_obj.where_secondary_overlap

        cls(prime_hdus, secondaries_hdus, where_secondary_overlap)

    @classmethod
    def from_full_files(cls, primary_filename, gridding, prime_color_params, secondary_filenames=None, hdu_index_primary=0, hdu_indices_secondaries=None, grid_method=half_step_overlap_grid, secondary_colors_params=None):
        """Initalise the class from fit files of the full images that will
        be reprojected after being made into cutouts

        Parameters
        ----------
        primary_filename : str
            Filename and path to the primary image.
        gridding : 2-list
            List (in arcmin) of the cutout box size.
        prime_color_params : dict
            The colorization parameters to apply to the primary image.
        secondary_filenames : List of str, optional
            Filename and path to the seconday images. The default is None.
        hdu_index_primary : int, optional
            Index where hdu contains the data and header. The default is 0.
        hdu_indices_secondaries : List of ints, optional
            Indices where secondary hdus contains the data and header.
            The default is None.
        grid_method : function, optional
            The function used to generate the cutout grid.
            The default is `half_step_overlap_grid`.
            half_step_overlap_grid overlaps each cutout 50% with its neigbours
        secondary_colors_params : List of dict, optional
            The colorization parameters to apply to the secondary images.
            The default is None.

        """
        return ReprojectCutouts.from_FitsCutter(FitsCutter.from_fits(primary_filename, gridding, secondary_filenames, hdu_index_primary, hdu_indices_secondaries, grid_method))


    def reproject(self, stacked_hdus, method=rpj.reproject_interp):
        """Takes a column stacked list of cutout hdu's, finds which has
        the highest pixel resolution, and reprojects the remaining images
        to the one with the highest pixel resolution. Function takes
        any missing cutouts whose footprints did not allow for a cutout
        to be made, meaning some reprojections might be at a lower resolution
        (if the missing cutout was the one with the highest pixel resolution).

        Parameters
        ----------
        stacked_hdus : nxm numpy array
            Array where columns are each image cutout hdus and rows are the
            same cutout location for different images.
        method : function, optional
            The method of reprojection. The default is rpj.reproject_interp.

        Returns
        -------
        stacked_hdus : Array where columns are each image cutout hdus and rows
                       are the same cutout location for different images.
                       These are now reprojected to the highest pixel
                       resolution per row
        """
        # high_hdu = stacked_hdus[:, self.highest_res]
        # other_hdu = np.delete(stacked_hdus, self.highest_res)
        num_cutouts, num_hdu = np.shape(stacked_hdus)

        for i in range(num_cutouts):

            res_ind = self.highest_res[i]
            cutout_row = stacked_hdus[i]

            highest_res_cutout_hdu = cutout_row[res_ind]
            highest_res_header = highest_res_cutout_hdu.header

            inds_update = np.delete(range(num_hdu), res_ind)
            imgs = []
            for n in inds_update:
                # try:
                if cutout_row[n] is None:
                    img = highest_res_cutout_hdu.data * np.nan
                else:
                    img = method(cutout_row[n], highest_res_header)[0]
                # except TypeError:
                #     img = highest_res_cutout_hdu.data * np.nan

                stacked_hdus[i,n] = update_hdu(copy.copy(highest_res_cutout_hdu.copy()), img, highest_res_header)
            # print(img)
        return stacked_hdus

class ColoredCutouts(object):
    """Takes in hdus of images (usually from cutouts), preps them for
    colorization by reprojection, and applies the colors using the params
    given. This is used as an intermediate class. If can be run alone,
    but all the saving options were easier to keep in
    `zooiverse_cutter.FitsCutter`. Better to run this class through
    zooiverse_cutter.FitsCutter.
    """

    def __init__(self, prime_cuts_hdu, prime_color_params, secondary_cuts_hdu=None, secondary_colors_params=None, where_secondary_overlap=None, run_reproject=True):
        """

        Parameters
        ----------
        prime_cuts_hdu : list of hdus
            List containing the hdus of the primary image. The primary image
            is the reference image used to generate the inital cutouts.
        prime_color_params : dict
            Dictionary containing the parameters needed to colorize the
            images. This include:
                1. 'rescalefn', the stretch, which can be:
                    - 'linear'
                    - 'sqrt'
                    - 'squared'
                    - 'log'
                    - 'power'
                    - 'sinh'
                    - 'asinh'
                If no 'rescalefn' given, or if set to None, the default
                is used, which is a 'linear' stretch
                2. 'min_max', the minimum and maximum intensity values to show.
                   If no value is given, or if None is given, the minimum
                   and maximum values are used.

                3. 'scaletype' the scaling used for 'min_max'. 'abs' uses the
                   absolute units of the image itself. 'perc' uses image
                   percentiles. 'perc' works on a per image basis, rather
                   than the global percentile value. This means that image
                   contrasts will look similar between themselves
                   but the `min_max` used for each image will differ.
                   If None/no value is entered for 'scaletype', 'abs' is used.
                 4. 'color', the color of the image/s. Either enter one of
                    the primary additve/subtractive colors as a string,
                    enter a hex string, or a 3-list of ints of rgb colors.
                    If no color/None is provided, greyscale is used.

        secondary_cuts_hdu : list of list of hdus
            List containing the hdu lists of any other images. The default
            is None.
        secondary_colors_params : List of dict, optional
            The colorization parameters to apply to the secondary images.
            The default is None.
        where_secondary_overlap : ndarray, optional
            Array/list containing the index values where the
            secondary images have data at the position of the primary cutout
            locations. The footprints of different observations might
            not overlap, and some cutouts will not exist as a result. This
            array provides the positions where this occurs. The default
            is None.
        run_reproject : bool, optional
            Whether to reproject the cutouts. If they already share
            the same projection, skipping this step should be faster.
            The default is True.

        Attributes
        ----------
        self.all_params_used : list
            list tracking every colorization that has been run on the
            images.
        self.color_params : list of dict
            All the colorization params for the primary and then secondary
            image cutouts.
        self.color_params_recently_used : list of dict
            The last used colorization parameters
        self.hdus : nxm numpy.ndarray
            column stacked hdus (each row contains the cutouts at a single
            location but in different bands). If `run_reproject` is True
            the hdus are reprojected to the highest pixel resolution
        self.final_images : list of rgb cubes
            The last run that generated colorized combined false color
            images. Only exists if the method `get_colored_arrays` has
            been run. Potentially saves time by avoiding repeats

        Returns
        -------
        None.

        """
        if run_reproject:
            self.hdus = ReprojectCutouts(prime_cuts_hdu, secondary_cuts_hdu, where_secondary_overlap=where_secondary_overlap).hdus_repr
        else:
            self.hdus = stack_hdus(prime_cuts_hdu, secondary_cuts_hdu, where_secondary_overlap)


        if secondary_colors_params is not None:
            secondary_cuts_hdu = np.atleast_1d(secondary_colors_params)
            self.color_params = [prime_color_params, *secondary_colors_params]
        else:
            self.color_params = [prime_color_params]

        #updates to last colors asked to be output from `get_col_obj` and therefore 'get_colored_arrays
        self.color_params_recently_used = self.color_params
        self.all_params_used = [self.color_params_recently_used]

        #self.col_objs = self.get_col_obj()

    @classmethod
    def from_filelist(cls):
        pass

    @classmethod
    def from_FitsCutter(cls, FitsCutter_obj, prime_color_params, secondary_colors_params=None, run_reproject=True):
        """Initalise the class using zooiverse_cutter.FitsCutter object.
        This object contains all the cutout information, and is just missing
        the colorization parameters

        Parameters
        ----------
        FitsCutter_obj : zooiverse_cutter.FitsCutter
            The main object that contains the cutout information, gridded etc.
        prime_color_params : dict
            The colorization parameters to apply to the primary image.
        secondary_colors_params : List of dict, optional
            The colorization parameters to apply to the secondary images.
            The default is None.
        run_reproject : bool, optional
            Whether to reproject the cutouts. If they already share
            the same projection, skipping this step should be faster.
            The default is True.

        Returns
        -------
        None.

        """
        prime_hdus, secondaries_hdus = [FitsCutter_obj.get_cutout_hdus()]
        where_seconary_overlaps = FitsCutter_obj.where_secondary_overlap

        cls(prime_hdus, prime_color_params, secondaries_hdus, secondary_colors_params, where_secondary_overlap, run_reproject)

    @classmethod
    def from_full_files(cls, primary_filename, gridding, prime_color_params, secondary_filenames=None, hdu_index_primary=0, hdu_indices_secondaries=None, grid_method=half_step_overlap_grid, secondary_colors_params=None, run_reproject=True):
        """Initialised from filenames by running zooiverse_cutter.FitsCutter
        and using that with the colorization parameters

        Parameters
        ----------
        primary_filename : str
            Filename and path to the primary image.
        gridding : 2-list
            List (in arcmin) of the cutout box size.
        prime_color_params : dict
            The colorization parameters to apply to the primary image.
        secondary_filenames : List of str, optional
            Filename and path to the seconday images. The default is None.
        hdu_index_primary : int, optional
            Index where hdu contains the data and header. The default is 0.
        hdu_indices_secondaries : List of ints, optional
            Indices where secondary hdus contains the data and header.
            The default is None.
        grid_method : function, optional
            The function used to generate the cutout grid.
            The default is `half_step_overlap_grid`.
            half_step_overlap_grid overlaps each cutout 50% with its neigbours
        secondary_colors_params : List of dict, optional
            The colorization parameters to apply to the secondary images.
            The default is None.
        run_reproject : bool, optional
            Whether to reproject the cutouts. If they already share
            the same projection, skipping this step should be faster.
            The default is True.
        """
        return ColoredCutouts.from_FitsCutter(FitsCutter.from_fits(primary_filename, gridding, secondary_filenames, hdu_index_primary, hdu_indices_secondaries, grid_method), prime_color_params, secondary_colors_params, run_reproject)

    #Not sure if catching will use up too much memory.
    #@functools.lru_cache
    def get_colored_arrays(self, color_param_list=None, colors=None):
        """
        Function organises the cutouts so that
        they can be converted into false colour images using multicolorfits.
        This is a slow function.

        Parameters
        ----------
        color_param_list : list of dictionaries, optional
            The parameters that affect the appearance of the colorized cutouts.
            The default is None.
        colors : list of str or list of arraylike, optional
            Color of the colorized images. The default is None.

        Returns
        -------
        final_images : list of rgbarray
            List containing the final combined colorised cutouts. This is a
            new attribute that is created here so that last run colorization
            is avalible to the class without re-running.
        """
        #params usually contains the color
        color_objs = self.get_col_obj(color_param_list)
        colored = []

        #Weird behaviour. Means that the color objects are not remade if a
        #different color is wanted, but instead the color is pulled
        #otherwise, the color given in `color_param_list` is used
        for i in range(len(color_objs)):
            if color_objs[i].is_greyscale and colors is not None:
                colored.append(color_objs[i].get_color(colors[i]))

            else:
                if color_objs[i].is_greyscale:
                    print('Warning: No color given so using grayscale')
                colored.append(color_objs[i].colored)

        colored_shuffled = []

        for i in range(len(colored[0])):
            colored_shuffled_column = []
            for j in range(len(color_objs)):
                if np.all(np.isnan(colored[j][i])):
                    continue #might need to append zeros for color balence instead of skipping?
                colored_shuffled_column.append(colored[j][i])
            colored_shuffled.append(colored_shuffled_column)

        colored_images = colored_shuffled
        del colored_shuffled

        #another self to help stop repeating
        self.final_images = [mcf.combine_multicolor(colored_img, gamma=2.2) for colored_img in colored_images]
        return self.final_images

    def get_stacked_images(self):
        """Gets the 2d arrays from the hdus then stacks them row wise
        into a 3d cube.

        Returns
        -------
        data3d : numpy 3darray
            Datacubes of each cutout location but stacked using different
            bands.

        """
        data3d = [self._get_img_from_hdu_list(hdu) for hdu in self.hdus]
        for i in range(len(data3d)):
            data3d[i] = np.array(data3d[i])

        return data3d

    #Not sure if catching will use up too much memory.
    #@functools.lru_cache
    def get_col_obj(self, color_param_list=None):
        """Wrapper to inialised  `zooiverse_cutterColorImages` for each
        color

        Parameters
        ----------
        color_param_list : list of dictionaries, optional
            The parameters that affect the appearance of the colorized
            cutouts. If None, then the last used parameters of
            The default is None.

        Returns
        -------
        color_objs : List of `zooiverse_cutter.ColorImages`
            List containing the colorisation objects. for the
            parameters given.

        """
        color_objs = []
        if color_param_list is None:
            color_param_list = self.color_params_recently_used

        #TODO:Might add the option here (with other code edits) so that if
        #`zooiverse_cutter.ColorImages` is saved as an attribute. Therefore
        #if it exists and the given `color_param_list` are the same as
        #the params in ColorImages, it is returned without running anything.
        # elif len(color_objs) > 0 and color_param_list == self.color_params_recently_used[-1]:
        #     #something here so analysis isn't rerun

        else:
            self.all_params_used.append(color_param_list)
            self.color_params_recently_used = color_param_list

        for i in range(np.shape(self.hdus)[1]):

            data2d = self._get_img_from_hdu_list(self.hdus[:,i])

            param = color_param_list[i]

            #color_objs.append(ColorImages(data2d, rescalefn=param['rescalefn'], min_max=param['min_max'], color=param['color'], scaletype=param['scaletype']))
            color_objs.append(ColorImages(data2d, **param))

        return color_objs


    def _get_img_from_hdu_list(self, hdus):
        """Loops through a hdu list to output the image data
        """
        return [hdu.data for hdu in hdus]

    def save_colored_cutout_as_imagefile(self, format='png'):
        #Moved to FitsCutter
        pass


class CutoutObjects(object):
    """This class is the initial class that makes a grid and makes cutouts.
    It is composed into zooiverse_cutter.FitsCutter so that when
    multiple images are given, their induvidual properties are stored
    seperatly in an object, and allows for more control.
    `CutoutObjects` creates cutouts of a fits file according to a grid method.
    The default is a 50% overlap grid. Currently no user option to change
    the percentage.
    ra and dec in variable names and methods actually output in the wcs
    found in the header. So if the header is in glon, glat, the wcs outputed
    will be in these units NOT ra and dec.
    """
    @u.quantity_input(gridding=u.arcmin)
    def __init__(self, hdu, gridding, grid_method=half_step_overlap_grid, remove_only_nans=False):
        """

        Parameters
        ----------
        hdu : astropy.io.fits.hdu.image.PrimaryHDU
            HDU of a full sized image that will be turned into cutouts
        gridding : 2-list
            List (in arcmin) of the cutout box size.
        grid_method : function, optional
            The function used to generate the cutout grid.
            The default is `half_step_overlap_grid`.
            half_step_overlap_grid overlaps each cutout 50% with its neigbours
        remove_only_nans : bool, optional
            Option to remove cutouts entirely if they only contain
            NaN values. The default is False.

        Attributes
        ----------
        self.cutout_centers_pixel : list of 2-list
            The cutout centers for the gridding.
        self.gridding_odd_arcmin : : 2-list
            The width and height of each box adjusted so that when converted
            to pixel units, they are an odd number of pixels
        self.gridding_odd_pixel : : 2-list
            The width and height of each box adjusted so that in pixel
            units, they are an odd number of pixels.
        self.nan_inds : List
            List containing the indices where the cutout is entirely made
            out of nans

        Returns
        -------
        None.

        """
        self.nan_inds = []
        self.remove_only_nans = remove_only_nans
        self.hdu = hdu
        self.gridding = gridding
        self.grid_method = grid_method
        self.cutout_centers_pixel, self.gridding_odd_pixel = self.get_centers(return_pixel_gridding=True)
        if self.remove_only_nans:
            self.nan_inds = self._nan_inds()
            self.cutout_centers_pixel = self._remove_only_nans(self.cutout_centers_pixel)


        pixelsize = get_pixelsize_hdu(self.hdu)

        self.gridding_odd_arcmin = self.gridding_odd_pixel * pixelsize * 60 * u.arcmin

    @classmethod
    def from_fits(cls, fits_filename, gridding, hdu_index=0, grid_method=half_step_overlap_grid):
        """

        Parameters
        ----------
        fits_filename : str
            Filename and path to the image.
        gridding : 2-list
            List (in arcmin) of the cutout box size.
        hdu_index : int, optional
            Index where hdu contains the data and header. The default is 0.
        grid_method : function, optional
            The function used to generate the cutout grid.
            The default is `half_step_overlap_grid`.
            half_step_overlap_grid overlaps each cutout 50% with its neigbours.
        Returns
        -------

        """
        # with fits.open(fits_filename) as hdulist:
        #     hdu = hdulist[hdu_index]
        hdu = fits.open(fits_filename)[hdu_index]

        return cls(hdu=hdu, gridding=gridding, grid_method=grid_method)

    def _remove_only_nans(self, param):

        param = np.delete(param, self.nan_inds, axis=0)
        return param

    def _nan_inds(self):
        percs   = self.percentage_of_nans()
        nan_inds = np.where(percs>50)[0]
        #self.nan_inds = nan_inds #only way for composition later
        return nan_inds

    def percentage_of_nans(self, cutout_objs=None, cutout_centers=None, gridding_odd=None):
        """Finds the percentage of NaNs contained in each image

        Parameters
        ----------
        cutout_objs : List of astropy.nddata.Cutout2d, optional
            List containing all the cutout objects. If no cutout object is
            provided, cutout objects are generated using `cutout_centers` and
            `gridding_odd`. Providing these allows a different grid to be
            imposed (this behaviour is wanted for imposing the primary grid
            onto secondary images for example. The default is None.
        cutout_centers : list of 2-list of ints, optional
            The cutout centers from the gridding. The default is None.
        gridding_odd : 2-list of ints, optional
            The gridding length and width, in pixel units that are odd,
            needed to generate the cutout of the desired length. The default
            is None.

        Returns
        -------
        percs : numpy.ndarray
            Array containing the percentage (0-100%) of pixels that contain
            NaNs in each cutout.

        """
        if cutout_objs is None:
            cutout_objs = self.get_cutouts(cutout_centers, gridding_odd)
        percs = np.zeros(len(cutout_objs))

        for i in range(len(cutout_objs)):
            image = cutout_objs[i].data
            percs[i] = get_perc_nan(image)

        return percs

    def get_centers(self, grid_method=None, return_pixel_gridding=False):
        """Method finds the xy indicies for tile/cutout centers


        Parameters
        ----------
        grid_method : function, optional
            Function needed to generate the grid parameters. The default
            is None. If None is given, the initalised grid method
            (`self.grid_method`) is used
        return_pixel_gridding : bool, optional
            Controls whether the function will return the width & height of
            all cutouts as a 2-list (adjusted so that they are odd).
            The default is False.

        Returns
        -------
        centers_xy : list of 2-list of ints
            The cutout centers from the gridding.

        gridding_xy_pixel_odd.value : 2-list of ints, optional
            The width and height that cutouts will have in pixel units.
            Will only output if `return_pixel_gridding` is set to True

        """
        if grid_method is None:
            grid_method = self.grid_method

        header = self.hdu.header

        gridding_xy_pixel_odd  = self._get_lens(header, self.gridding)
        x_indices, y_indices = grid_method((header['NAXIS2'], header['NAXIS1']), gridding_xy_pixel_odd)

        xx, yy = np.meshgrid(x_indices, y_indices)
        centers_xy = np.column_stack([xx.ravel(), yy.ravel()])
        self._remove_only_nans(centers_xy)

        if return_pixel_gridding:
            return centers_xy, gridding_xy_pixel_odd.value
        else:
            return centers_xy


    def _get_lens(self, header, gridding):
        pixel_size = get_pixelsize_header(header) * 60 * u.arcmin

        xygrid = gridding/pixel_size

        xygrid = np.round(xygrid)

        #turns grid odd. Odd is better for gridding
        if xygrid[0] % 2 !=0:
            xygrid[0] +=1
        if xygrid[1] % 2 !=0:
            xygrid[1] +=1

        return xygrid

    #Not sure if catching will use up too much memory. Also seems quick so not needed?
    #@functools.lru_cache
    def get_cutouts(self, cutout_centers=None, gridding_odd=None, cutout_centers_ra_dec=None):
        """Given a list of cutout centers and a pixel gridding, will
        generate astropy.nddata.Cutout2d objects matching the gridding.
        if no centers or gridding provided, uses the auto-generated grid.
        The behaviour wanted depends on how it is being used. For example,
        if a primary grid has been generated, and the same cutouts need
        to be made in secondary images at the primary grid parameter
        locations, can provide cutout centers of the primary in wcs
        coordinates, forcing the cutouts to be made there instead.

        Parameters
        ----------
        cutout_centers : list of 2-list of ints, optional
            The cutout centers that are used to make the cutouts. The default
            is None. None results in the object default gridding to be used
            (`self.cutout_centers_pixel`)
        gridding_odd : 2-list of ints, optional
            The gridding length and width, in pixel units that are odd,
            needed to generate the cutout of the desired length. The default
            is None. None results in the object default lengths being used
            (`self.gridding_odd_pixel`)
        cutout_centers_ra_dec : list of 2-list of floats, optional
            The cutout centers that are used to make the cutouts in wcs
            coordinates. The default is None.

        Returns
        -------
        cutout_objs : List of astropy.nddata.Cutout2d
            List containing the cutout objects generated using the parameters
            given.

        """
        if cutout_centers is None:
            cutout_centers = self.cutout_centers_pixel

        if cutout_centers_ra_dec is not None:
            cutout_centers = self.get_xy_centers_from_ra_dec(cutout_centers_ra_dec)


        if gridding_odd is None:
            gridding_odd = self.gridding_odd_pixel

        cutout_objs = []
        for cent in cutout_centers:
            try:
                cutout_objs.append(Cutout2D(self.hdu.data, position=cent,
                                          size=gridding_odd,
                                          wcs=wcs.WCS(self.hdu.header),
                                          mode='partial'))
            except NoOverlapError:
            # pass
                cutout_objs.append(None)


        return cutout_objs

    def get_centers_wsc(self, cutout_centers=None, hdu=None):
        """Converts the pixel center locations to WCS coordinates provided in
        the header (NOT to RA and DEC if the header is in galactic
        coordinates, for example)
        If no cutout centers given, the default cutout centers are used
        (`self.cutout_centers_pixel`)
        """
        if cutout_centers is None:
            cutout_centers = self.cutout_centers_pixel
        if hdu is None:
            hdu = self.hdu
        return convert_pixel2world(cutout_centers, hdu.header)

    def get_xy_centers_from_ra_dec(self, cutout_centers_ra_dec):
        """Converts WCS coordinates into pixel coordinates.
        """
        return convert_world2pixel(cutout_centers_ra_dec, self.hdu.header)


    def get_cutout_properties(self, cutout_centers_ra_dec=None):
        """Origonally designed to output summary values for tabulation.
        Outputs the cutout locations in pixel coordinates and in WCS
        coordinates and how many NaNs each cutout has
        as a percentage.
        If `cutout_centers_ra_dec` (i.e., wcs) are provided, these coordinates are used
        and are converted to pixel, otherwise the default cutout centers
        initalised with the object are used.

        Parameters
        ----------
        cutout_centers_ra_dec : nx2 numpy.ndarray, optional
            Cutout centers in WCS coordinates. The default is None.

        Returns
        -------
        xy_cutout_centers : list of 2-list of ints
            The cutout centers that are used to make the cutouts
        cutout_centers_ra_dec : list of 2-list of floats
            The cutout centers that are used to make the cutouts in wcs
            coordinates.
        perc_nan_in_cutout : numpy.ndarray
            Array containing the percentage (0-100%) of pixels that contain
            NaNs in each cutout.

        """
        if cutout_centers_ra_dec is not None:

            xy_cutout_centers = self.get_xy_centers_from_ra_dec(cutout_centers_ra_dec)
        else:
            xy_cutout_centers = self.cutout_centers_pixel
            cutout_centers_ra_dec = self.get_centers_wsc()

        perc_nan_in_cutout = self.percentage_of_nans()

        return xy_cutout_centers, cutout_centers_ra_dec, perc_nan_in_cutout


    def get_cutouts_footprint_ds9_region(self, ra_dec_cutout_centers=None):
        """Converts the cutout grid into ds9 box regions
        WARNING: Currently only works correctly if the image is in RA
        and DEC.

        Parameters
        ----------
        ra_dec_cutout_centers : list of 2-list of floats, optional
            The cutout centers that are used to make the cutouts in RA and
            dec. The default is None.

        Returns
        -------
        box_regions : List of regions.RectangleSkyRegion
            ds9 box regions.

        """
        if ra_dec_cutout_centers is None:
            ra_dec_cutout_centers = self.get_centers_wsc()

        box_regions = [get_box_ds9_region(ra_dec, self.gridding_odd_arcmin) for ra_dec in ra_dec_cutout_centers]

        return box_regions


    def get_cutout_hdus(self, cutout_centers_ra_dec=None):
        cutouts_obj = self.get_cutouts(cutout_centers_ra_dec=cutout_centers_ra_dec)


        hdu_cutouts = []
        for i in range(len(cutouts_obj)):
            data, header = get_cutout_data_header(cutouts_obj[i])
            hdu_cutouts.append(update_hdu(copy.copy(self.hdu.copy()), data, header))

        return hdu_cutouts


    def save_fits_cutouts(self, path, band, ra_dec_cutout_centers=None):

        if ra_dec_cutout_centers is None:
            cutout_centers_ra_dec = self.get_centers_wsc()

        cutout_objs = self.get_cutouts(cutout_centers_ra_dec=cutout_centers_ra_dec)
        self._save(save_cutout_as_fits, cutout_objs, ra_dec_cutout_centers, path, band, hdu_obj_orig=copy.copy(self.hdu.copy()))

        return ra_dec_cutout_centers

    def save_as_ds9regions(self, path, ra_dec_cutout_centers=None):

        if ra_dec_cutout_centers is None:
            ra_dec_cutout_centers = self.get_centers_wsc()

        box_regions = self.get_cutouts_footprint_ds9_region(ra_dec_cutout_centers)
        savepath = os.path.join(path, 'cutout_regions_%.1f_%.1farcmin.reg' % tuple(self.gridding.value))
        Regions(box_regions).write(savepath, overwrite=True)

        return ra_dec_cutout_centers

    def _save(self, save_method, objs, ra_dec_cutout_centers, path, band=None, filename_method=zooniverse_cutout_filename_generator, **kwargs):

        if ra_dec_cutout_centers is None:
            ra_dec_cutout_centers = self.get_centers_wsc()

        for i in range(len(objs)):
            filename = filename_method(ra_dec_cutout_centers[i], self.gridding, band=band)

            savepath = os.path.join(path, filename)

            save_method(savepath, objs[i], **kwargs)


#Main class to interact with the program
class FitsCutter(object):
    """This function makes cutouts for a `primary`/main image and makes
    same sized and location cutouts of and additional(`secondary`) fits
    files given.
    Currently there might be some errors if a cutout being made
    on the secondary fitsfile is outside the range of the primary

    Added colorization as methods as composition to this class since it
    was convenient to have all the gridding params for saving. This makes
    the class a bit less focused, but means this is the only class one needs
    to use to get the cutouts and save them.
    """

    # @u.quantity_input(gridding=u.arcmin)
    def __init__(self, prime_cuts, secondary_cuts=None):
        """

        Parameters
        ----------
        prime_cuts : zooiverse_cutter.CutoutObjects
            A object containing all the cutouts and their gridding
            parameters for the primary image. Primary image is used for the
            grid, and any secondary observations will use the grid
            generated for the primary.
        secondary_cuts : List of zooiverse_cutter.CutoutObjects, optional
            List containing the cutouts and their parameters for any
            additional observations. The default is None.

        Attributes
        ----------
        self.cutout_centers_ra_dec_prime : list of 2-list of floats
           The cutout centers that are used to make the cutouts in RA and
           dec.
        self.cutout_centers_ra_dec_secondaries : list of 2-list of floats
           The cutout centers that are used to make the cutouts in RA and
           dec. Can be different from `self.cutout_centers_ra_dec_prime`
           if the secondary footprint did not extend to the location
           of some cutouts in the primary
        self.gridding : 2-list
            List (in arcmin) of the cutout box size.
        self.where_secondary_overlap : list of numpy.ndarray
            List containing indices for each secondary image where the cutout
            overlapped with the primary and so a cutout could be made

        Returns
        -------
        None.

        """
        self.prime_cuts  = prime_cuts
        if not self.prime_cuts.remove_only_nans:
            self.prime_cuts = self._remove_nans(self.prime_cuts)

        self.gridding = self.prime_cuts.gridding_odd_arcmin
        self.cutout_centers_ra_dec_prime = self.prime_cuts.get_centers_wsc()

        if secondary_cuts is not None:
            self.secondary_cuts = np.atleast_1d(secondary_cuts)
            self.where_secondary_overlap = [np.where(np.array(sec_cut.get_cutouts(cutout_centers_ra_dec=self.cutout_centers_ra_dec_prime)) !=None)[0] for sec_cut in self.secondary_cuts]
            self.cutout_centers_ra_dec_secondaries = [self.cutout_centers_ra_dec_prime[inds] for inds in self.where_secondary_overlap]
        else:
            self.secondary_cuts = None

    @classmethod
    @u.quantity_input(gridding=u.arcmin)
    def from_hdu(cls, primary_hdu, gridding, secondary_hdus=None, grid_method=half_step_overlap_grid):
        """
        Initialised from hdu's which contain the data and headers in them
        Parameters
        ----------
        primary_hdu : astropy.io.fits.hdu.image.PrimaryHDU
            HDU of a full sized image that will used to generate a grid and
            then made into cutouts
        gridding : 2-list
            List (in arcmin) of the cutout box size of the cutouts
        secondary_hdus : list of astropy.io.fits.hdu.image.PrimaryHDU
            list of HDU of a full sized image that will be turned into cutouts
            following the grid generated in the primary image. The default
            is None.
        grid_method : function, optional
            Function needed to generate the grid parameters. The default
            is None. If None is given, the initalised grid method
            (`self.grid_method`) is used

        Returns
        -------

        """
        gridding = gridding.to(u.arcmin)
        prime  = CutoutObjects(primary_hdu, gridding)
        if secondary_hdus is not None:
            secondary_hdus = np.atleast_1d(secondary_hdus)
            secs  = [CutoutObjects(hdu, gridding) for hdu in secondary_hdus]
        else:
            secs = None

        return cls(prime_cuts=prime, secondary_cuts=secs)


    @classmethod
    @u.quantity_input(gridding=u.arcmin)
    def from_fits(cls, primary_filename, gridding, secondary_filenames=None, hdu_index_primary=0,
                    hdu_indices_secondaries=None, grid_method=half_step_overlap_grid):
        """
        Initialisedthe class from full sized obervations in fit files
        given the gridding parameters used.

        Parameters
        ----------
        primary_filename : str
            Filename and path to the primary image.
        gridding : 2-list
            List (in arcmin) of the cutout box size.
        secondary_filenames : List of str, optional
            Filename and path to the seconday images. The default is None.
        hdu_index_primary : int, optional
            Index where hdu contains the data and header. The default is 0.
        hdu_indices_secondaries : List of ints, optional
            Indices where secondary hdus contains the data and header.
            The default is None.
        grid_method : function, optional
            The function used to generate the cutout grid.
            The default is `half_step_overlap_grid`.
            half_step_overlap_grid overlaps each cutout 50% with its neigbours
        Returns
        -------

        """
        gridding = gridding.to(u.arcmin)
        prime = CutoutObjects.from_fits(primary_filename, gridding, hdu_index_primary, grid_method=grid_method)
        if secondary_filenames is not None:
            secondary_filenames = np.atleast_1d(secondary_filenames)
            num = len(secondary_filenames)
            if hdu_indices_secondaries is None:
                hdu_indices_secondaries = [0] * num
            hdu_indices_secondaries = np.atleast_1d(hdu_indices_secondaries)
            secs  = [CutoutObjects.from_fits(secondary_filenames[i], gridding, hdu_indices_secondaries[i], grid_method=grid_method) for i in range(num)]
        else:
            secs = None

        return cls(prime_cuts=prime, secondary_cuts=secs)

    def _remove_nans(self, cut_obj):
        cut_obj.remove_only_nans = True
        cut_obj.nan_inds = cut_obj._nan_inds()
        cut_obj.cutout_centers_pixel = cut_obj._remove_only_nans(cut_obj.cutout_centers_pixel)

        return cut_obj


    def tabulate_cutout_properties(self, path, bands):
        # this should be the database linking function to keep track of
        # save names (ids?) parameters
        props = self.cutout_properties()

        pass

    def get_cutout_properties(self):
        """Initially made for the method `self.tabulate_cutout_properties()`
        Method return the central world coordinates for the primary cutouts
        pixel coordinates for all cutouts (primary and secondary) and the
        percentage of each cutout that is NaN.

        Returns
        -------
        cutout_centers_ra_dec : list of 2-list of floats
            The cutout centers that are used to make the cutouts in wcs
            coordinates.
        xy_cutout_centers_prime : list of 2-list of ints
            The cutout centers that are used to make the cutouts
        perc_nan_in_cutout_prime : numpy.ndarray
            Array containing the percentage (0-100%) of pixels that contain
            NaNs in the cutouts generated from the prime image.
        xy_cutout_centers_secondaries : list of list 2-list of ints
            List containing the cutout centers in pixels, matching the
            same world coordinates as the primary cutouts
        perc_nan_in_cutout_secondaries : numpy.ndarray
            Array containing the percentage (0-100%) of pixels that contain
            NaNs in the secondary cutouts generated from the prime gridding.

        """
        xy_cutout_centers_prime, cutout_centers_ra_dec_prime, perc_nan_in_cutout_prime = self.prime_cuts.get_cutout_properties()

        if self.secondary_cuts is not None:
            all_sec_props = [sec_cut.get_cutout_properties(cutout_centers_ra_dec_prime) for sec_cut in self.secondary_cuts]
            xy_cutout_centers_secondaries, __, perc_nan_in_cutout_secondaries = np.column_stack(all_sec_props)
        else:
            xy_cutout_centers_secondaries = None
            perc_nan_in_cutout_secondaries = None

        return cutout_centers_ra_dec_prime, xy_cutout_centers_prime, perc_nan_in_cutout_prime, xy_cutout_centers_secondaries, perc_nan_in_cutout_secondaries

    def get_ds9_regions(self, ra_dec_cutout_centers=None):

        box_regions = self.prime_cuts.get_cutouts_footprint_ds9_region()

        return box_regions

    def get_cutouts(self):
        cutouts_prime = self.prime_cuts.get_cutouts()

        if self.secondary_cuts is not None:
            cutouts_secondaries = [self.secondary_cuts[i].get_cutouts(cutout_centers_ra_dec=self.cutout_centers_ra_dec_secondaries[i]) for i in range(len(self.secondary_cuts))]
        else:
            cutouts_secondaries = None

        return cutouts_prime, cutouts_secondaries

    def get_cutout_hdus(self):

        cutouts_prime, cutouts_secondaries = self.get_cutouts()

        hdu_primaries = self.prime_cuts.get_cutout_hdus()
        if self.secondary_cuts is not None:
            hdu_secondaries = [self.secondary_cuts[i].get_cutout_hdus(self.cutout_centers_ra_dec_secondaries[i]) for i in range(len(self.secondary_cuts))]
        else:
            hdu_secondaries = None


        return hdu_primaries, hdu_secondaries


    def get_ColoredCutouts(self, prime_color_params, secondary_colors_params=None):
        """By giving the colorization parameters, method will call into
        the other classes to make the false color image of all the bands
        (primary and secondary) given. The colorisation parameters are given
        at a dictionary for the primary image, and a list containg the
        dictionaries for any secondary images. The dictionary can
        be up to 4 parameters which are:
            1. 'rescalefn', the stretch, which can be:
                - 'linear'
                - 'sqrt'
                - 'squared'
                - 'log'
                - 'power'
                - 'sinh'
                - 'asinh'
            If no 'rescalefn' given, or if set to None, the default
            is used, which is a 'linear' stretch
            2. 'min_max', the minimum and maximum intensity values to show.
               If no value is given, or if None is given, the minimum
               and maximum values are used.

            3. 'scaletype' the scaling used for 'min_max'. 'abs' uses the
               absolute units of the image itself. 'perc' uses image
               percentiles. 'perc' works on a per image basis, rather
               than the global percentile value. This means that image
               contrasts will look similar between themselves
               but the `min_max` used for each image will differ.
               If None/no value is entered for 'scaletype', 'abs' is used.
             4. 'color', the color of the image/s. Either enter one of
                the primary additve/subtractive colors as a string,
                enter a hex string, or a 3-list of ints of rgb colors.
                If no color/None is provided, greyscale is used.
        While no error is generated if colors are not given, the outout color
        ends up as greyscale, which will not make a good false color image

        Parameters
        ----------
        prime_color_params : dict
            Colorization parameters for the primary cutouts
        secondary_colors_params : List of dicts, optional
            List containing dictionaries of the colorization parameters for
            any secondary images. The default is None.

        Returns
        -------
        colored_cutouts_rgb_arrays : List of rgb cubes
            All bands are combined into a false color image, which is
            represented in a rgb cube. An rgb does not mean the given colors
            are rgb, but their totally summed up contribution
            (brown, lime, purple for example) results in an rgb image at the
            end when stacked.

        """
        try:
            #do not want to rerun making the objects
            self.colored_cutout_obj
        except AttributeError:
            #------------adding a self so that we do not need to rerun
            #reprojection.
            prime_cuts_hdu, secondary_cuts_hdu = self.get_cutout_hdus()
            self.colored_cutout_obj = ColoredCutouts(prime_cuts_hdu, prime_color_params=prime_color_params, secondary_cuts_hdu=secondary_cuts_hdu, secondary_colors_params=secondary_colors_params, where_secondary_overlap=self.where_secondary_overlap)
            #prime_cuts_hdu, prime_color_params, secondary_cuts_hdu=None

        if secondary_colors_params is not None:
            params = [prime_color_params, *np.atleast_1d(secondary_colors_params)]
        else:
            params = [prime_color_params]

        #If it has been run before, this makes sure we do not rerun.
        if self.colored_cutout_obj.color_params_recently_used == params:

            try:
                colored_cutouts_rgb_arrays = self.colored_cutout_obj.final_images
            except AttributeError:
                colored_cutouts_rgb_arrays = self.colored_cutout_obj.get_colored_arrays(params)
        else:
            colored_cutouts_rgb_arrays = self.colored_cutout_obj.get_colored_arrays(params)

        return colored_cutouts_rgb_arrays


    def save_final_rgbs_as_np_array(self, path, bands, colored_cutouts_rgb_arrays=None, filename_method=zooniverse_cutout_filename_generator, **kwargs):
        """Saves the final combined false color images as numpy objects.
        Numpy objects are kind of like a pickle object. If no cubes are given,
        code assumes it has been run and saves the last.

        Parameters
        ----------
        path : str
            Path where to save the data.
        bands : List of str
            List containing the bands, or what ever information desired to
            describe what each image is, whether that be the band used,
            and the colorization parameters etc, as a string per
            imaged used to make the false color image.
        colored_cutouts_rgb_arrays : List of rgb cubes
            All bands are combined into a false color image, which is
            represented in a rgb cube. An rgb does not mean the given colors
            are rgb, but their totally summed up contribution
            (brown, lime, purple for example) results in an rgb image at the
            end when stacked. The default is None.
        filename_method : Function, optional
            The method to generate a filename for the saved cutouts. The
            default is zooniverse_cutout_filename_generator.
        **kwargs :

        Returns
        -------
        None.

        """
        self._save_rgb(path, bands, np.save, colored_cutouts_rgb_arrays, filename_method, **kwargs)

    def save_colored_cutout_as_imagefile(self, path, bands, colored_cutouts_rgb_arrays=None,
                                         filename_method=zooniverse_cutout_filename_generator,
                                         format='png', **kwargs):
        """Saves the final combined false color images as an image, such as
        a png. If no cubes are given, code assumes it has been run and
        saves the last.

        Parameters
        ----------
        path : str
            Path where to save the data.
        bands : List of str
            List containing the bands, or what ever information desired to
            describe what each image is, whether that be the band used,
            and the colorization parameters etc, as a string per
            imaged used to make the false color image.
        colored_cutouts_rgb_arrays : List of rgb cubes
            All bands are combined into a false color image, which is
            represented in a rgb cube. An rgb does not mean the given colors
            are rgb, but their totally summed up contribution
            (brown, lime, purple for example) results in an rgb image at the
            end when stacked. The default is None.
        filename_method : Function, optional
            The method to generate a filename for the saved cutouts. The
            default is zooniverse_cutout_filename_generator.
        format : str, optional
            The image format. The default is 'png'.
        **kwargs :

        Returns
        -------
        None.

        """
        format = kwargs.pop('format', format)
        origin = kwargs.pop('origin', 'lower')

        self._save_rgb(path=path, bands=bands, save_method=save_image,
                       colored_cutouts_rgb_arrays=colored_cutouts_rgb_arrays,
                       filename_method=filename_method, format=format, origin=origin,
                       **kwargs)

    def _save_rgb(self, path, bands, save_method, colored_cutouts_rgb_arrays=None, filename_method=zooniverse_cutout_filename_generator, **kwargs):

        if colored_cutouts_rgb_arrays is None:
            #params from most recent usage are used. This means it will
            #not rerun unless a new color is asked for
            try:
                colored_cutouts_rgb_arrays = self.colored_cutout_obj.final_images
            except AttributeError:
                colored_cutouts_rgb_arrays = self.colored_cutout_obj.get_colored_arrays()
        band = ''
        for i in range(len(bands)):
            band += '_' + bands[i]
        band = 'rgb_array' + band
        self.prime_cuts._save(save_method, colored_cutouts_rgb_arrays, None, path, band=band,
                              filename_method=filename_method, **kwargs)

    def save_reprojected_cutout_as_fitscube(self, path, bands, filename_method=zooniverse_cutout_filename_generator):
        """Saves the reprojected data as a cube so that after running once
        the observations, as cutouts, can be saved and used again without
        redoing some reduction

        Parameters
        ----------
        path : str
            Path where to save the data.
        bands : List of str
            List containing the bands, or what ever information desired to
            describe what each image is, whether that be the band used,
            etc, as a string per imaged used.
        filename_method : Function, optional
            The method to generate a filename for the saved cutouts. The
            default is zooniverse_cutout_filename_generator.

        Returns
        -------
        None.

        """
        try:
            self.colored_cutout_obj
        except AttributeError:
            #------------adding a self so that we do not need to rerun
            #reprojection.
            prime_cuts_hdu, secondary_cuts_hdu = self.get_cutout_hdus()
            self.colored_cutout_obj = ColoredCutouts(prime_cuts_hdu, prime_color_params=prime_color_params, secondary_cuts_hdu=secondary_cuts_hdu, secondary_colors_params=secondary_colors_params, where_secondary_overlap=self.where_secondary_overlap)

        data3d_cube = self.colored_cutout_obj.get_stacked_images()

        cube_hdus = self.colored_cutout_obj.hdus[:,0]
        for i in range(len(self.colored_cutout_obj.hdus[:,0])):
            header_3d = make_header_3d(cube_hdus[i].header, np.shape(data3d_cube)[0])

            cube_hdus[i] = update_hdu(cube_hdus[i], data3d_cube[i], header_3d)

        for i in range(len(bands)):
            band += '_' + bands[i]
        band = 'cube' + band
        self.prime_cuts._save(save_fitsfile, cube_hdus, path, band=band, filename_method=filename_method)


    def save_fits_cutouts(self, path, bands):
        """Induvidually saves the cutouts for the primary image and any
        secondary images

        Parameters
        ----------
        path : str
            Path where to save the data.
        bands : List of str
            List containing the bands, or what ever information desired to
            describe what each image is, whether that be the band used,
            etc, as a string per imaged used.

        Returns
        -------
        None.

        """


        bands = np.atleast_1d(bands)

        self.prime_cuts.save_fits_cutouts(path, bands[0])

        for i in range(len(bands)-1):
            self.secondary_cuts.save_fits_cutouts(path, bands[i+1], self.cutout_centers_ra_dec_secondaries[i])

    def save_as_ds9regions(self, path):
        self.prime_cuts.save_as_ds9regions(path)
