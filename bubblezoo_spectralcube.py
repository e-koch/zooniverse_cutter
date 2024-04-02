
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import scipy.ndimage as nd

from astropy.io import fits
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales

from spectral_cube import Projection

# Slightly modified version of this package
import multicolorfits as mcf


def add_proj_to_dict(data_path, galaxy_name, color_dict):
    '''
    Add a projection to the existing dictionary with the different bands.
    '''

    for this_band in color_dict.keys():

        # Search for the file in the requested band.
        this_galaxy_path = data_path / f"{galaxy_name}"

        this_band_hduext = color_dict[this_band]['hdu_ext']

        if 'edge_clip' in color_dict[this_band].keys():
            edge_clip = color_dict[this_band]['edge_clip']
        else:
            edge_clip = False

        if edge_clip:
            edge_clip_size = color_dict[this_band]['edge_clip_size']


        # Does it exist?
        all_matches = list(this_galaxy_path.glob(f"*{color_dict[this_band]['file_search_str']}*"))
        if len(all_matches) == 0:
            raise FileNotFoundError(f"Could not find {this_galaxy_path}")
        elif len(all_matches) > 1:
            raise ValueError(f"Found multiple files for {this_galaxy_path}")
        else:
            color_dict[this_band]['data'] = Projection.from_hdu(fits.open(all_matches[0])[this_band_hduext])

            # Just keep the numpy array for the mask
            this_data = fits.open(all_matches[0])[this_band_hduext].data
            this_mask = np.logical_and(np.isfinite(this_data), this_data != 0)

            # But assert the shape is at least the same
            if np.shape(color_dict[this_band]['data']) != np.shape(this_data):
                raise ValueError(f"Specified mask extension does not have a matching shape to the data: {all_matches[0]}")

            # Optioanlly clip the edges of the image. Helps visually clean-up edges of the HST bands.
            if edge_clip:
                this_mask = nd.binary_erosion(this_mask, iterations=edge_clip_size)

            color_dict[this_band]['mask'] = this_mask

            # Apply the mask to the data
            color_dict[this_band]['data'][~this_mask] = np.NaN

def make_cutouts(color_dict,
                 output_path,
                 gal_name=None,
                 target_band='f770w',
                 cutout_sizes=[1]*u.arcmin,
                 grid_overlap=0.5,
                 target_min_finite_frac=0.5,
                 img_format='png',
                 min_dpi=300,
                 save_kwargs={'origin': 'lower'},#, 'interpolation': 'nearest'},
                 save_bw_cutouts=True,
                ):

    # Save the headers in a subdirectory of outpout_path
    output_hdr_path = output_path / "headers"
    output_hdr_path.mkdir(parents=True, exist_ok=True)

    if 'format' in save_kwargs:
        img_format = save_kwargs['format']
        save_kwargs.pop('format')

    # Want the image size to be ~approx uniform.
    # Scale DPI up based on the largest cutout size.

    if 'dpi' in save_kwargs:
        min_dpi = save_kwargs['dpi']
        save_kwargs.pop('dpi')

    max_size = max(cutout_sizes)
    # cutout_dpis = [min_dpi * (max_size / size) for size in cutout_sizes]
    zoom_ratios = [np.round(max_size / size, 2).value for size in cutout_sizes]
    # print(cutout_dpis)

    # Reproject to a common grid.
    reproj_dict = {}

    # Reproject to the target header
    target_header = color_dict[target_band]['data'].header
    for this_key in color_dict.keys():
        if this_key == target_band:
            reproj_dict[this_key] = color_dict[target_band]['data']
            continue

        reproj_dict[this_key] = color_dict[this_key]['data'].reproject(target_header)


    # Make the cutouts, save to png files with 4 different rotations, and a txt file of the header for each cutout
    for cutout_size, zoom_ratio in zip(cutout_sizes, zoom_ratios):

        # Make the pixel grid.

        pixel_grid = reproj_dict[target_band].shape

        pixel_unit = u.Unit(target_header['CUNIT1'])
        scale_pixel = proj_plane_pixel_scales(reproj_dict[target_band].wcs)[0] * pixel_unit

        # cutout_pixel_size =  * grid_overlap
        cutout_pixel_size = np.ceil((cutout_size / scale_pixel).to(u.one).value).astype(int)
        cutout_pixel_step = np.ceil(cutout_pixel_size * grid_overlap).astype(int)

        y_inds = np.arange(cutout_pixel_step, pixel_grid[0],
                        cutout_pixel_step, dtype=int)

        x_inds = np.arange(cutout_pixel_step, pixel_grid[1],
                        cutout_pixel_step, dtype=int)

        # For including the in the filenames.
        size_str = f"{cutout_size.value:.2f}".replace(".", "p") + str(cutout_size.unit)

        iter = 0

        for y_ind in y_inds:
            for x_ind in x_inds:

                slicer = (slice(y_ind - cutout_pixel_size // 2,
                                y_ind + cutout_pixel_size // 2),
                          slice(x_ind - cutout_pixel_size // 2,
                                x_ind + cutout_pixel_size // 2))

                # Check that the cutout has finite values above the specified min.
                finite_frac = np.isfinite(reproj_dict[target_band][slicer]).sum() / float(reproj_dict[target_band][slicer].size)
                if finite_frac < target_min_finite_frac:
                    continue

                grey_images = {this_band: mcf.greyRGBize_image(nd.zoom(reproj_dict[this_band][slicer].value, zoom_ratio, order=0),
                                                               **color_dict[this_band]['greyRGBize_kwargs'])
                               for this_band in color_dict.keys()}

                colored_images = {this_band: mcf.colorize_image(np.nan_to_num(grey_images[this_band]),
                                                                color_dict[this_band]['color'],
                                                                **color_dict[this_band]['colorize_image_kwargs'])
                                  for this_band in color_dict.keys()}

                combined_RGB = mcf.combine_multicolor(list(colored_images.values()),
                                                      gamma=2.2)

                # Make several versions with different rotations:
                # rotation angle = this_angel * 90 deg for np.rotate90
                for this_angle in range(0, 4):

                    filename = output_path / f"{gal_name}_{size_str}_zoom{zoom_ratio:.2f}_{iter:04}_rot{(this_angle * 90):.0f}.{img_format}"

                    plt.imsave(filename, np.rot90(combined_RGB, k=this_angle),
                               **save_kwargs)
                            #    dpi=cutout_dpi,

                # Optionalely save the cutouts of the grey scales images per band:
                if save_bw_cutouts:
                    for this_band in grey_images:

                        # Also apply the gamma scaling factor used for the combined images.
                        this_bw_image= grey_images[this_band]**(1/2.2)

                        for this_angle in range(0, 4):

                            filename = output_path / f"{gal_name}_{size_str}_zoom{zoom_ratio:.2f}_{iter:04}_rot{(this_angle * 90):.0f}_{this_band}.{img_format}"

                            plt.imsave(filename, np.rot90(this_bw_image, k=this_angle),
                                       **save_kwargs)
                                    # dpi=cutout_dpi,


                # Save txt versions of the headers for each cutout. Along with the png file saved per-pixel,
                # we can reproduce the pixel to sky conversions.
                hdr_cutout = reproj_dict[target_band][slicer].header
                hdr_filename = output_hdr_path / f"{gal_name}_{size_str}_{iter:04}.header.txt"

                hdr_cutout.totextfile(str(hdr_filename), overwrite=True)


                iter += 1


def load_toml(path, job_id=None):
    """
    Load a toml file from disk
    """
    import tomllib
    with open(path, 'rb') as f:
        data = tomllib.load(f)

    data_path = Path(data['data_path'])
    output_path = Path(data['output_path'])

    # Check there is a parameter input for each specified band.
    color_dicts = {}
    for this_band in data['bands']:
        assert this_band in data['parameters']
        color_dicts[this_band]= data['parameters'][this_band]

    if job_id is not None:
        try:
            galaxy_name = [data['targets'][job_id]]
        except Exception as exc:
            print(f"Hit exception for job_id {job_id}")
            print(exc)
            import sys
            sys.exit(1)
    else:
        galaxy_name = data['targets']

    return galaxy_name, data_path, output_path, color_dicts



if __name__ == "__main__":

    import sys
    from pathlib import Path

    # The job number in the job array is used to select the galaxy.
    job_id = int(sys.argv[-1]) if sys.argv[-1] != "None" else None

    # config_file = "colors_config.toml"
    config_file = Path(sys.argv[-2])

    galaxy_names, data_path, output_path, color_dict = load_toml(config_file, job_id=job_id)

    data_path = Path(data_path)

    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    print(f"Running on galaxies: {galaxy_names}")
    print(f"Output to: {output_path}")

    for galaxy_name in galaxy_names:

        print(f"Working on {galaxy_name}")

        add_proj_to_dict(data_path, galaxy_name, color_dict)

        make_cutouts(color_dict, output_path,
                     gal_name=galaxy_name,
                     target_min_finite_frac=0.5,
                     cutout_sizes=[0.5, 1]*u.arcmin,
                     save_bw_cutouts=True)

