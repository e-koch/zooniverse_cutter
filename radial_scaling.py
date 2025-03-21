
from scipy.stats import binned_statistic
import numpy as np
import astropy.units as u

try:
    from phangs import PhangsGalaxy


    def radial_scaling(galaxy, image, threshold=100.,
                       verbose=True):
        """
        This function returns the radial scaling for a given radius.
        """

        gal = PhangsGalaxy(galaxy)

        this_rgal = gal.radius(header=image.header).to(u.kpc)

        bins = np.arange(np.floor(np.min(this_rgal.value)),
                         np.ceil(np.max(this_rgal.value)),
                         1.0)

        binned = binned_statistic(this_rgal.value.ravel(),
                                  image.value.ravel(),
                                  statistic=np.nanmean,
                                  bins=bins)

        if not any(binned.statistic > threshold):
            if verbose:
                print('No radial scaling needed.')
            return image

        if verbose:
            print('Applying radial scaling.')

        # Otherwise scale the inner portion of the image.
        # Assumes the center is the brightest (should be a reasonable approx)
        rad_scaling = (np.arctan(this_rgal.to(u.kpc).value / 2)) * 2 / np.pi

        return image * rad_scaling

except ImportError:
    print('phangs package not found.  Cannot use radial_scaling.py')

    def radial_scaling(galaxy, image, threshold=100.):
        return image
