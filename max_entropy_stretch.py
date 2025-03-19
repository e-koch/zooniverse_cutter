from lmfit import Minimizer, Parameters
import numpy as np
from astropy.visualization import ImageNormalize, AsinhStretch, ManualInterval

def image_entropy(pars, image):
    vmin = pars['vmin'].value
    vmax = pars['vmax'].value
    norm = ImageNormalize(image, interval=ManualInterval(vmin, vmax),
                          stretch=AsinhStretch())
    normdata = norm(image).clip(1e-4, 1)
    negentropy = np.nansum(normdata * np.log(normdata))
    # minimize negative entropy = maximize entropy
    return(negentropy)

def maxent_bounds(image, return_norm=False,
                  raise_error_on_fail=False):

    min_val = np.nanmin(image)
    max_val = np.nanmax(image)

    fit_fail = np.all(np.isnan(image)) or min_val == max_val

    if fit_fail:
        if raise_error_on_fail:
            raise ValueError('Image has no valid data or range.')
        else:
            return np.nanmin(image), np.nanmax(image)

    params = Parameters()
    params.add('vmin', value=np.nanpercentile(image, 1), vary=False, min=min_val)
    params.add('range', value=(max_val-min_val)/2,
               vary=True, min=0, max=(max_val-min_val))
    params.add('vmax', expr='vmin + range')

    mm = Minimizer(image_entropy, params, fcn_args=(image,), jac='3-point')
    result = mm.minimize(method='lbsfgb')

    if return_norm:
        norm = ImageNormalize(image,
                        interval=ManualInterval(params['vmin'].value, params['vmax'].value),
                        stretch=AsinhStretch())
        return norm
    else:
        return result.params['vmin'].value, result.params['vmax'].value
