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

def maxent_bounds(image, return_norm=False):
    params = Parameters()
    params.add('vmin', value=np.nanpercentile(image, 1), vary=False, min=np.nanmin(image))
    params.add('range', value=(np.nanmax(image)-np.nanmin(image))/2,
               vary=True, min=0, max=(np.nanmax(image)-np.nanmin(image)))
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
