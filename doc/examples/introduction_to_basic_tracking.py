"""
====================
Local Fiber Tracking
====================

Local fiber tracking is an approach used to model white matter pathways by
creating streamlines from local directional information. The idea is that if
one knows the directionality of segment of a tracts or pathways, she can
integrate along those directions to build the complete representation of that
structure. Local fiber tracking are widely within the field of diffusion MRI
because they are simple and robust, but they only use local information to
determine tracking directions (hence the name).

In order to do local fiber tracking one needs 3 things. 1) A way of getting
directions from a diffusion data set. 2) A method for identifying different
tissue types within the data set. 3) A set of seeds from which to begin
tracking.  This example shows how to combine the 3 parts described above to
create a tractography data sets.

"""

"""
To begin first lets load an example hardi data set from Stanford. If you have
not already downloaded this data set, the first time you run this example you
will need to be connected to the internet and this dataset will be downloaded
it to your computer.
"""

from dipy.data import read_stanford_labels

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.get_affine()

"""
This data set provides a label map where all white matter tissues are labeled
either 1 or 2, so lets create a white matter mask to restrict tracking to the
white matter.
"""

white_matter = (labels == 1) | (labels == 2)

"""
The first thing we need to begin fiber tracking is a way of getting directions
from this diffusion data set. In order to do that, we can fit the data to a
Constant Solid Angle Odf Model. This model will estimate the ODF, or
orientation distribution function, at each voxel. The ODF is the distribution
of water diffusion as a function of direction. The peaks of an ODF are good
estimates for the orientation of tract segments at a point in the image.
"""

from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.peaks import peaks_from_model, default_sphere

csa_model = CsaOdfModel(gtab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=white_matter)

"""
The next thing we will need is some way of restricting the fiber tracking to
areas with good directionality information. We've already created the white
matter mask, but we can go a step further and restrict fiber tracking to those
areas there the odf shows significant restricted diffusion by thresholding on
the GFA (General Fractional Anisotropy).
"""

from dipy.tracking.local import ThresholdTissueClassifier

classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

"""
We need one more thing before we can begin tracking, we need to specify where
to begin, or seed, the fiber tracking. Generally the seeds one chooses will
depend on the pathways she is interested in modeling. In this example we'll use
a 2x2x2 grid grid of seeds per voxel, in a sagittal slice of the Corpus
callosum. Tracking from this region will give us a model of the Corpus Callosum
tract. This slice has label value 2 in the labels image.
"""

from dipy.tracking import utils

seed_mask = labels == 2
seeds = utils.seeds_from_mask(seed_mask, density=[2, 2, 2], affine=affine)

"""
Finally we can bring it all together using ``LocalTracking``. We then display
the resulting streamlines using the fvtk module.
"""

from dipy.tracking.local import LocalTracking
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

# LocalTracking is lazy so the streamlines will not be actually computed until
streamlines = LocalTracking(csa_peaks, classifier, seeds, affine, step_size=.5)

# Compute streamlines and store as a list.
streamlines = list(streamlines)

# Prepare the display objects.
color = line_colors(streamlines)
streamlines_actor = fvtk.line(streamlines, line_colors(streamlines))

# Create the 3d display.
r = fvtk.ren()
fvtk.add(r, streamlines_actor)

# Save still images for this static example.
fvtk.record(r, n_frames=1, out_path='deterministic.png',
            size=(800, 800))

"""
.. figure:: deterministic.png
   :align: center

   **Corpus Callosum Deterministic**
"""

"""
We've created a deterministic set of streamlines, so called because if you
repeat the fiber tracking, keeping all the inputs the same, you will get
exactly the same set of streamlines. We can save the track as a "trk" file
so it can be loaded into other software for visualization or further analysis.
"""

from dipy.io.trackvis import save_as_trk
save_as_trk("CSA_detr.trk", streamlines, affine, labels.shape)

"""
Next lets try some probabilistic fiber tracking. For this we'll be using the
Constrained Spherical Deconvolution (CSD) model. This model represents each
voxel in the data as collection of small white matter fibers with different
orientations. The density of fibers along each orientation is known as the
Fiber Orientation Distribution, or FOD. While one could still use this model to
do deterministic fiber tracking by always tracking along the directions that
have the most fibers, but in order to do probabilistic fiber tracking we can
pick a fiber from the distribution at random at each new location along the
streamline.

Lets begin by fitting the data to the CSD model.
"""

from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)

"""
Next we'll need to make a ``ProbabilisticDirectionGetter``. Because the CSD
model represents the FOD using the spherical harmonic basis, we can use the
``from_shcoeff`` method to create the direction getter. This direction getter
will randomly sample directions from the FOD each time the tracking algorithm
needs to take another step.
"""

from dipy.tracking.local import ProbabilisticDirectionGetter

prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                    max_angle=30.,
                                                    sphere=default_sphere)

"""
We can pass this direction getter to ``LocalTracking`` just like we did before,
along with the ``classifier`` and ``seeds`` to get a probabilistic model of the
corpus callosum.
"""

classifier = ThresholdTissueClassifier(csd_fit.gfa, .25)
streamlines = LocalTracking(prob_dg, classifier, seeds, affine,
                            step_size=.5, max_cross=1)

# Compute streamlines and store as a list.
streamlines = list(streamlines)

# Prepare the display objects.
color = line_colors(streamlines)
streamlines_actor = fvtk.line(streamlines, line_colors(streamlines))

# Create the 3d display.
r = fvtk.ren()
fvtk.add(r, streamlines_actor)

# Save still images for this static example.
fvtk.record(r, n_frames=1, out_path='probabilistic.png',
            size=(800, 800))

"""
.. figure:: probabilistic.png
   :align: center

   **Corpus Callosum Probabilistic**
"""

save_as_trk("CSD_prob.trk", streamlines, affine, labels.shape)

