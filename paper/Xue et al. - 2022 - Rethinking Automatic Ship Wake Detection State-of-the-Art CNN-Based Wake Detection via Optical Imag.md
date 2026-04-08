IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 60, 2022 5613622
Rethinking Automatic Ship Wake Detection:
State-of-the-Art CNN-Based Wake
Detection via Optical Images
Fuduo Xue , Graduate Student Member, IEEE, Weiqi Jin , Su Qiu ,
and Jie Yang , Graduate Student Member, IEEE
Abstract— Most existing wake detection algorithms use Radon
```
transform (RT) due to the long streak features of ship wakes
```
in synthetic aperture radar images. The high false alarm rate
of RT requires that the algorithms have significant human
intervention for image preprocessing. When processing optical
images, these algorithms are greatly challenged because it is not
only sea clutter that interferes with the wake detection in the
optical waveband but also many other environmental factors.
Therefore, in this article, we address the automatic ship wake
detection task in optical images from the idea of the convolutional
```
neural network (CNN)-based object detection and design an end-
```
to-end detector named WakeNet as a novel method. WakeNet
processes all the wake textures clamped by the V-shaped Kelvin
arms as the object and conducts detection via the oriented
```
bounding box; the additional regression of wake tip coordinates
```
and Kelvin arm direction can improve the hard wake detection
performance while predicting the wake heading. According to the
X-shaped wake spectral pattern and the wake edge enhancement
principle in adjacent scales, spectral attention and multiscale
attention modules are also used. Furthermore, more than 11.6k
wake optical images are collected and annotated, providing a
benchmark dataset. The experimental results on the dataset
show that the proposed method has a better performance than
other CNN-based methods, and the detection results of the yacht
wake polarization images further verify its practical value. This
article demonstrates that the deep learning-based method has
considerable advantages over the traditional RT-based methods
in wake detection.
Index Terms— Benchmark dataset, convolutional neural
```
network (CNN), Radon transform (RT), ship wake.
```
I. I NTRODUCTION
S
HIP wake detection is an indirect method to obtain the
information about moving vessels and has important civil
and military significance. The wakes are the most obvious
features of moving ships in remote sensing images and can
reach tens of kilometers and contain significant ship informa-
tion, including heading, speed, and hull geometry. Detecting
a combination of ships and their wakes is more practical than
directly detecting the ship hull itself [1].
```
Manuscript received September 5, 2021; revised November 5, 2021;
```
```
accepted November 15, 2021. Date of publication November 17, 2021; date
```
of current version February 17, 2022. This work was supported in part by the
```
National Science Foundation of China under Grant 62171024. (Corresponding
```
```
author: Weiqi Jin.)
```
The authors are with the School of Optics and Photonics, Beijing Insti-
```
tute of Technology, Beijing 100081, China (e-mail: 18301224199@163.com;
```
```
jinwq@bit.edu.cn; edmondqiu@bit.edu.cn; yangjie_bit@163.com).
```
Digital Object Identifier 10.1109/TGRS.2021.3128989
```
Both synthetic aperture radar (SAR) and optical remote
```
sensing can perform sea wake observation tasks. Currently,
SAR occupies the dominant position due to its early tech-
```
nological start; since SEASAT SAR first reported on wake
```
imaging in 1978 [2], a large number of theoretical studies
```
(e.g., [3]–[8]) have given scholars a clear understanding of the
```
wake SAR imaging mechanism, which can be summarized as:
the interaction between a wake and gravity-capillary waves
that change sea surface roughness and thus modulate the
radar echoes to generate signals in SAR. Wake optical remote
sensing has only been developed in recent years. With greater
interest and research on the optical reflectance characteristics
of rough sea surface, more and more researchers argue that it
```
is completely feasible to observe wakes in the visible band;
```
by analogy to SAR imaging mechanism, some wake imaging
models for different optical sensors have been established
successively [8]–[11]. Since optical remote sensing is more
developed than SAR, wake optical remote sensing has rapidly
become a research hotspot.
This article focuses on the detection process after the wake
imaging. Ship wakes available for detection in remote sensing
images mainly include four categories [12]: Kelvin wake,
turbulent wake, internal wake, and narrow V-wake. Among
them, internal wake and narrow V-wake can only be observed
```
under specific environmental conditions (shallow seawater
```
```
stratification [13]) and imaging methods (based on the Bragg
```
```
scattering principle of SAR [7]). In general studies, the more
```
common Kelvin wake and turbulent wake thus become the
detection objects. Kelvin wake and turbulent wake have typical
```
patterns in both SAR and optical images; in SAR images,
```
they usually exhibit long streaks [14], i.e., the outer bright
lines formed by the two Kelvin arms and the middle dark
line formed by the turbulent wake. Due to the coherent
imaging and generally low spatial resolution of SAR, the
texture details of transverse and divergent waves inside the
Kelvin wake are less observed. In contrast, due to the higher
resolution and incoherent imaging of optical remote sensing,
as long as the wake energy is sufficiently stronger than the
sea clutter energy, wake transverse and divergent waves can
show texture details in most optical images [1]. In addition,
the Kelvin arms exhibit wider, alternately bright, and dark
```
semantic line features (Fig. 1). From this perspective, wakes
```
in optical images have more features that can be extracted and
interpreted than wakes in SAR images.
1558-0644 © 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
5613622 IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 60, 2022
Fig. 1. Ship wake system in an optical remote sensing image. Generally,
a complete wake system is composed of Kelvin wake and turbulent wake.
A far-field Kelvin wake consists of transverse waves propagating parallel to
```
the ship heading and divergent waves diffusing to both sides of the ship hull;
```
the superposition of transverse and divergent waves at the wake edge forms
```
two Kelvin arms (cusp waves), and the angle between the two Kelvin arms is
```
39◦ in classical theory. A turbulent wake consists of a turbulent region with
a lot of bubble foams in the front and a smooth region with fewer surface
waves in the back.
Most of the existing wake detection algorithms are designed
for SAR images due to the short development of wake optical
remote sensing. The prominent linear features of wakes in
SAR images lead these algorithms to invariably use the
```
Radon transform (RT)-based [15], [16] or similar scanning
```
methods [17]. The performance of the algorithm is mainly
```
promoted by enhancing the signal-to-noise ratio (SNR) of the
```
wake to the sea clutter [18] or by separating the wake from
the sea clutter [19], [20]. Therefore, a natural idea to solve the
task of wake detection in optical images is to transfer these
RT-based algorithms [21], just like what has been done when
discussing the wake optical imaging mechanism. However, the
differences between optical and SAR images and the inherent
defects of traditional algorithms make it difficult to meet the
practical requirements through simple modifications on these
algorithms, which is reflected in the following aspects.
```
1) Traditional algorithms focus on the preprocessing of
```
removing sea clutter or enhancing wake signals because
the detectability of wakes in rough sea SAR images
is mainly affected by the sea state [7]. However, the
```
sea scene in the optical image is complex and diverse;
```
besides, the sea clutter, sunglint, coasts and ports, sta-
```
tionary ships (that do not produce wakes), cloud cover,
```
and even the overlapping of multiple wake textures can
hinder the wake detection.
```
2) Since the textural details of the wake in the SAR image
```
are not rich, we can simply use RT to extract the Kelvin
arms and the turbulent wake features. By contrast, the
rich textural features of wakes in optical images yield
more variable patterns. The Froude number determines
the influence of gravity on the motion of fluids, which
```
is expressed as: Fr = U/(g L)1/2 (where U is the ship
```
velocity, L is the ship length, and g is the local gravi-
```
tational acceleration [22]). The Kelvin wake elevation
```
```
distribution depends on the Froude number; although
```
the wake has the consistent features on the whole,
its internal patterns vary with the hull shape and ship
motion states. RT-based algorithms are obviously unable
to effectively extract such multifeature patterns.
```
3) A common problem of the RT-based algorithms is: since
```
the linear feature is low-level information with poor
generalization, sea clutter introduces a large number
of false alarms, and these false alarms require that
the algorithms have significant human intervention in
the preprocessing [23], [24], thereby greatly reducing the
automation level. In addition, the iterative preprocessing
is time-consuming.
In recent years, the optical remote sensing object detec-
```
tion based on convolutional neural network (CNN) research
```
```
has made remarkable achievements (e.g., [25]–[27]). These
```
algorithms have a high degree of automation due to the
```
end-to-end structure; some special structures and strategies
```
significantly improve the detection performance of small and
arbitrary-oriented objects in complex backgrounds [28]–[31].
In addition, the in-depth research on weakly supervised learn-
ing and representation learning has gradually alleviated the
dependence on the cumbersome bounding box annotations and
improved the classification performance on new categories of
targets that beyond the predetermined scenes [32], [33]. Ship
wakes are typical maritime arbitrary-oriented objects, which
are much larger than the ship objects usually detected by the
CNN-based algorithms. The above three problems encountered
by the RT-based algorithms can theoretically be effectively
solved by the CNN-based method. Besides, there are a variety
of optical images that are easy to access, and it is not difficult
to construct a wake imagery dataset. Thus, CNN-based wake
detection is promising.
This article proposes a novel CNN-based solution for the
automatic ship wake detection in optical images. A single-
```
stage anchor-based detector is designed; it analyzes the two
```
Kelvin arms and all the wake textures clamped by them as
objects, learns transverse wave and divergent wave features,
and implements detection using the oriented bounding box
```
(OBB). FcaNet [34] is used as the backbone to indirectly
```
extract the spectral features of the wake through channel
attention. To better utilize the spatial correlation between
feature maps at different levels in the feature pyramid net-
```
work (FPN) [35], a multiscale attention module (MSAM)
```
is proposed. On these bases, we also added an additional
subnetwork, containing an RT layer to predict the wake tip
location and the Kelvin arm direction to improve the hard
wake detection performance and realize wake orientation and
ship locating. Moreover, we collected a large-scale dataset
containing 11 600 carefully annotated wake optical images to
effectively train and evaluate our method. The contributions
and innovations of this article are summarized as follows.
```
1) We propose a single-stage object detection network for
```
automatic wake detection in optical images that are
rarely studied at present. To the best of our knowledge,
this is the first time that CNN has been applied to optical
wake detection.
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
XUE et al.: RETHINKING AUTOMATIC SHIP WAKE DETECTION: STATE-OF-THE-ART CNN-BASED WAKE DETECTION 5613622
```
2) The proposed detector adopts a multitask learning strat-
```
egy, which can simultaneously predict the wake prob-
ability score, the wake bounding box, the Kelvin arm
```
direction, and the wake tip location; its performance
```
is also better than other CNN-based methods. The
additional regression subnetwork combines the CNN’s
feature learning capacity with RT, further verifying
the compatibility between CNN and traditional image
transformation.
```
3) We explain in principle how the added attention mecha-
```
nisms enhance the CNN to more effectively extract and
aggregate wake features.
```
4) We constructed a wake imagery dataset, containing
```
11 600 images and 15 356 annotated instances, as the
benchmark dataset. This dataset is not only the first
known open-access wake detection dataset but also a
rare large-scale maritime object detection dataset now
available to the remote sensing community.
The rest of this article is organized as follows. Section II
summarizes the related works. Section III introduces the
constructed wake imagery dataset. Section IV details the
CNN-based wake detector proposed according to the wake
features. Section V validates the proposed method by analyz-
ing the experimental results. Finally, Section VI provides the
conclusions to our study.
```
Notations: The notations in this article follow the following
```
rules. Italic characters belong to constants or scalars. Lower-
case and uppercase regular font characters represent functions
and transforms, respectively. Bold lowercase characters indi-
cate vectors and bold uppercase characters represent matrices
```
or tensors (i.e., feature maps). In addition, the letter “b” in the
```
notation “ab c” denotes superscript rather than power.
II. R ELATED W ORK
The related research works can be divided into three aspects:
ship wake detection algorithm in SAR images, CNN-based
optical remote sensing object detection algorithm, and remote
sensing object detection dataset. The following is a summary
of the featured works in each aspect.
A. Ship Wake Detection Algorithms
Most of the existing wake detection algorithms are proposed
for SAR images. The wakes exhibit simple streak-like patterns
in SAR images, so the algorithms tend to extract low-level
handcrafted features, which are mainly linear features. The
main implementations include RT and scanning, both of which
are essentially the same, and both require preprocessing to
eliminate the interference caused by the sea clutter.
The RT-based algorithms detect the wake linear features by
calculating the projection of the image pixels along various
directions. Early studies often used filtering or enhancement
methods to improve the wake SNR. Rey et al. [15] performed
high-pass filtering and Wiener filtering in the image domain
and the Radon domain, respectively, to suppress sea clutter.
Copeland et al. [16] performed RT in the overlapping local
windows instead of the entire image so that the algorithm
could better respond to wakes composed of short line seg-
ments. Based on the spatial correlation of the wake signals
at adjacent scales, the edge features of wake are enhanced by
calculating the multiscale image modulus obtained by wavelet
transform in [18]. Arnold-Bos et al. [36] used RT to detect
the X-shaped feature in the wake frequency spectrum rather
than the line segments in the image domain, avoiding false
alarms that may be formed by sea wave textures. Recent
studies mainly use the idea of compressed sensing to separate
the wake signals from the sea clutters. Biondi [19], [20] used
```
the robust principal component analysis (RPCA) to perform
```
two-level low-rank sparse decomposition in the image and
Radon domains and then detected the sparse component where
the wake signal is retained. On this basis, Zhao et al. [37]
```
adopted the random sample consensus (RANSAC) algorithm
```
to impose constraints on the finer linear features of wakes.
Yang et al. [38] used the morphological component analysis
```
(MCA) to construct a wake texture dictionary and a sea-
```
background texture dictionary and then separated the signal
and the background. Moreover, similar to the scanning-based
method, Graziano et al. [23], [24] made decisions on the RT
output according to a priori after detecting the wake tip to
reduce the false alarm rate. Recently, to fill the research gaps in
optical wake detection, Liu et al. [21] directly transferred the
```
above RT-based algorithm framework; the authors proposed
```
a series of criteria to suppress false alarms according to the
wake hydrodynamic characteristics and the features of ocean
optical remote sensing images, designed an RT-based optical
image wake detection algorithm, and discussed the impact of
hyperparameters, wake verification, and spatial resolution on
the detection performance.
The scanning-based method, first proposed in [17], ini-
tially obtains a scanning curve by calculating the mean
image brightness in all directions around the ship object and
then determines the possible wake direction from this curve.
Nan et al. [39] and Björn and Domenico [40] improved the
algorithm performance by adopting preprocessing to enhance
the wake signal. Wei et al. [41] used a periodic function as
the scanning curve to detect the alternately bright and dark
streaks of the Kelvin arm.
In addition, there are a few deep learning-based SAR
image wake detection algorithms, such as [42] and [43]. They
adopted the image block strategy and utilized the image block
```
with label and data as input; the networks did not predict
```
the object coordinates and were essentially binary classifiers
that judged whether the image block contained a wake object.
These simple methods have a definite gap in performance with
the mainstream CNN remote sensing object detectors.
B. Optical Remote Sensing Object Detector Based
on Deep Learning
Inspired by the great success of CNN-based natural scene
object detection, CNN-based optical remote sensing object
detection technology has developed rapidly. The current opti-
cal remote sensing object detectors are primarily modified
by the natural scene object detectors with outstanding per-
```
formance (e.g., two-stage R-CNN [44], Fast R-CNN [45],
```
```
and Faster R-CNN [46]; single-stage anchor-based SSD [47],
```
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
5613622 IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 60, 2022
```
YOLO [48], and RetinaNet [49]; as well as single-stage
```
```
anchor-free FCOS [50] and CenterNet [51]), and the pipelines
```
are similar to their original algorithms. There are three char-
acteristics of the remote sensing images that are different
from the natural scene images, namely, high background
complexity, small object, and arbitrary-oriented object.
The influence of a complex background on object detection
can generally be solved by using an attention mecha-
```
nism: via reweighting the feature map, highlighting useful
```
object features, and suppressing the redundant semantic back-
ground information. Typical works include the following.
Pang et al. [52] used a global attention module to enable
the network to make better use of the contextual semantics
of small objects and improved the accuracy of classification.
Zhang et al. [53] proposed a spatial and scale attention module
to enable different types of objects with different scale features
to have different intensity responses at each layer of the FPN.
Yang et al. [54] used a supervised semantic segmentation
method to reweight the features of objects in different classes
into different channels and, respectively, enhanced and weak-
ened the object and background features in the spatial domain.
Small object detection is a challenging task in optical remote
sensing. The current common solutions include multiscale
strategy, hard mining method, and so on. By integrating
these methods, some elegant small object detectors have been
proposed, such as [52] and [53].
In order to detect arbitrary-oriented objects in remote sens-
ing images, the existing research extends the classic horizon-
tal detector to the rotating case. The distance loss function
```
directly extended from the horizontal bounding box (HBB)
```
loss function is sometimes contradictory to the OBB inter-
```
section of union (IoU), and this affects the regression of the
```
high aspect ratio bounding boxes. Therefore, some special
regression strategies and loss functions are proposed. In some
two-stage detectors such as [27], the regression of the HBB
before rotation and the regression of the bounding box rotation
angle are performed separately, and the detector only needs to
calculate the horizontal IoU. A few studies such as [55] treat
rotation detection as an angle classification task at the expense
of high precision of the predicted rotation angle. The current
mainstream method is to obtain a numerically differentiable
rotating IoU through engineering or approximate methods
and use the IoU as a part of the loss function to directly
guide the regression. Typical methods include the following.
Chen et al. [56] approximated the IoU by calculating the ratio
of the overlapping pixels of the two bounding boxes to the total
pixels they occupy. By converting the OBB into a Gaussian
distribution, Yang et al. [57] calculated the Gauss–Wasserstein
distance to approximate IoU. Yang et al. [54] used the IoU
as a coefficient to combine with smooth L1 loss, avoiding the
derivation of rotating IoU.
In addition, because of the multifarious and disorderly
categories of geospatial objects, the traditional boundary box
annotation method is time-consuming and difficult to consider
```
all types of images; the acquisition of the target sample
```
images is limited by remote sensors and the dataset size is
usually small. These problems prompt some researchers try to
apply weakly supervised learning and representation learning
into multiclass remote sensing target detection. To alleviate
the dependence on bounding box annotations, Li et al. [32]
used the mutual information between scenes pairs and the
pointwise scene tags to train the discriminative convolutional
weights and the class-specific activation weights and proposed
a target detector using only scene-level tags via the sliding-
voting strategy. Li et al. [33] constructed a remote sensing
knowledge graph to generate the semantic representation of
remote sensing scene categories and proposed a deep align-
ment network to achieve cross-modal matching between visual
features and semantic information and to accurately classify
some new categories of targets that beyond the predetermined
scenes. These studies provide references for the detection tasks
of some undiscussed targets or difficult-to-build datasets. The
relationship between ship and wake detection is also expected
to be further studied via the above methods.
C. Object Detection Dataset in Remote Sensing Community
The limitation of training samples is the biggest challenge
faced by deep learning remote sensing applications. Insuffi-
cient data acquisition has restricted the current development
of CNN-based methods in wake detection in SAR images.
Although, by observation, Wahl et al. [58] argued that dis-
cernable wakes are common in sea surface SAR images, there
are not many wake samples that researchers can actually use
since SAR is far less reliable than optical remote sensing
in terms of revisit cycle and data publicity. Many traditional
algorithms only used simulated SAR images to verify their
performance. In the preliminary practice of CNN-based SAR
wake detection [43], only 200 wake samples were available,
limiting the scale of the network that could be deployed.
Compared with SAR, optical remote sensing images
have better data publicity. In recent years, the mainstream
open-access optical remote sensing object detection datasets
have greatly promoted the corresponding research. Table I
summarizes their basic information. However, the ship
objects in the existing maritime object detection datasets
```
(e.g., HRSC2016 [62] and MASATI [65]) are still insufficient.
```
To solve the problem of small object detection in complex
backgrounds, researchers selected a large number of stationary
ships docked side by side as objects. Therefore, these datasets
are difficult to be transferred into the wake detection studies.
III. SWIM: A WAKE D ETECTION B ENCHMARK D ATASET
The utilization of the highly data-driven CNN object detec-
tor demands high standards on its training data, which includes
rich images and instances that can avoid network overfitting
to improve generalization. Accurate representation and appro-
priate annotation of instances ensure that the network can
extract the generic and discriminant features. To facilitate the
deep learning-based optical wake detection, we constructed
```
a ship wake imagery mass (SWIM) dataset. It contains
```
14 610 visible waveband satellite and aerial images and can
provide up to 15 356 accurately annotated wake instances.
The comparison between SWIM and other typical datasets
noted in Table I indicates that SWIM is currently a rare,
large-scale maritime object detection dataset. The SWIM
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
XUE et al.: RETHINKING AUTOMATIC SHIP WAKE DETECTION: STATE-OF-THE-ART CNN-BASED WAKE DETECTION 5613622
TABLE I
O PEN -A CCESS O BJECT D ETECTION D ATASETS IN R EMOTE S ENSING C OMMUNITY
Fig. 2. Typical images in the proposed SWIM dataset. SWIM Dataset contains a variety of ship wake images with different imaging qualities under various
```
sea states, illumination, and other environmental conditions. We labeled some images as “difficult” (365 in total) for some wake objects that could only be
```
```
confirmed by careful visual identification and added 3010 negative images without wake objects (pure sea surface, sea surface with coastal areas, and sea
```
```
surface with only stationary ships), which can be used on demand to make the training process more flexible.
```
dataset is now available for the academic research com-
munity to use at https://www.kaggle.com/lilitopia/swimship-
wake-imagery-mass.
A. Data Collection, Annotation, and Division
As described in Section I, the category and motion state
of a ship, imaging condition, sea state, illumination, weather,
and even the overlapping of wakes all contribute to the image
complexity of the wakes on the sea surface. To enable the
detector to extract general features from diverse backgrounds,
we collected images of coastal areas around Asia, Europe,
Africa, North America, and Oceania on Google Earth. All the
images were taken from 2009 to 2021. The objects to be
detected included the wakes of ships, ranging from small yacht
to huge container ships, and the backgrounds included open
seas, ports, straits, and canals, ensuring the intraclass diversity.
The images in the dataset have spatial resolutions of 2.5–0.5 m
and are uniformly cropped in size, 768 × 768 pixels.
Fig. 2 shows the typical image examples.
With the V-shaped Kelvin arm as its most prominent feature,
the wake instance is highly directional. Thus, we considered
using an OBB and a set of landmarks to accurately annotate
```
a wake instance [refer to Fig. 3(a)] and saved the data into
```
an XML file according to the format of PASCAL VOC
dataset [67].
The OBB annotating continues with the consistent practice
of the remote sensing community on highly directional objects:
the center point coordinate, width, height, and heading of the
```
OBB are encoded by five parameters (x c, y c, w, h, and θ) so
```
that the wake instance is completely enclosed in the box. The
angle θ is defined by the long-edge-based method: the rotation
angle of the long edge of HBB is defined as θ, clockwise
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
5613622 IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 60, 2022
```
Fig. 3. Visualization of the annotation methods. (a) SWIM uses an OBB
```
```
and a set of landmarks to accurately locate the wake instance. (b) Annotation
```
when the ship hull is outside the image range. It is worth mentioning that
when the wake is close to the image edge, using an HBB is optimal to avoid
```
the limitation of the image edge on the box; however, this also means that
```
more sea background is enclosed in the box, which is obviously unfavorable
to the texture-oriented wake detection.
```
is positive, and the range is (−π/2, π/2). (There is also an
```
OpenCV-based method defined in [54] that derives the angle
```
between the width of OBB and the x-axis as θ.)
```
In most cases, when the ship moves along a straight path, the
OBB’s heading is consistent with that of the wake. However,
when the ship turns, the wake also exhibits curvature, and the
OBB’s heading will be inconsistent from the wake heading
because the OBB still has to completely enclose the wake
instance. In this case, additional annotation is needed to
accurately locate the actual wake heading. Therefore, we set
```
the wake tip coordinates (x p, y p) and the directions of the
```
```
two Kelvin arms (θ1 , θ2) (the angle with the x-axis, clockwise
```
```
is positive, and the range is [0, π)) as landmarks to improve
```
the location accuracy. Depending on the actual requirements,
we can design a network that only predicts the OBB or predicts
both kinds of annotations.
All the images in the dataset are manually annotated by an
annotator with expert experience in wake recognition and then
validated by others. Since the boundary of the wake texture in
sea clutters is not always obvious, we minimize the ambiguity
of data annotation caused by the subjective judgment of the
annotator in the annotating process by strictly complying with
the following criteria.
```
1) The ideal case is when the ship moves along a straight
```
```
path and the two Kelvin arms are also clear; we use
```
OBB to completely enclose the wake instance, and let
the OBB’s heading run parallel to the angular bisector
```
of the two Kelvin arms (at this time, the angular bisector
```
of the Kelvin arms basically coincides with the turbulent
```
wake). In addition, we annotate the wake tip location and
```
the directions of the two Kelvin arms.
```
2) In the case when the ship moves in a straight line and
```
only one Kelvin arm is clear enough, we keep the OBB’s
heading parallel to the turbulent wake, adjust the OBB
size according to the more defined Kelvin arm, and then
operate symmetrically on the other side. In addition,
we annotate the wake tip location and the directions
of the two Kelvin arms according to the Kelvin wake
```
texture near the stern (Kelvin wake near the stern is the
```
strongest and usually generates observable signals on
```
both sides).
```
```
3) In the case when the ship turns and the Kelvin arms
```
exhibit curvatures, we rotate OBB from the direction
parallel to the turbulent wake, and let OBB completely
enclose the Kelvin wake on the premise of chang-
ing OBB’s heading as little as possible. In addition,
we annotate the wake tip location and the directions of
the two Kelvin arms near the stern.
```
4) In the case when the ship hull is out of the image,
```
```
as shown in Fig. 3(b), the annotation can be performed
```
only when most of the far-field wakes are in the image
```
field of view (FOV) and the annotator can fully confirm
```
that the incomplete object is the wake. The annotating
```
method of OBB is the same as above; we annotate the
```
intersection point of the turbulent wake and the image
```
boundary as the wake tip location; starting from this
```
point, we annotate the two directions parallel to the
Kelvin arms.
```
5) In the case when only the ship hull and the turbulent
```
```
wake can be observed (the Kelvin wake may not be
```
imageable in the optical image under unsatisfactory
imaging conditions, a very low observation altitude
angle may also lead to the overlapping of narrow Kelvin
```
wake and turbulent wake), and the annotation can be
```
performed only if the annotator can confirm the ship
hull and the clear turbulent wake. We let the OBB
completely enclose the turbulent wake and additionally
annotate the hull location and turbulent wake edges in
both directions.
```
6) The following are the cases where the instance is labeled
```
as “difficult.”
```
a) The annotator can confirm the presence of ship and wake,
```
but the quality of the wake signal is poor, making it difficult
to distinguish from sea clutter.
```
b) The ship has just started, and the Kelvin wake has no
```
more than five wavelengths, which cannot form an extractable
pattern.
```
c) Only a small part of the wake is in the image FOV. In this
```
case, the ship hull is usually located far away from the image
range. After the wake is diffused, its texture is weak and large,
which can be regarded as the sea background.
According to these criteria, all instances in the SWIM
dataset are clearly classified and annotated.
The SWIM dataset has more than 11.6k positive images.
```
We randomly selected 3/5 (6960 images) as the train-
```
```
ing set, 1/5 (2320 images) as the verification set, and
```
```
1/5 (2320 images) as the test set to ensure that the distribution
```
of instances in the training set and the test set was approxi-
mately matched.
B. Statistics of the Dataset
Since SWIM images are widely collected from various ship
wakes in major coastal areas around the world, the statisti-
cal information of the SWIM dataset shown in Fig. 4 also
reflects some characteristics of ship wakes in optical images.
```
Fig. 4(a) shows that the distribution of the number of wake
```
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
XUE et al.: RETHINKING AUTOMATIC SHIP WAKE DETECTION: STATE-OF-THE-ART CNN-BASED WAKE DETECTION 5613622
```
Fig. 4. Statistical histograms of instances in SWIM. (a) Number of instances in a single image. (b) Number of instances of different instance sizes (the long
```
```
side size of the bounding box). (c) Number of instances of different bounding box aspect ratios. (d) Number of instances of different bounding box rotation
```
```
angles. (e) Number of instances of different bisected angles. (f) Number of instances of different included angles.
```
instances in a single image in the SWIM dataset is balanced.
Even in traffic-intensive areas such as canals or straits, the
channel width limits the number of wake instances, which
alleviates the overlapping interference between multiple wakes
and enables the detector to learn more general features from
```
an independent wake instance. Fig. 4(b) shows that the wake
```
```
instance size in the SWIM dataset is moderate; most of the
```
instance size is 40%−60% of the image width, and it is easier
to learn wake features than to directly learn small hull features
```
in terms of size alone. The aspect ratio of the OBB in Fig. 4(c)
```
can represent the wake angle, which has two peaks near
```
0.8 [≈tan(39◦)] and 1.2 [≈1/tan(39◦)], indicating that most
```
wakes in the optical image still satisfy the classical Kelvin
```
wake angle. Fig. 4(d) shows the characteristics of the long-
```
edge-based rotation angle definition: since the ±90◦ rotation
of OBB is equivalent to the exchange of its width and height.
There is no instance in the SWIM dataset that has a rotation
```
angle of 90◦ (accordingly, there are more instances that have
```
```
a rotation angle of 0◦), which avoids the possible errors
```
in tangent calculation, and the OBB is symmetrically and
uniformly distributed in other directions. Similarly, it can be
```
observed in Fig. 4(e) that the angular bisector directions of the
```
two annotated Kelvin arms are evenly distributed, except that
the annotations near ±180◦ are equivalent to the annotations
```
near 0◦. Finally, Fig. 4(f) shows the distribution of the included
```
```
angle of the two annotated Kelvin arms (i.e., wake angle); all
```
wake angles in the dataset are acute angles, and there are two
peaks near 16◦ and 39◦, respectively, corresponding to only
the turbulent wake angle and the classical Kelvin wake angle,
where the peak near 39◦ is narrow and high, indicating that
there are far more instances with the classical Kelvin wake
angle than other instances. In addition, the wake angle can be
learned as a feature.
IV. WAKE N ET : P ROPOSED WAKE D ETECTION M ETHOD
In order to achieve automated wake detection, we designed
an end-to-end single-stage object detection network named
WakeNet. Its pipeline is shown in Fig. 5. WakeNet is mainly
```
composed of four modules: 1) a backbone network for feature
```
```
extraction; 2) an FPN with a newly designed MSAM that
```
enhances the contextual spatial relevance of feature maps at
```
different scales; 3) classic subnetworks for OBB classification
```
```
and regression; and 4) an additional subnetwork for landmark
```
regression. WakeNet performs forward inference in a single
step, so for an original optical remote sensing image, only two
preprocessing steps are required: rendering the image to RGB
format and cropping it into image piles of 768 × 768 pixels.
A. Selection of Backbone
The background in wake images mainly includes coastal
land and sea clutter. Since the wake itself is a sea wave rather
than a man-made object like a ship, the land-sea segmentation
is implemented implicitly during the training process, and
coastal interference is easily eliminated. On the other hand,
intuitively, the CNN-based method classifies wakes and sea
clutters by texture differences. However, wakes and sea waves
```
are superimposed and coupled; when the Kelvin arm signal is
```
not obvious, the boundary between the object and background
is also vague. In addition, even if the sea clutters in optical
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
5613622 IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 60, 2022
Fig. 5. Pipeline of WakeNet, our proposed single-staged wake detection network. WakeNet consists of a backbone using FcaNet, an FPN with an MSAM, and
a multitask regression subnetwork. It uses the additional supervision of hull coordinates and wake directions for multitask learning and outputs the multitask
loss of each anchor in the training phase.
images are Gaussian distributed, these waves may also produce
periodic textures similar to wakes in scale and pattern, causing
false alarms. Therefore, in order to extract low-SNR wake
objects from the sea clutter background, only the information
extracted by the texture difference is insufficient.
Fortunately, the wake spectrum provides us with extra
reliable information. Existing studies [68] show that the Kelvin
wake and the turbulent wake have stable X-shaped and linear
distribution patterns in the frequency spectrum, respectively.
```
As shown in Fig. 6(a), these patterns do not change their
```
shapes with the influence of sea states, illumination, and so
on and remain the same after the nonlinear MTF modulation
```
of the (optical or SAR) imaging system. On the contrary,
```
sea clutter exhibits a widely distributed speckle pattern in
the spectrum, which is quite different from the wake signal.
Based on these considerations, we choose ResNet integrated
```
with Fca modules (i.e., FcaNet) [34] as the backbone of our
```
detector. It can not only extract the wake texture features in the
image domain but also learn the wake features in the frequency
domain.
```
The Fca module performs discrete cosine transform (DCT)
```
on the feature map when implementing channel attention to
selectively extract the spectral features. The implementation is
```
shown in Fig. 6(b). The input feature X ∈ RH ×W ×C is first
```
divided into n groups [X0 , X1, . . . , Xn−1 ] along the channel
dimension. The number of channels in each group is C/n
```
(C should be divisible by n). Then, the 2-D DCT of each
```
group of the feature maps at different frequencies is calculated
```
freqk = DCT
```

Xk

=
H −1
```
i=0
```
W −1
```
j=0
```
```
B ki, j · Xki, j,: (1)
```
```
where Xk and freqk ∈ RC/n represent the kth (k ∈ {0, 1, . . . ,
```
```
n − 1}) group input feature maps and their 2-D DCT outputs;
```
```
Xki, j,: is the element of Xk at position (i , j ), which is a vector;
```
B ki, j is the element of the 2-D DCT basis Bk selected for the kth
```
group input (refer to [34] for the DCT basis selection rules).
```
Using the 2-D DCT outputs, the weight of the multispectral
channel attention can be calculated
```
weight = sigmoid
```

f c

concat

[freq0, freq1, . . . , freqn−1]

.
```
(2)
```
Finally, each channel of the input feature is reweighted to
obtain the final output ˜X of the Fca module
```
˜X = weight  X (3)
```
```
where  represents the Hadamard product (i.e., elementwise
```
```
multiplication). In practice, the Fca module is integrated
```
behind each residual module of ResNet. Wake detection rarely
uses color channel features, and the texture information of
```
the feature map between each channel is highly redundant;
```
by using the Fca module, we can extract features of different
frequencies from the input images into different channels of
the feature map to reduce the false alarm rate. Since it does not
involve a complex operation like FFT, the DCT basis is also
precalculated, and the Fca module ensures low computational
complexity.
B. FPNs Enhanced by the MSAM
As shown in Fig. 5, WakeNet uses levels 3–7 of the
FPN to extract multiscale features of the object. Due to the
particularity of the wake detection task, FPN may further
improve its performance in the following aspects.
```
1) Kuo and Chen [18] proved that the spatial correlation
```
operation of wake images in adjacent scales can greatly
enhance the edge features of wakes, and FPN just
provides us with a large number of feature maps that
```
are highly correlated on adjacent scales; however, the
```
multiscale feature fusion in FPN is very direct since it
only upsamples the high-level feature map to the same
resolution as the low-level feature map and then directly
adds it to the low-level feature map, which fails to make
full use of the internal spatial correlation between the
feature maps in adjacent scales.
```
2) When the wake internal texture is not obvious, the
```
network mainly extracts linear features of the Kelvin
```
arm and the turbulent wake; these linear features are
```
generally narrow, so the local receptive field generated
by continuous convolution operations will cause large
spatial errors in high-level semantic features. Because
the feature fusion of FPN lacks context, the inconsis-
tency of the semantic features in the high-level feature
map and the spatial features in the low-level feature
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
XUE et al.: RETHINKING AUTOMATIC SHIP WAKE DETECTION: STATE-OF-THE-ART CNN-BASED WAKE DETECTION 5613622
```
Fig. 6. Selection of backbone. (a) Wake images (after grayscale processing) and their frequency spectra; the frequency spectra show obvious pattern
```
distributions. The bright spot in the center of the spectrum and the linear low-frequency component correspond to the wake tip and the turbulent wake,
```
respectively; the linear component in the spectrum is perpendicular to the actual turbulent wake direction. The X-shaped high-frequency component corresponds
```
to the far-field Kelvin wake, where the inner region near the inflection point corresponds to the transverse wave component and the two outer regions correspond
to the divergent wave component. The inflection point position and the boundary between the two components are determined by the ship velocity. Sea clutter
corresponds to random noise in the spectrum. The transverse wave component in wake 2 is basically invisible, so the corresponding part of its spectrum
```
cannot be observed, but the X shape does not change. (b) Comparison between the original ResNet residual module and the adopted Fca module. The Fca
```
module performs the 2-D DCT with different frequencies for each feature map when achieving channel attention, which has advantages in learning stable wake
```
frequency domain features. (c) Fca module generates weights by splitting the input feature maps and multiplying them element-by-element by the preselected
```
DCT bases of different frequencies and weighting the different channels of feature maps to achieve channel attention.
map will lead to conflicts between classification and
localization tasks.
```
3) Unlike common overlapping objects that occlude each
```
other, the spatial textures of overlapping wakes are
```
also superimposed on each other; a single-level feature
```
map in FPN cannot distinguish the overlapping wake
objects effectively only by the coarse-grained high-level
semantic features, resulting in missed detection.
To solve these problems, we propose an MSAM inspired
by the attention mechanism [69]. Under the premise that the
computational cost is not too high, MSAM improves the fea-
ture fusion scheme of FPN. It selectively encodes contextual
information with strong spatial correlation between adjacent
scales in each element of the feature map in a weighted
manner, thereby improving the feature discrimination ability
of FPN at various scales.
We apply MSAM on levels 3–5 of the FPN, and the pipeline
is shown in Fig. 7. Similar to FPN, SWIM first passes the low-
level feature map Cl ∈ RHl ×Wl ×Cl and the high-level feature
map Ch ∈ RH h ×Wh ×C h on the adjacent scales through the 1 × 1
convolutional layer to generate feature maps Ml ∈ RHl ×Wl ×C o
and Mh ∈ RH h ×Wh ×C o with the same number of channels.
Then, Ml and Mh are reshaped into long vectors Al ∈ RNl ×C o
and Ah ∈ RN h ×C o along the channel dimension, where
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
5613622 IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 60, 2022
Fig. 7. Proposed MSAM. This module calculates the heatmap to model the inherent spatial correlation between adjacent-scale feature maps and can extract
the wake edge features better than the original feature fusion method of directly upsampling the high-level feature maps and then adding them to the low-level
feature maps.
```
Nl = Hl × Wl and Nh = H h × Wh , and feature vectors
```
Bl and Bh are obtained through the fully connected layer to
further reduce the number of channels to be processed. Next,
```
matrix multiplication is performed on (Bl )T and Bh (where
```
```
superscript T means transpose) and apply a softmax layer to
```
calculate the spatial attention heatmap H between the feature
maps at different levels
h j,i =
exp
c0 −1
```
k=0 b hj,k · b li,k
```

Nl −1
```
i=0 exp
```
c0 −1
```
k=0 b hj,k · b li,k
```
```
 (4)
```
where b l , b h , and h are elements in Bl , Bh , and H, respectively,
and h j,i measures the influence of the correlation between the
i th position of the low-level feature map and the j th position
of the high-level feature map on the j th position of the
high-level feature map. Finally, the matrix multiplication is
performed on HT and Ah , and the result is reshaped back to the
```
shape of RHl ×Wl ×C o and weighted by the parameter α; the final
```
output low-level feature map Pl is obtained by elementwise
summing the result above and the feature map Ml
```
Pl = α
```

reshape

HT Ah

- Ml . (5)
The coefficient α is initialized to 0 and gradually adjusted
```
via training. Equation (5) indicates that the spatially correlated
```
semantic features in the high-level feature map are selectively
aggregated into the low-level feature map by reweighting.
C. Multitask Regression
In order to obtain robust performance, WakeNet adopts an
anchor-based strategy to detect wake objects by regressing
the predefined OBBs. It needs to be clarified again that our
motivation of using OBB to detect wake stems from the fact
that the wake components clamped by the Kelvin arms in the
optical image have highly characteristic textures. We believe
that even without considering the false alarms caused by
nonsea clutter such as harbors, detecting the entire wake
should still obtain more robust results than detecting only the
Kelvin arms or the turbulent wake. Therefore, we retained
the classification head and the OBB regression head used in
```
the common detection tasks (the top of Fig. 8). Moreover,
```
we also realize that Kelvin arm is usually the most obvious
```
feature of the wake. Thus, we designed a new head (the bottom
```
```
of Fig. 8) for regressing the landmarks, including the wake tip
```
coordinates and the Kelvin arm direction to make full use of
these semantic line features and to improve the performance
in low SNR case. In addition, these landmarks indicate the
wake heading, which is not available from OBB regres-
sion. In summary, for the outputs of the above three heads,
WakeNet minimizes the following multitask loss in the training
```
phase:
```
```
L( p, t, q)
```
```
= Lclass( p, p∗) + λ1 LOBB(t, t∗) + λ2 Llandmarks(q, q∗) (6)
```
where Lclass , LOBB , and Llandmarks represent the classifica-
tion loss, OBB regression loss, and landmark regression
loss, respectively. The hyperparameters λ1 and λ2 are set to
1.0 and 0.3, respectively, showing that we increase the sig-
nificance of OBB regression, while the landmarks are mainly
used for auxiliary supervision.
```
1) Classification: The structure of the classification head is
```
shown in the uppermost subnetwork in Fig. 8. Consistent with
mainstream object detectors, WakeNet uses the focal loss [49]
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
XUE et al.: RETHINKING AUTOMATIC SHIP WAKE DETECTION: STATE-OF-THE-ART CNN-BASED WAKE DETECTION 5613622
Fig. 8. Implementation details of the three subnetworks: classification, OBB regression, and landmark regression. Among them, the classification and OBB
```
regression subnetworks use the traditional regression heads; the newly designed landmark regression subnetwork consists of two branches, which are used to
```
regress the wake tip coordinates and the Kelvin arm direction. The angle branch uses RT to improve the linear feature detection performance.
as the classification loss
```
Lclass( p, p∗) = 1A
```
A
```
i=1
```
```
Lfocal( p i , p∗i ) (7)
```
```
Lfocal( p i , p∗i ) = −(1 − ˆp i ) γ log( ˆp i )
```
ˆp i = p i , if p
∗i = 1
```
1 − p i , otherwise (8)
```
```
where p i is the predicted score of the i th anchor; the total
```
anchors number is A and p∗ i is the ground-truth label of
the i th anchor. The anchors having the highest IoU overlap
or IoU overlap greater than 0.5 with the ground-truth boxes
```
are regarded as positive anchors, p∗ i = 1; the remains are
```
negative anchors, p∗ i = 0. In this article, the hyperparameter
γ used to focus on difficult samples is set to 2.0.
```
2) OBB Regression: The structure of the OBB regression
```
head is shown in the middle subnetwork in Fig. 8. The subnet-
```
work outputs the offset t = (t x c , t y c , tw , th , tθ ) of the predicted
```
```
OBB parameters (x c, y c, w, h, and θ) relative to the predefined
```
```
anchor parameters (x ac , y ac , wa , h a , and θa )
```
t x c =

x c − x ac

/wa , t y c =

y c − y ac

/h a
```
tw = log
```

w/wa

, th = log

h/ h a

```
(9)
```
tθ = tan

θ − θ a

```
. (10)
```
```
Similarly, we encode the offset t∗ = (t∗x c , t∗y c , t∗w , t∗h , t∗θ ) of
```
```
the OBB ground-truth values (x∗c , y∗c , w∗, h∗, and θ ∗) relative
```
to the anchor parameters
t∗x c =

x∗c − x ac

/wa , t∗y c =

y∗c − y ac

/h a
t∗w = log

w∗/wa

, t∗h = log

h∗/ h a

t∗θ = tan

θ ∗ − θ a

```
. (11)
```
```
The loss function LOBB (t, t∗) calculates the regression loss.
```
Here, we notice that when using the common smooth L1
loss [46] that is inconsistent with the rotation IoU, the OBB
regression may be limited. By using the long-edge-based
Fig. 9. Two possible cases where OBB regression may be limited when
```
using a smooth L1 loss that is incompatible with IoU. (a) Generally, the OBB
```
rotating at a certain angle and the OBB reverse rotating the supplementary
angle corresponds to the same IoU, while the ground truth has only one angle,
leading to a large loss in one of the situations. This issue can be avoided
by using the long-edge-based rotation angle definition and adding tangent
```
functions. (b) OBB rotating at a certain angle and the OBB reverse rotating
```
the complementary angle after resizing its height and width corresponding to
the same IoU, while the use of smooth L1 loss will obviously cause a large
loss in one of the situations. This problem cannot be avoided.
rotation angle definition method and by adding the tangent
```
function [i.e., tθ in (8) and t∗θ in (9)], the problem of unequal
```
loss values when OBB regression angles are supplementary,
```
as shown in Fig. 9(a), can be solved. Nevertheless, the problem
```
```
shown in Fig. 9(b) cannot be solved.
```
Through our experiments, we find that IoU-based loss
functions cannot effectively generate gradients because the
aspect ratios of the wake objects are not very high when
compared to that of the ship objects. To solve the above
problems, we use a more intuitive five-parameter L5pmr loss
```
function [70] while also retaining the advantage of smooth
```
L1 loss, which demonstrates that the gradient is small when
the difference is small
```
LOBB(t, t∗)
```
= −
A
```
i=1 p∗i L5pmr(t, t∗)A
```
```
i=1 p∗i
```
```
(12)
```
```
L5pmr = min ∇lcp + ∇lwh1 + ∇lθ 1, ∇lcp + ∇lwh2 + ∇lθ 2
```
```
∇lcp = LsmoothL1(t∗x c , t x c ) + LsmoothL1(t∗y c , t y c )
```
```
∇lwh1 = LsmoothL1(t∗w , tw ) + LsmoothL1(t∗h , th )
```
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
5613622 IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 60, 2022
Fig. 10. Discrete RT performed on the feature map: a feature line in the
```
image domain is accumulated to a feature point in the Radon domain; the
```
feature line clusters in the feature map are transformed into point clusters
after passing through the RT layer. The figure also shows the definition of the
resolution of a line in the image domain and the discrete Radon domain.
```
∇lwh2 = LsmoothL1(t∗w , th ) + LsmoothL1(t∗h , tw )
```
```
∇lθ 1 = LsmoothL1(t∗θ , tθ ), ∇lθ 2 = LsmoothL1(t∗θ , tθ − π2 )
```
```
LsmoothL1(t∗, t)
```
```
= 0.5(t
```
```
∗ − t)2, if |t∗ − t| < 1
```
```
|t∗ − t| − 0.5, otherwise. (13)
```
Therefore, the network will automatically select the equiv-
alent regression path with a smaller loss, and the problem in
```
Fig. 9(b) can be solved.
```
```
3) Landmark Regression: The structure of the landmark
```
regression head is shown in the lowermost part of Fig. 8.
It uses an independent coordinate branch and an angle branch
to regress the wake tip coordinates and the Kelvin arm
direction. Regardless of whether the ship hull is visible or not,
```
the wake tip exhibits as an intuitive point feature semantically;
```
therefore, compared with the OBB regression head, the coor-
dinate branch only reduces the convolutional layers to achieve
a lightweight status. On the other hand, the CNN convolution
kernel with rectangular local receptive field cannot efficiently
extract and aggregate the discontinuous lines of the Kelvin
arm, and directly extracting the direction of the semantic line
```
from the feature map is also not intuitive; thus, we recall
```
the traditional method and add a discrete RT layer 1 [71] at
the forefront of the angle branch to improve the regression
accuracy by transforming the line features in the feature map
into the peak point features in the Radon domain.
Some implementation details of the RT layer are worth
explaining. As shown in Fig. 10, given a feature X ∈ RH ×W ×C ,
the discrete obtains its feature distribution Y ∈ R×P×C
```
in the discrete Radon domain; the transform is realized by
```
accumulating pixels along a line l in the feature map into a
```
point ( ˆθl , ˆρl ) in the Radon domain:
```
```
Y( ˆθl , ˆρl ) =
```

i∈l
```
X(i )
```
ˆθl = θl
θRdn

ˆρl = ρlρ
Rdn

```
. (14)
```
1 Hough transform is actually adopted in our method to pursue higher
computational efficiency and a simple, differentiable implementation. In fact,
if one is not too pedantic, discrete Radon transform and Hough transform
are mathematically equivalent. In keeping with the terms used consistently in
```
wake detection studies, we still use the term “Radon transform (RT)” here.
```
```
The last two equations in (14) represent the discrete quan-
```
tization process, where θl and ρl are the continuous values
corresponding to the line l in the Radon domain. RT traverses
all the lines in the feature map, and θl and ρl thus have
```
ranges of [0, π) and [−(H 2 + W 2)1/2/2, (H 2 + W 2 )1/2/2],
```
respectively. θRdn and ρRdn represent the distance and
angular resolution, respectively, so that one can distinguish two
lines in the discrete Radon domain. The distance and angular
resolution θimg and ρimg that one can distinguish two lines
in the feature map are calculated by the following equation 2:
θRdn = π , ρRdn =
√
H 2 + W 2
P
θimg ≈ arctan
 1
```
max{H, W } − 1
```

```
, ρimg ≈ 1. (15)
```
We set the output feature map size of the RT layer appro-
priately so that the distance resolution in the Radon domain is
slightly lower than that in the image domain on the premise
```
of high angular resolution. The height (angle dimension) of
```
the RT layer output feature map is set to four times that of
```
the input feature map, while the width (distance dimension)
```
remains unchanged. At this time, in the layers P3 –P7 of FPN,
the angular resolutions of the output feature maps are 0.66,
0.72, 0.75, 0.77, and 0.78 of the input feature map resolutions,
while the distance resolutions of outputs are all 1.41 of the
input feature map resolutions. Through the RT layer, the
parallel line clusters with a certain width in the input feature
map are aggregated into dense point clusters in the output
feature map. Using the subsequent convolutional layers to
further aggregate the context of adjacent elements, the angle
branch implicitly performs the nonmaximum suppression of
linear features.
After concatenating the feature maps of the coordinate and
```
angle branches, similar to (9), the network outputs the offset
```
```
q = (q x p , q y p , qθ1 , qθ2 ) of the predicted parameters (x p, y p,
```
```
θ1 , and θ2) relative to the predefined parameters (x ap, y ap, θ a1 ,
```
```
and θa2 ) (further encoded by the anchor parameters)
```
q x p =

x p − x ap

/wa =

x p − x ac

/wa
q y p =

y p − y ap

/h a =

y p − y ac

/h a
qθ1 = tan

θ1 − θ a1

, qθ2 = tan

θ2 − θ a2

```
. (16)
```
```
In the same way, similar to (10), we encode the offset
```
```
q∗ = (q∗x p , q∗y p , q∗θ1 , q∗θ2 ) of the landmarks ground truth (x∗p, y∗p,
```
```
θ∗1 , and θ∗2 ) relative to the predefined parameters (x ap, y ap , θ a1 ,
```
```
and θa2 )
```
q∗x p =

x∗p − x ap

/wa =

x∗p − x ac

/wa
q∗y p =

y∗p − y ap

/h a =

y∗p − y ac

/h a
q∗θ1 = tan

θ ∗1 − θ a1

, q∗θ2 = tan

θ∗2 − θ a2

```
. (17)
```
```
The encoding of predefined parameters (x ap, y ap, θa1 , and θ a2 )
```
```
from anchor parameters (x ac , y ac , wa , h a , and θ) follows special
```
2 Readers may notice that for a square pixel image, the minimum distance
and angle that one can distinguish two lines that vary with the rotation of
```
the line. Therefore, (15) actually defines the maximum possible minimum
```
distance and angle to ensure generality. Imagine an extreme case: for an N × 2
```
image (N = 2), its corresponding distance and angular resolutions are only
```
```
equal to (15).
```
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
XUE et al.: RETHINKING AUTOMATIC SHIP WAKE DETECTION: STATE-OF-THE-ART CNN-BASED WAKE DETECTION 5613622
```
Fig. 11. Encoding and decoding rules for landmarks. (a)−(c) In the training process, the encoding of the landmarks ground truth and the predefined landmark
```
```
parameters are noted in the following. The wake tip coordinates are regressed from the center of the anchor (x ac , y ac ) to the ground-truth point (x∗p , y∗p ). One
```
must always define the sequence of the two ground-truth Kelvin arms directions in a clockwise rotation direction, according to the “acute angle criterion,”
θ a1 and θa2 that are predefined in the direction of the semimajor axis of the anchor box, and the ranges of both the ground truth and the predefined landmarks
```
are limited to (−π /2, π /2]. The regression path guided by the loss function and its equivalent path are marked with dark gray and light gray arrows in the
```
```
figure. (d) In the test process, to decode the Kelvin arm direction angle, connect a line from the predicted wake tip point (x p , y p ) to the center of the anchor
```
box, and take the semimajor axis direction of the anchor box with the smaller angle to the line as the predefined direction. Then, the ambiguity can be
disambiguated by controlling the network output offsets as an acute angle.
```
rules, as shown in Fig. 11. According to (16) and (17), the
```
```
center point coordinates of the anchor (x ac , y ac ) is directly used
```
as the predefined initial value of the wake tip coordinates
```
(x ap, y ap). Predefining the Kelvin arm direction according to
```
the anchor direction requires more careful discussion, because
even if a rotating rectangular box has spatial directivity, its
180◦ rotation invariance along the long/short axis still causes
a four-direction-rotating ambiguity. Many observations and
hydrodynamical experiments show that it is almost impossible
for a ship to produce wakes with a wake angle greater than
or equal to 90◦ under natural conditions, which we call the
“acute angle criterion.” In fact, the interference of bow and
stern waves only makes it more likely that the wake angle
```
will be less than the classical Kelvin angle; wake instances in
```
SWIM dataset also demonstrate this aspect. According to this
criterion, the predefined wake direction should obviously be
consistent with the long axis direction of the anchor, and the
two cases corresponding to the obtuse wake angle are therefore
dismissed. At this time, the predefined wake direction still
has a 180◦ rotation ambiguity along the long axis. Therefore,
we use the tangent function during coding and limit the range
```
of the predefined direction and the ground truth to (−π, π]
```
in the training phase so that the two ambiguities will have the
```
same training loss and regression path [refer to Fig. 11(a)–(c)].
```
In the test phase, since the detector has been able to predict the
wake tip location accurately, predicted wake tip coordinates are
used as a priori to first determine the actual heading of the
```
wake [refer to Fig. 11(d)] to eliminate the direction ambiguity
```
of the Kelvin arm.
Finally, Llandmarks adopted the balanced L1 loss [72] to
enable the detector to focus on the regression of the Kelvin
arms with low SNR
```
Llandmarks(q, q∗)
```
=
A
```
i=1 p∗i
```

```
j∈{x p ,y p ,θ1,θ2 } LbalancedL1
```

q∗j − q j

A
```
i=1 p∗i
```
```
(18)
```
```
LbalancedL1( q)
```
=
⎧
⎪⎨
⎪⎩
α
```
b (b| q|+1) ln
```
 b
β | q|+1

− α| q| if | q| < β
γ | q|+ γb − αβ otherwise
```
b = eγ /α − 1. (19)
```
In this article, α = 0.5, β = 1.2, and γ = 0.9.
V. E XPERIMENTS
This section introduces experimentation, including imple-
mentation details, comparison of the detection results with
other CNN-based methods on the SWIM dataset, the ablation
study, and detection results on the real-shot wake polarization
images.
A. Implementation Details
```
1) Dataset: The SWIM dataset is used as the benchmark
```
dataset, and its details and settings are given in Section III. The
size of the training images is kept as the original size of SWIM
```
dataset images (768 × 768 pixels). Since the wake dataset has
```
no research precedent, to ensure that the SWIM dataset can
provide stable performance and fair comparison, we use all the
11 600 positive images for training and evaluation, while the
negative images and the 365 instances labeled as “difficult” are
currently ignored. Data augmentation includes random color
distortion, random horizontal and vertical flips, and random
affine transformation, all with a probability of 0.5.
```
2) Evaluation Metric: We primarily use the mainstream
```
```
average precision (AP) in PASCAL-VOC [67] to evaluate
```
the performance of WakeNet and other CNN-based methods.
AP is the average value of precision under different recalls.
The definitions of precision and recall indicators are as
```
follows:
```
```
Precision = TPTP + FP (20)
```
```
Recall = TPTP + FN (21)
```
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
5613622 IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 60, 2022
Fig. 12. Some examples of the detection results obtained by our WakeNet on the SWIM test set. The score threshold of the bounding boxes shown in
```
the figure is 0.5. The green boxes and the blue line segments indicate correct detection; the red boxes and yellow line segments indicate missed detection.
```
```
(a)–(h) Examples of correct detection. (i)–(l) Examples of missed detection.
```
```
where true positive (TP) denotes the number of correctly
```
```
detected objects, false positive (FP; i.e., false alarm) denotes
```
the number of incorrectly detected objects, and false negative
```
(FN) denotes the number of missed objects. Notably, since the
```
other CNN-based methods cannot predict the wake tip location
and the Kelvin arm direction, the above indicators are still
based on whether the IoU of the predicted bounding box and
ground-truth box is greater than 0.5. The detection result of
WakeNet on the wake tip and Kelvin arms is currently only
shown in its output
```
3) Training Settings: All models are implemented in the
```
Pytorch framework and with a single NVIDIA RTX 3090
```
(24 GB) GPU. When using FcaNet-50 and FcaNet-101 as the
```
backbone, the batch size is set to 15 and 12, respectively.
We use the Adam optimizer, the learning rate starts from
0.0001 and then is divided by 10 at 70 and 90 epochs, and
the training terminates at 100 epochs. In addition, for training
efficiency, the NVIDIA APEX mixed-precision training toolkit
is used.
B. Representative Detection Results and Ablation Study
```
1) Typical Detection Results: We first present some rep-
```
resentative detection results of WakeNet on the SWIM test
```
set in Fig. 12. As shown in Fig. 12(a)−(h), WakeNet is
```
robust enough to cope with the various adverse environmental
conditions, such as weak wake objects, strong sea clutter
noise, partially occluded wake objects, floating algae on the
sea surface, and sun glints. At the current score threshold,
WakeNet also produces virtually no false alarms on the SWIM
test set. Apart from the case that WakeNet fails to detect
the object due to too few wake texture features shown in
```
Fig. 12(i), missed detection mainly occurs when multiple
```
highly overlapping wakes are generated by several ships
with the same heading and similar velocities. In these cases,
WakeNet usually can only detect one of the multiple wakes,
```
as shown in Fig. 12(j)−(l). This is because the textures of the
```
wakes with the same heading and ship velocity are highly
coupled, and the X-shaped spectrum components are also
coincidental. It is difficult for WakeNet to effectively extract
sufficient features to distinguish the two wake objects at this
TABLE II
A BLATION S TUDY OF WAKE N ET ON THE SWIM VALIDATION S ET
time. The fact that WakeNet can better detect the two back-
```
propagating wake objects in Fig. 12(c) and (h) and multiple
```
```
nonoverlapping wake objects in Fig. 12(d), (f), and (g) [even
```
though the wake textures are not as obvious as those in
```
Fig. 12(j) and (l)] also indirectly demonstrates this result. This
```
problem may be further solved by adding an instance-level
denoising module [54].
```
2) Ablation Study: We also conduct a series of ablative
```
experiments to determine how FcaNet, MSAM, and the wake
landmark regression subnetwork contribute to the wake detec-
tion performance. Several different configurations are adopted
for the ablation study, mainly including Baseline—essentially
the RetinaNet that directly expands the original HBB regres-
sion head and smooth L1 loss to detect rotating objects,
```
using ResNet-50 or ResNet-101 as its backbone; Baseline +
```
FcaNet—the baseline with only FcaNet-50 or FcaNet-101
```
as its backbone; Baseline + MSAM—the baseline with the
```
```
MSAM included only; and WakeNet—as shown in Fig. 5,
```
the full implementation of the proposed modules, with the
landmark regression subnetwork included.
Table II shows the experimental results of the above con-
figurations on the SWIM validation set. Using ResNet-50
and ResNet-101 as its backbone, the baseline reached APs
of 72.72% and 73.41%, respectively. The network achieved
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
XUE et al.: RETHINKING AUTOMATIC SHIP WAKE DETECTION: STATE-OF-THE-ART CNN-BASED WAKE DETECTION 5613622
Fig. 13. Some visualized results of the C3−C5 level feature maps of the backbone and the P3−P5 level feature maps of the FPN. Due to the large number
of channels in the feature map of each level, we selected the two with the most obvious visual effects from the results of different configurations to display in
```
this article. (a) Curved wake detection in the presence of a terrestrial background. (b) Container ship wake detection in a pure ocean background. (c) Detection
```
of the two ship wakes in a complex scene.
an AP improvement of 0.45% and 0.59% after replacing
the backbone, demonstrating the effectiveness of the spectral
information in wake feature extraction. The network achieved
an AP improvement of 0.43% and 1.02% after adding MSAM,
proving that using the spatial correlation between adjacent
scales can enhance the wake objects. For a deeper backbone,
the performance improvement is even more pronounced by
strengthening the context at adjacent scales. Applying these
two modules together on the 50- and 101-layer networks
achieves 1.61% and 1.60% AP boost, respectively, proving that
the two modules are complementary to each other. On these
bases, by modifying the loss function, the network increases
AP by 0.49% and 0.21%, which proves that for an arbitrary-
oriented object, even if its aspect ratio is not very high, the
detection performance can still be improved by modifying the
loss function. Furthermore, WakeNet performs an additional
wake landmark regression that significantly improves the wake
```
detection performance (1.48% and 2.04%), indicating that the
```
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
5613622 IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 60, 2022
linear feature of the Kelvin arm is still a critical feature of
wakes in optical images. Finally, by increasing the backbone
depth from 50 to 101 layers, the AP improvement of the
baseline and WakeNet is 0.69% and 0.96%, respectively,
showing that the proposed methods can better utilize the
feature representation capabilities provided by the deep neural
network.
In order to demonstrate the feature extraction capabili-
ties of the adopted Fca module and the proposed MSAM
module more intuitively, we visualize the C3−C5 layers of
the backbone and the P3−P5 layers of the FPN under two
```
different network configurations (i.e., whether the Fca module
```
is used in the backbone and whether the MSAM module is
```
used in the FPN). Fig. 13(a) shows the layers’ responses to
```
a curved wake in the presence of a terrestrial background.
We can see from C3−C5 layers that the original ResNet is
not strong in distinguishing the terrestrial background and the
```
wake; the wake edges in all levels of feature maps are very
```
fuzzy, and the semantic responses in the high-level feature
maps also have loose ranges. In contrast, the responses of the
FcaNet layers are clearer, more accurate, and stronger. Besides,
it can be seen from P3−P5 layers that the original FPN is
inferior to the FPN using the MSAM module in terms of the
feature locating accuracy of the low-level feature maps and
the semantic feature extraction of the high-level feature maps.
The FPN using the MSAM module can produce consistent
responses from low-level to high-level feature maps, and the
```
curvature of the wake is also very clear. Fig. 13(b) shows
```
the layers’ responses to the container ship wake in a pure
ocean background. It can be seen from C3−C5 layers that the
Fca module can significantly improve the feature extraction
of Kelvin arms and turbulent wakes and suppress sea clutter
noises. Also, P3−P5 layers responses indicate that the MSAM
significantly enhances the FPN’s response to the striped edge
features of the Kelvin arms and turbulent wakes, and this not
only facilitates the subsequent OBB locating but also makes
the subnetwork containing the RT layer easier to perform the
Kelvin arm detection. The same is true for the analysis of
the layers’ responses to the two wakes in a complex scene in
```
Fig. 13(c). Observing the P4 and P5 layers, it can be found that
```
the FPN using the MSAM module also has a stronger response
to the inconspicuous wake on the right side of the image. The
above experimental results prove that the performance of the
modules used is consistent with their design logic.
C. Comparison With the CNN-Based Methods
We compare the experimental results of WakeNet and
some other representative CNN-based methods on the SWIM
test set. The methods involved are R2 CNN [73], R-RetinaNet,
R-YOLOv3, and BBAVectors [74], which cover the current
main object detection network structures and arbitrary-oriented
object detection implementation strategies. R 2 CNN uses the
two-stage Faster R-CNN as the baseline and obtains the OBB
through RoI pooling. R-RetinaNet performs OBB detection
by directly regressing the angle parameter on the basis of
single-stage anchor-based RetinaNet’s four-parameter HBB
detection [49]. R-YOLOv3 modifies the single-stage anchor-
TABLE III
Q UANTITATIVE C OMPARISONS OF THE M ETHODS
ON THE SWIM T EST S ET
Fig. 14. Precision–recall curves of different methods on the SWIM test set.
based YOLOv3 [48] to regress the four vertex coordinates
of the OBB. BBAVectors adds an orientation map into the
heat map branches of the single-stage anchor-free CenterNet
and captures the OBB by regressing the box boundary-aware
vectors. Table III shows the quantitative comparison result.
Our WakeNet outperforms the others remarkably, verifying
its effectiveness in ship wake detection. Among the methods
used for comparison, the single-stage RetinaNet has an AP
higher than other methods. The performances of the two-stage
R2 CNN and the anchor-free BBAVectors are close, while the
performance of the single-stage R-YOLOv3 is poor.
Fig. 14 shows a more detailed comparison of the precision–
recall curves. It can be seen that through a number of improve-
ments in wake feature extraction, WakeNet can always have
higher wake detection precision than other methods under
most recalls. The precision of WakeNet, R-RetinaNet, and
BBAVectors is close at lower recall, but as the recall increases,
the precision of R-RetinaNet and BBAVectors declined more
rapidly, thus failing to match the WakeNet’s performance.
R2 CNN and R-YOLOv3 have similar precision when the
recall is low, but the precision of R2 CNN does not decrease
significantly as the recall increases, so it has an AP similar to
that of the BBAVectors. The precision of R-YOLOv3 is always
lower than other methods.
Fig. 15 shows a visual comparison of typical examples. It is
obvious that because of the landmark regression branch, Wak-
eNet can predict the wake tip location and the direction of the
```
two Kelvin arms while detecting the wake object. Fig. 15(a)
```
shows the wake detection results in the complex sea surface
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
XUE et al.: RETHINKING AUTOMATIC SHIP WAKE DETECTION: STATE-OF-THE-ART CNN-BASED WAKE DETECTION 5613622
Fig. 15. Some examples of the detection results obtained by the different methods on the SWIM test set. The score threshold of the bounding boxes shown in
```
the figure is 0.5. The green boxes indicate correct detection; the red boxes indicate false alarms. (a)–(c) Examples of single target detection. (d)–(g) Examples
```
of multiple target detection.
```
background; the single-stage anchor-based R-RetinaNet and
```
R-YOLOv3 fail to detect the wake objects, and the anchor-
free BBAVectors produces false alarms in the sea clutters,
```
while the remaining methods perform better. Fig. 15(b) shows
```
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
5613622 IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 60, 2022
Fig. 16. Some examples of the detection results obtained by the SOTA RT-based method [21] and our WakeNet on the SWIM test set. The white lines and
the black lines in the RT-based method results represent the detected bright and dark wakes, respectively. The score threshold of the bounding boxes shown
```
in the WakeNet results is 0.5. (a)–(d) Examples of single target detection. (e)–(f) Examples of multiple target detection.
```
the situation where the front part of the wake is occluded.
With the exception of R-RetinaNet and WakeNet that predict
OBBs completely surrounding the wake objects, other methods
```
are affected to varying degrees; the accuracy of WakeNet’s
```
prediction of the wake tip coordinates is also decreased due
```
to the occlusion. In Fig. 15(c), R2 CNN, R-YOLOv3, and
```
BBAVectors all detect the large-size freighter hull as the object,
indicating that in the absence of customized feature extraction
and auxiliary supervision, the detectors are not clear about
```
the objective of the wake features learning. Fig. 15(d) and
```
```
(e) shows the detection results in the cloud coving region
```
and the sea surface sunglint region. For objects with low
```
SNR, WakeNet shows good robustness; it has neither missed
```
detection due to cloud occlusion like other single-stage net-
works nor false alarms due to sunglints like other two-stage
```
networks and anchor-free networks. Fig. 15(f) and (g) shows
```
the detection results of multiple wake objects. The perfor-
```
mances of the several methods are similar; in comparison,
```
WakeNet locates OBB more accurately and is less prone to
produce false alarms. All in all, the above detection results
demonstrate the higher performance and practical application
value of WakeNet.
D. Visual Comparison With the SOTA RT-Based Method
To demonstrate the superiority of the CNN-based detector
compared to the RT-based method in optical wake detection,
we reproduce the state-of-the-art RT-based optical wake detec-
```
tion algorithm introduced in [21] (some hyperparameters of the
```
```
algorithm are modified according to the SWIM dataset) and
```
compare it with our WakeNet on the SWIM test set. According
to the difference between the wake optical images and the
wake SAR images, the RT-based algorithm first performs
normalized RT line detection on the subimages with the ship
hull in the center and then removes false alarms through
the postprocessing steps including pixel value verification,
turbulent wake identification, included angle verification, and
contrast verification and finally output turbulent wake and
Kelvin arm lines. It should be noted that the algorithm needs to
generate subimages with the ship hull in the center to achieve
high performance, and this is on the premise of ship detection.
Although we only focus on the wake detection process and
follow the authors’ method to directly input the ground-truth
wake tip coordinates into the algorithm as the priori hull
position, the method of “detecting the ship hull first and then
detecting the wake” actually violates the motivation of wake
detection. It is also unable to detect some special wakes such
as the submarine wakes without hull, and the wakes shown in
```
Fig. 13(b) and (d) that are not connected to the hull due to
```
cover. In addition, because using only RT is obviously not a
good way to extract curved wakes, we do not analyze such
cases. These show the inflexibility of the RT-based method.
Fig. 16 shows a comparison of typical detection results of
```
the two methods. Fig. 16(a) shows an ideal situation; it can
```
be seen that both methods detect the turbulent wake and the
```
two Kelvin arms well. The sea clutter noise in Fig. 16(b)
```
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
XUE et al.: RETHINKING AUTOMATIC SHIP WAKE DETECTION: STATE-OF-THE-ART CNN-BASED WAKE DETECTION 5613622
Fig. 17. Some examples of the detection results on real-shot yacht wake polarization images. These images were taken by a DoFP polarizing camera in a
bay area in Shenzhen, China, during April 2021. The label in the figure indicates the wake probability score output by WakeNet, and the score threshold of
the bounding boxes is 0.25. I is the traditional light intensity, Q and U are the Stokes linear polarization component images, and DoLP and AoP are the
```
DoLP and AoP images, respectively. (a) Images taken at an altitude of 50 m, incident angle is 81.47◦ , clear sky, and windspeed above the sea is 3.0 m/s.
```
```
(b) Images taken at an altitude of 50 m, incident angle is 81.47◦, clear sky, and windspeed above the sea is 2.8 m/s. (c) Images taken at an altitude of 300 m,
```
```
incident angle is 79.44◦, overcast, and windspeed above the sea is 3.8 m/s. (d) Images taken at an altitude of 50 m, incident angle is 86.04◦ , light rain, and
```
windspeed above the sea is 5.0 m/s.
is complex, so the RT-based method only detects the more
obvious turbulent wake and misses the Kelvin arms with
```
low SNR; WakeNet outputs an OBB with the correct width,
```
indicating that it accurately detected the Kelvin arms. The
```
wake signal in Fig. 16(c) is very weak, so the RT-based method
```
```
can only detect the obvious turbulent wake; the performance
```
```
of WakeNet is not affected. Fig. 16(d) shows the situation
```
when there is interference from the terrestrial background. Due
to the hull-centered subimage strategy, the RT-based method
is local, and the terrestrial background does not cause false
alarms, but the large-scale sea clutters do. WakeNet is not
affected, indicating that the generalization ability of handcraft
criteria is far inferior to that of learning-based methods.
```
Fig. 16(e) shows the situation where the two wakes interfere
```
with each other. Due to the obvious wake stripes, both
algorithms output good results. However, by observing the
image details, it can be found that the RT-based method is
disturbed by another wake when detecting the wake on the
upper part of the image, and the handcraft criterion cannot
```
distinguish the wake stripes of this nonown ship. Fig. 16(f)
```
shows the extreme case where the hull is invisible and the
wake SNR is low: the RT-based method fails to generate
subimages with enough pixels for the wake target on the left
edge of the image based on RT cannot generate subimages
with enough pixels for the target on the left edge of the image
and generates false alarms for the target on the right side of
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
5613622 IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 60, 2022
the image. The Kelvin arms predicted by WakeNet have some
errors with the ground truth, but the wake is basically located
accurately and no false alarm is generated. In summary,
traditional RT-based algorithm inevitably requires handcraft
postprocessing to suppress false alarms, and it is difficult to
fully consider various environmental factors affecting optical
wake detection. The RT-based algorithm with good precision
and recall in mild detection environment is difficult to cope
with the complex detection conditions in the SWIM dataset,
and the robustness is far inferior to the deep learning-based
algorithm.
E. Detection Results on the Real-Shot Polarization Images
To further demonstrate the feasibility of the CNN-based
method in the actual wake detection, we use WakeNet
```
(FcaNet-101 as the backbone) trained from the SWIM dataset
```
to detect the yacht wakes in the real-shot sea surface polariza-
tion images. Polarization imaging is an effective novel optical
wake imaging method, which can sensitively respond to the
roughness difference caused by wakes on the sea surface and
is less affected by environmental conditions. The synthesized
linear-polarized Q and U images have higher contrast of
wake objects than the light intensity I image, and the cal-
```
culated degree of linear polarization (DoLP) and angle of
```
```
polarization (AoP) images is also expected to achieve the 3-D
```
reconstruction of the sea surface. For more details about wake
polarization imaging, readers may refer to [11]. The experi-
mental images to be detected are monochromatic polarization
yacht wake images captured using an LUCID TRI050S-P
```
division of focal plane (DoFP) polarizing camera in a bay
```
area in Shenzhen, China, during April 2021, all with a size of
1024 × 1224 pixels. To facilitate inputting them into WakeNet
for detection, we cropped the center of the image with a size
of 768 × 768 pixels and filled the RGB channels with the
same gray value to meet the format requirements. Terrestrial
observation has a large incident angle, and the grayscale
distribution of the polarization image is different from that of
```
the intensity image (e.g., the turbulent wake exhibits as a bright
```
region in the intensity image, while it exhibits as a dark region
```
in the polarization image). In addition, the monochrome image
```
cannot provide the color features, and the performance of the
network trained by the light intensity image dataset SWIM is
thus obviously greatly limited. Even so, we still found that
WakeNet effectively extracted the wake texture features and
obtained a stable detection result, proving the feasibility in
practical applications. The typical detection results are shown
in Fig. 17.
```
Images in Fig. 17(a) are collected under ideal conditions
```
of a clear sky and a low sea state, so WakeNet correctly
detects the main wake objects in all component images and
gives generally high probability scores. Since the Kelvin wake
divergent wave textures in Q and U images are clearer than
that in I image, the network outputs wider bounding boxes
and more accurate Kelvin arm directions, indicating that for
an object detection task, the high-performance image detec-
tion algorithm and the effective imaging method complement
```
each other. Images in Fig. 17(b) are collected under similar
```
```
conditions as Fig. 17(a), and WakeNet also outputs accurate
```
detection results for distant wake objects. Similarly, due to the
larger range of the Kelvin arm features that can be observed
in the Q image, the network outputs a wider bounding box.
Besides, there is an incomplete wake object on the left edge
of the image. Due to the perspective distortion, its features are
greatly different from the training data, and the network does
```
not respond to it. Images in Fig. 17(c) are collected under
```
more common weather conditions and sea states. WakeNet
outputs the correct detection results in the I , Q, and DoLP
images. Because the U image and the AoP image are generally
darker and quite different from the light intensity image, the
network has an incorrect wake location in the U image and
does not respond to the object in the AoP image, which is
```
also reasonable. Images in Fig. 17(d) are collected under bad
```
weather conditions. The sea surface textures in all component
images are complex and the wake SNR is low. However,
WakeNet still outputs the correct detection results in the I , Q,
and DoLP images, indicating the robustness of the network
```
against harsh sea states. Similar to Fig. 17(c), the sea clutters
```
in the U and AoP images are so complex that even human
vision cannot recognize the wake objects in the images.
Correspondingly, WakeNet does not respond to the objects in
the two images nor does it output false alarms due to sea clut-
ters. In conclusion, it can be inferred from these experimental
results that WakeNet can obtain stable performance in various
optical remote sensing wake detection tasks after further fine-
tuning of the network through some specific imagery.
VI. C ONCLUSION
In this article, a CNN-based method for ship wake detection
in optical images is proposed. This method considers the
inherent differences between the optical images and SAR
images on the sea surface, processes the V-shaped Kelvin arms
and their internal transverse waves and divergent waves as an
object, and performs wake detection by the OBB. The cus-
tomized backbone and FPN are able to learn the wake spectral
characteristics and the spatial correlation between the wake
features at different scales. The additional landmark regression
subnetwork, including an RT layer, improves the hard wake
detection performance while predicting the wake heading.
Moreover, in order to fill the research gap, we collected a
large-scale optical wake detection dataset. Detection results on
the SWIM dataset show that our method is superior to other
```
CNN-based methods; the ablative study verifies the effective-
```
ness of the various modules in the proposed method, and
the detection results on real-shot polarization images further
demonstrate the practical value of the proposed method. The
research proves that the CNN-based optical wake detection
algorithm has the accuracy, robustness, and automation level
that the traditional RT-based method does not and provides
some useful enlightenment for the application of deep learning
in ship wake detection.
As future work, we would like to further optimize the
structure of the landmark regression subnetwork to perform
local RT according to the actual anchor position. We would
also like to design a special loss function for the Kelvin
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
XUE et al.: RETHINKING AUTOMATIC SHIP WAKE DETECTION: STATE-OF-THE-ART CNN-BASED WAKE DETECTION 5613622
arm regression to enhance the consistency among the various
subnetworks. Improving the algorithm’s speed is also an
important direction for further research. In addition, we may
try to propose a new evaluation metric that considers both
OBB regression and landmark regression to measure the wake
detection performance more comprehensively.
R EFERENCES
[1] Y. Liu and R. Deng, “Ship wakes in optical images,” J. Atmos. Ocean.
Technol., vol. 35, no. 8, pp. 1633–1648, Aug. 2018.
[2] J. F. Vesecky and R. H. Stewart, “The observation of ocean surface
phenomena using imagery from the SEASAT synthetic aperture radar:
An assessment,” J. Geophys. Res., vol. 87, no. C5, p. 3397, 1982.
[3] K. Hasselmann, “Theory of synthetic aperture radar ocean imaging:
A MARSEN view,” J. Geophys. Res., vol. 90, no. C3, pp. 4659–4686,
1985.
[4] J. K. E. Tunaley, E. H. Buller, K. H. Wu, and M. T. Rey, “The simulation
of the SAR image of a ship wake,” IEEE Trans. Geosci. Remote Sens.,
vol. 29, no. 1, pp. 149–156, Jan. 1991.
[5] K. Oumansour, Y. Wang, and J. Saillard, “Multifrequency SAR observa-
tion of a ship wake,” IEE Proc.-Radar, Sonar Navigat., vol. 143, no. 4,
pp. 275–280, Aug. 1996.
[6] I. Hennings, R. Romeiser, W. Alpers, and A. Viola, “Radar imaging
of Kelvin arms of ship wakes,” Int. J. Remote Sens., vol. 20, no. 13,
pp. 2519–2543, 1999.
[7] G. Zilman, A. Zapolski, and M. Marom, “On detectability of a ship’s
Kelvin wake in simulated SAR images of rough sea surface,” IEEE
Trans. Geosci. Remote Sens., vol. 53, no. 2, pp. 609–619, Jun. 2014.
[8] J.-K. Wang, M. Zhang, J.-L. Chen, and Z. Cai, “Application of facet
scattering model in SAR imaging of sea surface waves with Kelvin
wake,” Prog. Electromagn. Res. B, vol. 67, pp. 107–120, 2016.
[9] Y. Liu, R. Deng, and J. Zhao, “Simulation of Kelvin wakes in optical
images of rough sea surface,” Appl. Ocean Res., vol. 89, pp. 36–43,
Aug. 2019.
[10] M. Song, S. Wang, P. Zhao, Y. Chen, and J. Wang, “Modeling Kelvin
wake imaging mechanism of visible spectral remote sensing,” Appl.
Ocean Res., vol. 113, Aug. 2021, Art. no. 102712.
[11] F. Xue, W. Jin, S. Qiu, and J. Yang, “Airborne optical polarization
imaging for observation of submarine Kelvin wakes on the sea surface:
Imaging chain and simulation,” ISPRS J. Photogramm. Remote Sens.,
vol. 178, pp. 136–154, Aug. 2021.
[12] W. G. Pichel, P. Clemente-Colón, C. C. Wackerman, and K. S. Friedman,
“Ship and wake detection,” in Synthetic Aperture Radar Marine User’s
Manual, C. R. Jackson and J. R. Apel, Eds. Washington, DC, USA:
U.S. Dept. Commerce, NOAA/NESDIS, U.S. Govt. Print. Office, 2004,
pp. 277–303. [Online]. Available: http://www.sarusersmanual.com
[13] G. Watson, R. D. Chapman, and J. R. Apel, “Measurements of the
internal wave wake of a ship in a highly stratified sea loch,” J. Geophys.
Res., vol. 97, no. C6, pp. 9689–9703, 1992.
[14] G. Zilman, A. Zapolski, and M. Marom, “The speed and beam of a
ship from its wake’s SAR images,” IEEE Trans. Geosci. Remote Sens.,
vol. 42, no. 10, pp. 2335–2343, Oct. 2004.
[15] M. T. Rey, J. K. Tunaley, J. T. Folinsbee, P. A. Jahans, J. A. Dixon,
and M. R. Vant, “Application of radon transform techniques to wake
detection in seasat-A SAR images,” IEEE Trans. Geosci. Remote Sens.,
vol. 28, no. 4, pp. 553–560, Jul. 1990.
[16] A. C. Copeland, G. Ravichandran, and M. M. Trivedi, “Localized radon
transform-based detection of ship wakes in SAR images,” IEEE Trans.
Geosci. Remote Sens., vol. 33, no. 1, pp. 35–45, Jan. 1995.
[17] K. Eldhuset, “An automatic ship and ship wake detection system for
spaceborne SAR images in coastal regions,” IEEE Trans. Geosci. Remote
Sens., vol. 34, no. 4, pp. 1010–1019, Jul. 1996.
[18] J. M. Kuo and K. S. Chen, “The application of wavelets correlator for
ship wake detection in SAR images,” IEEE Trans. Geosci. Remote Sens.,
vol. 41, no. 6, pp. 1506–1511, Jun. 2003.
[19] F. Biondi, “Low-rank plus sparse decomposition and localized radon
transform for ship-wake detection in synthetic aperture radar images,”
IEEE Geosci. Remote Sens. Lett., vol. 15, no. 1, pp. 117–121, Jan. 2018.
[20] F. Biondi, “A polarimetric extension of low-rank plus sparse decom-
position and radon transform for ship wake detection in synthetic
aperture radar images,” IEEE Geosci. Remote Sens. Lett., vol. 16, no. 1,
pp. 75–79, Jan. 2019.
[21] Y. Liu, J. Zhao, and Y. Qin, “A novel technique for ship wake detection
from optical images,” Remote Sens. Environ., vol. 258, Jun. 2021,
Art. no. 112375.
[22] J. N. Newman, Marine Hydrodynamics. Cambridge, MA, USA: MIT
Press, 1977.
[23] M. Graziano, M. D’Errico, and G. Rufino, “Wake component detection
in X-band SAR images for ship heading and velocity estimation,”
Remote Sens., vol. 8, no. 6, p. 498, Jun. 2016.
[24] M. D. Graziano, M. Grasso, and M. D’Errico, “Performance analysis of
ship wake detection on sentinel-1 SAR images,” Remote Sens., vol. 9,
no. 11, p. 1107, 2017.
[25] G. Cheng, P. Zhou, and J. Han, “Learning rotation-invariant convo-
lutional neural networks for object detection in VHR optical remote
sensing images,” IEEE Trans. Geosci. Remote Sens., vol. 54, no. 12,
pp. 7405–7415, Dec. 2016.
[26] P. Ding, Y. Zhang, W.-J. Deng, P. Jia, and A. Kuijper, “A light and faster
regional convolutional neural network for object detection in optical
remote sensing images,” ISPRS J. Photogramm. Remote Sens., vol. 141,
pp. 208–218, Jul. 2018.
[27] L. Li, Z. Zhou, B. Wang, L. Miao, and H. Zong, “A novel CNN-based
method for accurate ship detection in HR optical remote sensing images
via rotated bounding box,” IEEE Trans. Geosci. Remote Sens., vol. 59,
no. 1, pp. 686–699, Jan. 2021, doi: 10.1109/TGRS.2020.2995477.
[28] C. Deng, M. Wang, L. Liu, Y. Liu, and Y. Jiang, “Extended feature
pyramid network for small object detection,” IEEE Trans. Multimedia,
early access, Apr. 20, 2021, doi: 10.1109/TMM.2021.3074273.
[29] X. Wang, R. Girshick, A. Gupta, and K. He, “Non-local neural net-
```
works,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR),
```
Jun. 2018, pp. 7794–7803.
[30] J. Ma et al., “Arbitrary-oriented scene text detection via rotation
proposals,” IEEE Trans. Multimedia, vol. 20, no. 11, pp. 3111–3122,
Nov. 2018.
[31] X. Yang et al., “SCRDet: Towards more robust detection for small,
cluttered and rotated objects,” in Proc. IEEE/CVF Int. Conf. Comput.
```
Vis. (ICCV), Oct. 2019, pp. 8231–8240.
```
[32] Y. Li, Y. Zhang, X. Huang, and A. L. Yuille, “Deep networks under
scene-level supervision for multi-class geospatial object detection from
remote sensing images,” ISPRS J. Photogramm. Remote Sens., vol. 146,
pp. 182–196, Dec. 2018.
[33] Y. Li, D. Kong, Y. Zhang, Y. Tan, and L. Chen, “Robust deep align-
ment network with remote sensing knowledge graph for zero-shot and
generalized zero-shot remote sensing image scene classification,” ISPRS
J. Photogramm. Remote Sens., vol. 179, pp. 145–158, Sep. 2021.
[34] Z. Qin, P. Zhang, F. Wu, and X. Li, “FcaNet: Frequency channel
attention networks,” 2020, arXiv:2012.11879.
[35] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,
“Feature pyramid networks for object detection,” in Proc. IEEE Conf.
```
Comput. Vis. Pattern Recognit. (CVPR), Jul. 2017, pp. 936–944.
```
[36] A. Arnold-Bos, A. Martin, and A. Khenchaf, “Obtaining a ship’s speed
and direction from its Kelvin wake spectrum using stochastic matched
filtering,” in Proc. IEEE Int. Geosci. Remote Sens. Symp., Jul. 2007,
pp. 1106–1109.
[37] Y.-H. Zhao, X. Han, and P. Liu, “A RPCA and RANSAC based algorithm
for ship wake detection in SAR images,” in Proc. 12th Int. Symp.
```
Antennas, Propag. EM Theory (ISAPE), Dec. 2018, pp. 1–4.
```
[38] G. Yang, J. Yu, C. Xiao, and W. Sun, “Ship wake detection for SAR
images with complex backgrounds based on morphological dictionary
learning,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process.
```
(ICASSP), Mar. 2016, pp. 1896–1900.
```
[39] J. Nan, C. Wang, B. Zhang, F. Wu, H. Zhang, and Y. Tang, “Ship wake
CFAR detection algorithm in SAR images based on length normalized
```
scan,” in Proc. IEEE Int. Geosci. Remote Sens. Symp. (IGARSS),
```
Jul. 2013, pp. 3562–3565.
[40] B. Tings and D. Velotto, “Ship wake detectability and classification
on TerraSAR-X high resolution data,” in Proc. 12th Eur. Conf. Synth.
```
Aperture Radar (EUSAR), Jun. 2018, pp. 1307–1310.
```
[41] Y. Wei, Z. Wu, H. Li, J. Wu, and T. Qu, “Application of periodic structure
scattering in Kelvin ship wakes detection,” Sustain. Cities Soc., vol. 47,
May 2019, Art. no. 101463.
[42] J. P. Fitch, S. K. Lehman, F. U. Dowla, S. Y. Lu, E. M. Johansson,
and D. M. Goodman, “Ship wake-detection procedure using conjugate
gradient trained artificial neural networks,” IEEE Trans. Geosci. Remote
Sens., vol. 29, no. 5, pp. 718–726, Sep. 1991.
[43] K.-M. Kang and D.-J. Kim, “Ship velocity estimation from ship wakes
detected using convolutional neural networks,” IEEE J. Sel. Topics Appl.
Earth Observ. Remote Sens., vol. 12, no. 11, pp. 4379–4388, Nov. 2019.
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.
5613622 IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 60, 2022
[44] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature
hierarchies for accurate object detection and semantic segmentation,”
2013, arXiv:1311.2524.
[45] R. Girshick, “Fast R-CNN,” in Proc. IEEE Int. Conf. Comput. Vis.
```
(ICCV), Dec. 2015, pp. 1440–1448.
```
[46] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards
real-time object detection with region proposal networks,” IEEE
Trans. Pattern Anal. Mach. Intell., vol. 39, no. 6, pp. 1137–1149,
Jun. 2017.
[47] W. Liu et al., “SSD: Single shot multibox detector,” in Proc. Eur. Conf.
```
Comput. Vis. (ECCV), 2016, pp. 21–37.
```
[48] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look
```
once: Unified, real-time object detection,” in Proc. IEEE Conf. Comput.
```
```
Vis. Pattern Recognit. (CVPR), Jun. 2016, pp. 779–788.
```
[49] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for
dense object detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 42,
no. 2, pp. 318–327, Feb. 2020.
[50] Z. Tian, C. Shen, H. Chen, and T. He, “FCOS: Fully convolutional
one-stage object detection,” in Proc. IEEE/CVF Int. Conf. Comput. Vis.
```
(ICCV), Oct. 2019, pp. 9626–9635.
```
[51] X. Zhou, D. Wang, and P. Krähenbühl, “Objects as points,” 2019,
```
arXiv:1904.07850.
```
[52] J. Pang, C. Li, J. Shi, Z. Xu, and H. Feng, “R2-CNN: Fast tiny object
detection in large-scale remote sensing images,” IEEE Trans. Geosci.
Remote Sens., vol. 57, no. 8, pp. 5512–5524, Aug. 2019.
[53] G. Zhang, S. Lu, and W. Zhang, “CAD-Net: A context-aware detection
network for objects in remote sensing imagery,” IEEE Trans. Geosci.
Remote Sens., vol. 57, no. 12, pp. 10015–10024, Aug. 2019.
[54] X. Yang, J. Yan, X. Yang, J. Tang, W. Liao, and T. He, “SCRDet++:
Detecting small, cluttered and rotated objects via instance-level feature
denoising and rotation loss smoothing,” 2020, arXiv:2004.13316.
[55] X. Yang, J. Yan, and T. He, “On the arbitrary-oriented object detection:
Classification based approaches revisited,” 2020, arXiv:2003.05597.
[56] Z. Chen et al., “PIoU loss: Towards accurate oriented object detection
```
in complex environments,” in Proc. Eur. Conf. Comput. Vis. (ECCV),
```
2020, pp. 195–211.
[57] X. Yang, J. Yan, Q. Ming, W. Wang, X. Zhang, and Q. Tian, “Rethinking
rotated object detection with Gaussian wasserstein distance loss,” 2021,
```
arXiv:2101.11952.
```
[58] T. Wahl, “Ship traffic monitoring using ERS-1 SAR,” in Proc. 1st ERS-
1 Symp., Cannes, France, Nov. 1992, pp. 823–828.
[59] S. Razakarivony and F. Jurie, “Vehicle detection in aerial imagery:
A small target detection benchmark,” J. Vis. Commun. Image Represent.,
vol. 34, pp. 187–203, Jan. 2016.
[60] H. Zhu, X. Chen, W. Dai, K. Fu, Q. Ye, and J. Jiao, “Orientation
robust object detection in aerial images using deep convolutional neural
```
network,” in Proc. IEEE Int. Conf. Image Process. (ICIP), Sep. 2015,
```
pp. 3735–3739.
[61] T. N. Mundhenk, G. Konjevod, W. A. Sakla, and K. Boakye, “A large
contextual dataset for classification, detection and counting of cars
```
with deep learning,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2016,
```
pp. 785–800.
[62] Z. Liu, H. Wang, H. Weng, and L. Yang, “Ship rotated bounding box
space for ship extraction from high-resolution optical satellite images
with complex backgrounds,” IEEE Geosci. Remote Sens. Lett., vol. 13,
no. 8, pp. 1074–1078, Aug. 2016.
[63] G.-S. Xia et al., “DOTA: A large-scale dataset for object detection in
aerial images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.,
Jun. 2018, pp. 3974–3983.
[64] K. Li, G. Wan, G. Cheng, L. Meng, and J. Han, “Object detection in
optical remote sensing images: A survey and a new benchmark,” ISPRS
J. Photogramm. Remote Sens., vol. 159, pp. 296–307, Jan. 2020.
[65] A. J. Gallego, A. Pertusa, and P. Gil, “Automatic ship classification from
optical aerial images with convolutional neural networks,” Remote Sens.,
vol. 10, no. 4, p. 511, 2018.
[66] D. Lam et al., “XView: Objects in context in overhead imagery,” 2018,
```
arXiv:1802.07856.
```
[67] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and
A. Zisserman, “The PASCAL visual object classes (VOC) challenge,”
Int. J. Comput. Vis., vol. 88, no. 2, pp. 303–338, Jun. 2010.
[68] Y.-X. Sun, P. Liu, and Y.-Q. Jin, “Ship wake components: Isola-
tion, reconstruction, and characteristics analysis in spectral, spatial,
and TerraSAR-X image domains,” IEEE Trans. Geosci. Remote Sens.,
vol. 56, no. 7, pp. 4209–4224, Jul. 2018.
[69] J. Fu, J. Liu, J. Jiang, Y. Li, Y. Bao, and H. Lu, “Scene segmentation
with dual relation-aware attention network,” IEEE Trans. Neural Netw.
Learn. Syst., vol. 32, no. 6, pp. 2547–2560, Jun. 2021.
[70] W. Qian, X. Yang, S. Peng, Y. Guo, and J. Yan, “Learning modulated
loss for rotated object detection,” 2019, arXiv:1911.08299.
[71] K. Zhao, Q. Han, C.-B. Zhang, J. Xu, and M.-M. Cheng, “Deep Hough
transform for semantic line detection,” IEEE Trans. Pattern Anal. Mach.
Intell., early access, May 3, 2021, doi: 10.1109/TPAMI.2021.3077129.
[72] J. Pang, K. Chen, J. Shi, H. Feng, W. Ouyang, and D. Lin, “Libra
R-CNN: Towards balanced learning for object detection,” in Proc.
```
IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2019,
```
pp. 821–830.
[73] Y. Jiang et al., “R2CNN: Rotational region CNN for orientation robust
scene text detection,” 2017, arXiv:1706.09579.
[74] J. Yi, P. Wu, B. Liu, Q. Huang, H. Qu, and D. Metaxas, “Oriented
object detection in aerial images with box boundary-aware vectors,”
```
in Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV), Jan. 2021,
```
pp. 2149–2158.
```
Fuduo Xue (Graduate Student Member, IEEE)
```
received the B.S. degree in optoelectronic informa-
tion engineering from Beijing Jiaotong University,
Beijing, China, in 2018. He is currently pursuing the
Ph.D. degree in optical engineering with the School
of Optoelectronics, Beijing Institute of Technology,
Beijing.
His research interests include optical imaging tech-
niques, maritime remote sensing, and underwater
photogrammetry.
Weiqi Jin received the Ph.D. degree in military
optics from the Beijing Institute of Technology,
Beijing, China, in 1990.
He has been a Ph.D. Tutor, a Professor, and the
Director of the Key Laboratory of Opto-Electronic
Imaging Technology and Systems, Beijing Institute
of Technology. He is also the Director of the Chinese
Optical Society and the Beijing Institute of Optics.
His main research interests include polarization
imaging technology, image and video processing,
and photoelectric detection technology.
Dr. Jin is an Advanced Member of the Chinese Institute of Electronics.
Su Qiu received the B.S. and Ph.D. degrees in
optical engineering from the Beijing Institute of
Technology, Beijing, China, in 2003 and 2013,
respectively.
He has been a Master’s Supervisor and a Lecturer
with the Key Laboratory of Opto-Electronic Imaging
Technology and Systems, Beijing Institute of Tech-
nology. His main research interests include image
information processing, photoelectric detection, and
photoelectric imaging technology.
```
Jie Yang (Graduate Student Member, IEEE)
```
received the B.S. degree in optical information and
technology from the Beijing Institute of Technol-
ogy, Beijing, China, in 2017, where she is cur-
rently pursuing the Ph.D. degree with the School
of Optoelectronics.
Her research interests are polarization imaging
technology and its application.
Authorized licensed use limited to: University of Science & Technology of China. Downloaded on January 04,2026 at 08:21:29 UTC from IEEE Xplore. Restrictions apply.