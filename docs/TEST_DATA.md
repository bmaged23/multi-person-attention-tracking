# Test data provenance

Source: TV Human Interaction Dataset, Oxford Visual Geometry Group.
https://www.robots.ox.ac.uk/~alonso/tv_human_interactions.html

Original download:
https://www.robots.ox.ac.uk/~alonso/data/tv_human_interactions_videos.tar.gz

Original documentation is retained locally in input/tvhi/SOURCE_README.txt.
The 300 clips show interactions and people in varied poses. The source provides
upper-body, discrete head-orientation, and interaction annotations; they are not
attention labels. No attention accuracy is reported from these examples.

The dataset is available for research purposes only. Its author states that
video copyrights remain with their rightful owners. Source footage and generated
videos stay under ignored input/output directories. Do not treat this dataset as
freely licensed portfolio/social-media footage.

Selected full processing examples:
- negative_0021.avi: 118 frames; frontal and upward-looking faces.
- handShake_0036.avi: 291 frames; two interacting people with changing head poses.

Outputs are under output/tests/. Person names are dummy labels. Track counts
can exceed human counts because these clips include lost faces and changes
of viewpoint; spatial association is not face recognition.
