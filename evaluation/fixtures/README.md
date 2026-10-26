# Reproducible AMI API fixtures

Two **2-second, 50-frame** excerpts from AMI meeting `IS1004d`, left and right
camera views starting at 60 seconds. AMI Meeting Corpus, University of Edinburgh
and consortium contributors: https://groups.inf.ed.ac.uk/ami/corpus/ .
Released under **CC BY 4.0**, with license text in
[docs/licenses/AMI_CC_BY_4.0.txt](../../docs/licenses/AMI_CC_BY_4.0.txt).

Modifications: selected the first 50 decoded frames from each previously downloaded
12-second clip, removed audio, and re-encoded losslessly as FFV1/Matroska. Every
BGR pixel was compared with the source decoder output and matched. No attention
labels were supplied by AMI; attention outputs are heuristic regression references.

The release policy records fixture hashes and original 12-second clip hashes.
These small raw inputs are redistributed here for repeatable API checks. They are
excluded from the inference image. COCO test images remain separately provisioned.
