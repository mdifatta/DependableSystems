# DependableSystems
Code repo for the dependable systems class project.

This repo contains:
- the llfi-script.py script which is can be used to run all the four steps of LLFI fault injection (see below for more details)
- a one-file modified version of ezSIFT, a standalone implementation of SIFT keypoints detector, in the fi_sample folder
- a sample input.yaml in the fi_sample folder
- a sample input image in the format P5 .pgm in the images folder

In order to make our code work properly follow this folders structure:
<pre>
~/
| -- Desktop/
|     | -- images/
|     |     | -- img1.pgm
|     |     
|     | -- fi_sample/
|     |     | -- feature_extract.cpp
|     |     | -- input.yaml
|     |  
|     | -- llfi-script.py
</pre>
Notes:
1. call all the project folders following the pattern 'fi_xxxx', the Python script will search for this pattern name.
2. the input images must being in the P5 .pgm format
3. the input.yaml file is just a sample: add, remove and edit the section based on you needs
4. the Python script can be run with two command line arguments:
    * -c to run only the 'compile' and 'instrument' steps of the pipeline
    * -p to run only the 'profile' and 'inject' steps of the pipeline on ALL the images in the images/ folder
    CAVEAT: with no args the script won't do anything, with both args the script will run the entire pipeline

Check out https://github.com/DependableSystemsLab/LLFI for LLFI's repo
and https://github.com/robertwgh/ezSIFT for eszSIFT's repo.
