![GitHub](https://img.shields.io/github/license/hunar4321/life_code)

# Weeb-Detector
An AI taught to recognize weebs in discord profile pictures. It takes an image and then outputs how certain it is that the image contained anime. I personally collected and labeled close to 1000 discord profile pictures from random servers to use as training material. Unfortunately due to github restrictions it had to be uploaded in a .zip file in the Dataset directory. The checkpoint of the model couldn't be included for the same reason. The AI trained from the database might have weird behavior on non-square images or images with high resolution and detail

Examples:
---------------------------------
Weeb:

![test_weeb](https://user-images.githubusercontent.com/96934612/200219363-665f4541-5ca7-4f87-9cf4-51a34afc93af.png)

![2022-11-06_19-01_1](https://user-images.githubusercontent.com/96934612/200219421-56276b0d-3e14-435a-b2f2-d3a5451ff5db.jpg)

Normal:

![test_normal](https://user-images.githubusercontent.com/96934612/200219383-ae7486b9-4727-4e09-9a0d-de0bc2242fe5.png)

![2022-11-06_19-01](https://user-images.githubusercontent.com/96934612/200219424-275dbf65-0712-4124-ad2d-329664ec302b.jpg)

How to use:
---------------------------------
1. Unzip the database
2. Run the DetectorTrainer.py script to train a model(It shouldn't take longer than half an hour on a potato PC)
3. Run TestImage.py (Replace the image path with a path to your own image)

Do what you want with the code, but credit would be much appreciated. Contact info in the profile
