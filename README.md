
These are some notes and code for merging my raw photos of the Solar Eclipse 2024.

This is the resulting merged photo:
![Eclipse2024](https://github.com/stassev/Solar_Eclipse_2024/assets/6117115/2851e296-b187-4353-9b3a-9ec1215fad4c)

Here is an example of a jpeg directly from the camera:
![DSC02739](https://github.com/stassev/Solar_Eclipse_2024/assets/6117115/4c75afa9-ed8c-4351-a854-7d4245976ab8)


Steps (2-6 use Python/OpenCV):
1. Use Darktable to export raw images to 32-bit exr in linear light. The profile that was used is in the .xmp file.
2. Use Hough transform to find eclipsed Sun disk. Recenter and crop images.
3. Find the angular average of the intensity profile for each image. From the profiles, it is clear which radial portions of an image are under/over-exposed.
4. Use some ad hoc method to determine actual intensity profile of the corona. Didn't have darks/flats, so couldn't really do better than this.
5. Compare actual intensity to intensity in each image. Use that to weigh away under/over exposed portions of images.
6. Merge images and flatten radial intensity by dividing by actual intensity profile.
7. Use GIMP to do final touches.
