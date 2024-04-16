#Copyright: Svetlin Tassev.
#Released under GPLv3 license.

import os
import numpy as np
import matplotlib.pyplot as plt

import cv2


import exifread
raw_dir='/home/user/Pictures/Eclipse2024/'
# List of raw image files to read exif data.
#The corresponding cropped image files will be used in the final merge.

image_files = [
    'DSC02729.ARW',
    'DSC02739.ARW',
    'DSC02744.ARW'
]

# Extract ISO and Exposure Time, and calculate Unnormalized Light Amount in each image.
unnormalized_light = {}
exptime={} # Will contain the exposure times
for image_file in image_files:
    with open(raw_dir+image_file, 'rb') as f:
        tags = exifread.process_file(f)
        iso_tag = tags.get('EXIF ISOSpeedRatings')
        exposure_time_tag = tags.get('EXIF ExposureTime')
        if iso_tag and exposure_time_tag:
            iso = int(iso_tag.values[0])
            exposure_time_str = str(exposure_time_tag.values[0])
            if '/' in exposure_time_str:
                num, den = exposure_time_str.split('/')
                exposure_time = float(num) / float(den)
            else:
                exposure_time = float(exposure_time_str)
            unnormalized_light[image_file.split('.')[0]] = iso * exposure_time
            exptime[image_file.split('.')[0]] =  exposure_time
        else:
            unnormalized_light[image_file.split('.')[0]] = None


#####################################################################
# Find the angular averaged intensity as a function of radius from
# the center of the image. That will be used to calibrate
# the image brightness between images.

# Create a dictionary to store the I2 1D array for each base_filename
Idic = {}
plt.figure(figsize=(12, 8))
lines = []
labels = []

#plt.figure()
for filename in os.listdir(dir):
    if filename.endswith('.exr') and filename.startswith('cropped_'):
        base_filename = filename[len('cropped_'):-len('.exr')]
        # Load the image as a 32-bit depth NumPy array
        img = cv2.imread(os.path.join(dir, filename), flags=(cv2.IMREAD_COLOR|cv2.IMREAD_ANYDEPTH))
        img = img.astype(np.float32)

        # Calculate the center of the image
        center_x = img.shape[1] // 2
        center_y = img.shape[0] // 2

        # Calculate the distance of each pixel from the center
        x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Calculate the brightness of each pixel
        brightness=0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]

        # Create a 1D array I1 of size N=1000, where each element corresponds to a radius interval
        N = 1000
        Rmax = np.sqrt(center_x**2+ center_y**2)
        I1 = np.zeros(N)
        count = np.zeros(N)
        rr=np.arange(N)*Rmax/(N-1)

        # Iterate through each pixel in the image using NumPy slicing
        idx = np.round(r * (N-1) / Rmax).astype(int)
        I1_inc = np.bincount(idx.ravel(), weights=brightness.ravel(), minlength=N)
        count_inc = np.bincount(idx.ravel(), minlength=N)
        I1 += I1_inc
        count += count_inc

        # Calculate the average intensity as a function of radius
        I2 = I1 / count/unnormalized_light[base_filename]
        Idic[base_filename] = I2
        # Plot the I2 array
        

# The code below calculates the angular averaged I3=intensity(r) after correcting for
# under/over exposure between images (II3 is the upscaled in resolution version of I3). 
# This part of the code ends at the line saying # DONE!
# We will use this intensity profile to match the images in intensity and flatten the image
# so that the inner and outer portions of the corona are more or less equally bright.

# None of what follows is rigorous (one should be doing flats, darks,
# response function of the detector, etc) but rather ad hoc for my camera. You should
# explore what the best way to get I3 is for your detector.

# Scale the brightness and unnormalized_light, which is just indicative, and not 
# necessarily perfectly precise by matching all intensity curves (for all images) at rr[235], where they 
# all seem to match to within a few precent
# for my detector.
# Change according to your circumstances.
for key in Idic:
    unnormalized_light[key] *= Idic[key][235*N//1000]
    Idic[key] /= Idic[key][235*N//1000]
    
    line, = plt.loglog(rr, Idic[key], label=key)
    lines.append(line)
    labels.append(key)

# Create a new 1D array I3 that will contain intensity(r)
I3 = np.zeros_like(Idic[list(Idic.keys())[0]],dtype=np.float64)
k=0

# Get the disk of the moon data from the longest exposure image. The 140 corresponds to rr[140]=
#edge of disk.
I3[:140*N//1000]=Idic['DSC02744'][:140*N//1000]
for key in Idic:
	#Inside rr[235], use the least over-exposed image (hopefully, it is not overexposed).
    I3[140*N//1000:235*N//1000] = np.maximum(I3[140*N//1000:235*N//1000], Idic[key][140*N//1000:235*N//1000])
    #Outside of rr[235] use the longes exposure image.
    x=np.where(Idic[key][235*N//1000:]<1.e-12)
    Idic[key][235*N//1000:][x]=1e12
    if (k==0):
        I3[235*N//1000:] = Idic[key][235*N//1000:]
    else:
        I3[235*N//1000:] = np.minimum(I3[235*N//1000:], Idic[key][235*N//1000:])
    k+=1

from scipy.optimize import curve_fit

#Smooth I3 beyond rr[170] by fitting a polynomial to log(I)vs log(r).

index = np.arange(len(I3))[170*N//1000:]
log_index = np.log(index)
log_I3 = np.log(I3[170*N//1000:])

# Define the polynomial function
def poly_func(x, a, b, c,d,e,f):
    return a * x**2 + b * x + c+d*x**3+e*x**4+f*x**5

# Perform the non-linear fit
popt, pcov = curve_fit(poly_func, log_index, log_I3, p0=[1, 1, 1,1,1,1])

# Print the fit parameters
print("Fit parameters:")
for i, param in enumerate(popt):
    print(f"Parameter {i+1}: {param:.4f}")


# Replace I3[160:] with the fitted values
II3=I3.copy()
II3[170*N//1000:] = np.exp(poly_func(log_index, *popt))


from scipy.signal import savgol_filter
# Smooth the I3[160:] portion using a Savitzky-Golay filter
window_length = 21  # Adjust this parameter to control the smoothing level
polyorder = 3  # Degree of the smoothing polynomial
II3[150*N//1000:] = savgol_filter(II3[150*N//1000:], window_length, polyorder)


from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Now interpolate II3 to 20 times better resolution.
index = np.arange(len(II3))
log_index = np.log(index + 1)
log_I3 = np.log(II3)

# Perform log-log cubic interpolation
f = interp1d(log_index, log_I3, kind='cubic')
N0=N
N=len(I3) * 20
# Create a new index array with 20 times the resolution
new_index = np.linspace(np.exp(log_index[0]), np.exp(log_index[-1]), N)

# Evaluate the interpolated values at the new index
new_log_I3 = f(np.log(new_index))

# Create the I4 array
II3 = np.exp(new_log_I3)
# Plot the I3 array
#plt.figure(figsize=(12, 8))
rr=np.arange(N)*Rmax/(N-1)
plt.loglog(rr, II3, label='I3')
plt.title("I3 for all images")
plt.xlabel("Radius Interval")
plt.ylabel("Average Brightness")
plt.legend(loc='upper left')
plt.savefig('plot.pdf')
#plt.show()

# DONE!
######################################################
#These weights are used to weigh the corresponding pixels of 
# different images. Pixels that are over/underexposed will have a weight
# that's tiny because of the sigma of 0.1*I3 that's chosen.
weights = {}
for key in Idic:
    weights[key] = np.exp(-((I3 - Idic[key])**2) / (2 * (0.1*I3)**2))
    weights[key]*=exptime[key]
# Create the weighted averaged image
weighted_img = np.zeros_like(img,dtype=np.float32)
total_weights = np.zeros_like(img[:,:,0],dtype=np.float32)[:,:,None]
k=0
for filename in os.listdir(dir):
    if filename.endswith('.exr') and filename.startswith('cropped_'):
        k+=1
        base_filename = filename[len('cropped_'):-len('.exr')]
        # Load the image as a 32-bit depth NumPy array
        img = cv2.imread(os.path.join(dir, filename), flags=(cv2.IMREAD_COLOR|cv2.IMREAD_ANYDEPTH))
        img = img.astype(np.float32)

        # Calculate the center of the image
        center_x = img.shape[1] // 2
        center_y = img.shape[0] // 2

        # Calculate the distance of each pixel from the center
        x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Calculate the index of each pixel in the 1D array
        idx = np.round(r * (N-1) / Rmax).astype(int)
        z=np.where(np.sum(img,axis=2) >1.e-30)
        # Apply the weights to the image
        I=(img /unnormalized_light[base_filename])
        I/= II3[idx][:,:,None]**(0.9)
        #
        # Save the flattened in intensity image.
        cv2.imwrite(dir+f'flat_cropped_{os.path.splitext(filename)[0]}.exr', (I/np.max(I)).astype(np.float32))
        idx = np.round(r * (N0-1) / Rmax).astype(int)
        
        
        W=((weights[base_filename][idx]))
        blurred_W = cv2.GaussianBlur(W, (33, 33), sigmaX=4, sigmaY=4)
        if base_filename!='DSC02729':
			#in the inner portions of the corona, use only the least exposed image, which has 
			# good detail of the prominences.
            blurred_W*=np.pi/2+np.arctan((r-1000)/10)
            
        I*= blurred_W[:,:,None]
        weighted_img[:,:,0][z] += I[:,:,0][z]
        weighted_img[:,:,1][z] += I[:,:,1][z]
        weighted_img[:,:,2][z] += I[:,:,2][z]
        total_weights[:,:,0][z] += blurred_W[z]
            
        

# Normalize the weighted averaged image
weighted_img /= total_weights
weighted_img/=np.max(weighted_img)
cv2.imwrite('FINAL.exr', weighted_img, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

