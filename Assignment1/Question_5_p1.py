import numpy
import scipy.io.wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import dct

#INPUT THE AUDIO FILE
sample_rate, signal = scipy.io.wavfile.read('/home/rohitk/Desktop/MLSP/a1/speechFiles/clean.wav')
##PRE-EMPHASIS OF AUDIO FILE
pre_emphasis = 0.97
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
####

frame_size = 0.025
frame_stride = 0.01

frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_length))
pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(numpy.int32,copy=False)]

#####DOING THE WINDOW

frames *= numpy.hamming(frame_length)
#####DOING THE STFT OF THE SIGNAL
NFFT = 511
mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

mm=numpy.max(numpy.max(pow_frames))

#print(numpy.shape(pow_frames))

time=numpy.arange(pow_frames.shape[0])
#print(np.shape(time))
freq=numpy.arange(pow_frames.shape[1])


plt.pcolormesh(time, freq, pow_frames.T, vmin=0,vmax=mm)
plt.imshow(pow_frames.T)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#plt.show()


#####################################################################################
###Doing the log thing

numpy.seterr(divide = 'ignore')

stft_log=numpy.log(pow_frames)

numpy.seterr(divide = 'warn')
##################################################################################
#######################################
num_ceps = 128
###type represent the type of dct we are doing,there are three main type of dct that availabe
##axis along which we do dct,here we will do column wise ie along freq domain hence axis=1

mfcc = dct(pow_frames, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
print(numpy.shape(mfcc))

plt.pcolormesh(numpy.arange(310), numpy.arange(128), mfcc.T)
plt.imshow(mfcc.T)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#plt.show()
##################################################################################

mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

mfcc_c = mfcc.T

#print(numpy.shape(mfcc))
#############################################

#S=numpy.matmul(mfcc,mfcc.T)
S_c=numpy.cov(mfcc_c)
#print(numpy.shape(S))

L , U = numpy.linalg.eigh(S_c)

#print(numpy.shape(L))
#print(numpy.shape(U))

L=numpy.diag(L)
#print(numpy.shape(L))
L_s=numpy.sqrt(L)
LL=numpy.linalg.inv(L_s)
#######################################################

temp_c=numpy.matmul(LL,U.T)
Y_c=numpy.matmul(temp_c,mfcc_c)
#print(numpy.shape(Y))

kk_c=numpy.cov(Y_c)
##print(kk_c)

error_cc=numpy.sum(numpy.abs(kk_c)) - numpy.trace(numpy.abs(kk_c))
print('This is the sum of absolute value of off diag elemenets when clean signal is there and its corresponsing whitening factor')
print(error_cc/16256)
################################################################################


#INPUT THE AUDIO FILE
sample_rate, signal = scipy.io.wavfile.read('/home/rohitk/Desktop/MLSP/a1/speechFiles/noisy.wav')
##PRE-EMPHASIS OF AUDIO FILE
pre_emphasis = 0.97
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
####

frame_size = 0.025
frame_stride = 0.01

frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_length))
pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(numpy.int32,copy=False)]

#####DOING THE WINDOW

frames *= numpy.hamming(frame_length)
#####DOING THE STFT OF THE SIGNAL
NFFT = 511
mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

mm=numpy.max(numpy.max(pow_frames))

#print(numpy.shape(pow_frames))

time=numpy.arange(pow_frames.shape[0])
#print(np.shape(time))
freq=numpy.arange(pow_frames.shape[1])


plt.pcolormesh(time, freq, pow_frames.T, vmin=0,vmax=mm)
plt.imshow(pow_frames.T)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#plt.show()


#####################################################################################
###Doing the log thing

numpy.seterr(divide = 'ignore')

stft_log=numpy.log(pow_frames)

numpy.seterr(divide = 'warn')
##################################################################################
#######################################
num_ceps = 128
###type represent the type of dct we are doing,there are three main type of dct that availabe
##axis along which we do dct,here we will do column wise ie along freq domain hence axis=1

mfcc = dct(pow_frames, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
print(numpy.shape(mfcc))

plt.pcolormesh(numpy.arange(310), numpy.arange(128), mfcc.T)
plt.imshow(mfcc.T)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#plt.show()
##################################################################################

mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

mfcc_n = mfcc.T

#print(numpy.shape(mfcc))
#############################################

#S=numpy.matmul(mfcc,mfcc.T)
S_n=numpy.cov(mfcc_n)
#print(numpy.shape(S))

L , U = numpy.linalg.eigh(S_n)

#print(numpy.shape(L))
#print(numpy.shape(U))

L=numpy.diag(L)
#print(numpy.shape(L))
L_s=numpy.sqrt(L)
LL=numpy.linalg.inv(L_s)
#######################################################

temp_n=numpy.matmul(LL,U.T)
Y_n=numpy.matmul(temp_c,mfcc_n)
#print(numpy.shape(Y))

kk_n=numpy.cov(Y_n)
print(numpy.shape(kk_c))

error_nn=numpy.sum(numpy.abs(kk_n)) - numpy.trace(numpy.abs(kk_n))
print('This is the sum of absolute value of off diag elemenets when clean signal is there and its corresponsing whitening factor')
print(error_nn/16256)
################################################################################



