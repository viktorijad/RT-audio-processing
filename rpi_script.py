import pyaudio
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as animation
import matplotlib.style as ms
from matplotlib import style
import scipy.signal as signal
import itertools
from __future__ import print_function
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import butter, lfilter, freqz
import statistics as stat
import IPython.display
import librosa
import librosa.display
import parselmouth
import seaborn
import time
from neopixel import *
import argparse
import RPi.GPIO as GPIO
seaborn.set()

ms.use('seaborn-muted')

#style.use('fivethirtyeight')

# LED strip configuration:
LED_COUNT      = 23      # Number of LED pixels.
LED_PIN        = 18      # GPIO pin connected to the pixels (18 uses PWM!).
#LED_PIN        = 10      # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ    = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 10      # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 150     # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False   # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53

COL_RED         = Color(0, 255, 0)
COL_GREEN       = Color(255, 0, 0)
COL_YELLOW      = Color(255, 255, 0)

# Define in and output pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(25, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Define functions which animate LEDs in various ways.
# def colorWipe(strip, color, wait_ms=50):
    # """Wipe color across display a pixel at a time."""
    # for i in range(strip.numPixels()):
        # strip.setPixelColor(i, color)
        # strip.show()
        # time.sleep(wait_ms/1000.0)

# def theaterChase(strip, color, wait_ms=50, iterations=10):
    # """Movie theater light style chaser animation."""
    # for j in range(iterations):
        # for q in range(3):
            # for i in range(0, strip.numPixels(), 3):
                # strip.setPixelColor(i+q, color)
            # strip.show()
            # time.sleep(wait_ms/1000.0)
            # for i in range(0, strip.numPixels(), 3):
                # strip.setPixelColor(i+q, 0)

def wheel(pos):
    """Generate rainbow colors across 0-255 positions."""
    if pos < 85:
        return Color(pos * 3, 255 - pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return Color(255 - pos * 3, 0, pos * 3)
    else:
        pos -= 170
        return Color(0, pos * 3, 255 - pos * 3)

# def rainbow(strip, wait_ms=20, iterations=1):
    # """Draw rainbow that fades across all pixels at once."""
    # for j in range(256*iterations):
        # for i in range(strip.numPixels()):
            # strip.setPixelColor(i, wheel((i+j) & 255))
        # strip.show()
        # time.sleep(wait_ms/1000.0)

# def rainbowCycle(strip, wait_ms=20, iterations=5):
    # """Draw rainbow that uniformly distributes itself across all pixels."""
    # for j in range(256*iterations):
        # for i in range(strip.numPixels()):
            # strip.setPixelColor(i, wheel((int(i * 256 / strip.numPixels()) + j) & 255))
        # strip.show()
        # time.sleep(wait_ms/1000.0)

# def theaterChaseRainbow(strip, wait_ms=50):
    # """Rainbow movie theater light style chaser animation."""
    # for j in range(256):
        # for q in range(3):
            # for i in range(0, strip.numPixels(), 3):
                # strip.setPixelColor(i+q, wheel((i+j) % 255))
            # strip.show()
            # time.sleep(wait_ms/1000.0)
            # for i in range(0, strip.numPixels(), 3):
                # strip.setPixelColor(i+q, 0)

def simpleColor(strip, color):
    """ Just show a single color."""
    strip.setPixelColor(12, color)
    strip.show()
    # time.sleep(0.5)
    # strip.setPixelColor(12, Color(0,0,0))
    # strip.show()
    # time.sleep(0.5)

CHANNELS = 1
SAMPLING_RATE = 22050 # Viktorija changed from 44100

p = pyaudio.PyAudio()
fulldata = np.array([])
dry_data = np.array([])

buffer = np.zeros(1024)
audio_data = list()

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlim(0,1024)
ax1.set_yticks( [-100, -50, 0, 50, 100],minor=False )
ax1.set_title("Raw Audio Signal")

###############################################################
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def averageFilter(data,widthSegments):
    length = len(data)
    pitchAverageFilter = []
    for i in range(widthSegments,length-widthSegments):
        sumPitch = 0
        for k in range(i-widthSegments,i+widthSegments):
            sumPitch = sumPitch + data[k]
        sumPitch = sumPitch/(2*widthSegments)
        pitchAverageFilter.append(sumPitch)
    return pitchAverageFilter

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values.T)
    plt.pcolormesh(Y, X, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values[0], linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values[0], linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

def analyze(snd):
	# audio_path = 'C://Users/Fabi/Documents/presentation-analysis/audio/test_1.wav'
	# y, sr = librosa.load(audio_path, dtype=np.float32)
	# snd = parselmouth.Sound("./audio/nash_1.wav")
	# intensity = snd.to_intensity()
	# print(len(intensity.xs()))
	# spectrogram = snd.to_spectrogram()
	# print(len(spectrogram.x_grid()))
	# plt.figure()
	# draw_spectrogram(spectrogram)
	# plt.twinx()
	# draw_intensity(intensity)
	# plt.xlim([snd.xmin, snd.xmax])
	# plt.show() # or plt.savefig("spectrogram.pdf")
	# audio_path = './audio/monoton.wav'

	# y, sampling_rate = librosa.load(audio_path, dtype=np.float32)
	# print(len(y))

	segmentation_time = 0.1

	# Time of the Signal
	sr = int(SAMPLING_RATE * segmentation_time)
	# NoWindows = int(np.ceil(y.size/sr))
	NoWindows = 100 # todo: discuss which value should this have
	signal_avg = np.zeros(NoWindows)
	signal_max = np.zeros(NoWindows)
	signal_min = np.zeros(NoWindows)
	signal_std = np.zeros(NoWindows)
	intensitys = np.zeros(NoWindows)
	std = np.zeros((NoWindows,sr))

	# snd = parselmouth.Sound(audio_path)
	intensity = snd.to_intensity(time_step=0.1) # todo: check if this works
	pitch = snd.to_pitch(time_step=0.1)

	print(len(intensity.values[0]))

	for i in range(1, NoWindows+1):
		if i == NoWindows:
			signal_avg[i-1] = sum(abs(y[(i-1)*sr:])/(y[(i-1)*sr:]).size)
			signal_max[i-1] = max(abs(y[(i-1)*sr:]))
			signal_min[i-1] = min(abs(y[(i-1)*sr:]))
			std[i-1,:y.size-(i-1)*sr] = y[(i-1)*sr:].T
			signal_std[i-1] = stat.stdev(std[i-1,:])
		else:
			signal_avg[i-1] = sum(abs(y[(i-1)*sr:i*sr])/y[(i-1)*sr:i*sr].size)
			signal_max[i-1] = max(abs(y[(i-1)*sr:i*sr]))
			signal_min[i-1] = min(abs(y[(i-1)*sr:i*sr]))
			std[i-1,:] = y[(i-1)*sr:i*sr]
			signal_std[i-1] = stat.stdev(std[i-1,:])

	#print(intensity)

	# Make a new figure
	plt.figure(figsize=(12,4))
	order = 7
	fs = 2250.0       # sample rate, Hz
	cutoff = 50  # desired cutoff frequency of the filter, Hz

	plt.plot(averageFilter(intensity.values[0],5) , 'c')
	plt.plot(gaussian_filter(averageFilter(intensity.values[0],5), sigma=2) , 'r')
	intensity_avg = sum(intensity.values[0])/len(intensity.values[0])
	print("average intensity: " + str(intensity_avg))
	p = np.poly1d([0,0,intensity_avg+2])
	x = np.arange(len(intensity.values[0]))
	z = p(x)
	plt.plot(x,z)
	p = np.poly1d([0,0,67]) #66
	x = np.arange(len(intensity.values[0]))
	z = p(x)
	plt.plot(x,z)
	#plt.plot(butter_lowpass_filter(intensity.values[0], cutoff, fs, order ), 'g')

	plt.figure(figsize=(12,4))
	plt.plot(intensity.values[0] , 'b')
	# Make a new figure
	plt.figure(figsize=(12,4))

	# show waveform    
	#plt.plot(averageFilter(pitch.selected_array['frequency'],5), 'r')
	pitch_avg = sum(pitch.selected_array['frequency'])/len(pitch.selected_array['frequency'])
	p = np.poly1d([0,0,pitch_avg+2])
	x = np.arange(len(pitch.selected_array['frequency']))
	z = p(x)
	plt.plot(x,z)

	pitchFilteredX = []
	pitchFilteredY = []
	for k in range(len(pitch.selected_array['frequency'])):
		if pitch.selected_array['frequency'][k] > 50 and pitch.selected_array['frequency'][k] < 300:
			pitchFilteredY.append(pitch.selected_array['frequency'][k])
			pitchFilteredX.append(k)


	plt.plot(pitch.selected_array['frequency'])
	plt.plot(gaussian_filter(pitch.selected_array['frequency'], sigma=3), 'y')
	#plt.plot(butter_lowpass_filter(pitch.selected_array['frequency'], cutoff, fs, order ), 'g')
	# Make a new figure
	plt.figure(figsize=(12,4))

	plt.plot(pitchFilteredX, pitchFilteredY)
	plt.plot(pitchFilteredX, gaussian_filter(pitchFiltered, sigma=3), 'y')

	plt.figure(figsize=(12,4))

	# show waveform
	plt.plot(y)

	#print("signal_avg is: ", signal_avg)
	#print("signal_max is: ", signal_max)
	#print("signal_min is: ", signal_min)
	 
	signal_avg_avg = sum(abs(signal_avg)/signal_avg.size)
	signal_max_max = max(abs(signal_max))
	signal_min_min = min(abs(signal_min))
	signal_std_std = stat.stdev(signal_std)
	 
	#print("signal_avg_avg is: ", signal_avg_avg)
	#print("signal_max_max is: ", signal_max_max)
	#print("signal_min_min is: ", signal_min_min)
	#print("signal_std_std is: ", signal_std_std)

###############################################################
def animate(i):
    global buffer,audio_data
    xs = i * np.arange(1024)
    ys = buffer + audio_data
    if len(buffer) < 10:
        ys = np.arange(1024)
    ax1.clear()
    ax1.plot(xs,ys)
    buffer = buffer + audio_data

def main():
######################LED STRIPS START#########################
    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clear', action='store_true', help='clear the display on exit')
    args = parser.parse_args()

    # Create NeoPixel object with appropriate configuration.
    strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
    # Intialize the library (must be called once before other functions).
    strip.begin()
	
	simpleColor(strip, COL_GREEN)

    # print ('Press Ctrl-C to quit.')
    # if not args.clear:
        # print('Use "-c" argument to clear LEDs on exit')

    # try:

        # while True:
            # """
            # print ('Color wipe animations.')
            # colorWipe(strip, Color(255, 0, 0))  # Red wipe
            # colorWipe(strip, Color(0, 255, 0))  # Blue wipe
            # colorWipe(strip, Color(0, 0, 255))  # Green wipe
            # print ('Theater chase animations.')
            # theaterChase(strip, Color(127, 127, 127))  # White theater chase
            # theaterChase(strip, Color(127,   0,   0))  # Red theater chase
            # theaterChase(strip, Color(  0,   0, 127))  # Blue theater chase
            # print ('Rainbow animations.')
            # rainbow(strip)
            # rainbowCycle(strip)
            # theaterChaseRainbow(strip)
            # """
            # if GPIO.input(24) == 0:
                # simpleColor(strip, COL_YELLOW)
            # elif GPIO.input(25) == 0:
                # simpleColor(strip, COL_RED)
            # else:
                # simpleColor(strip, Color(0,0,0))

    # except KeyboardInterrupt:
        # if args.clear:
            # colorWipe(strip, Color(0,0,0), 10)
######################LED STRIPS STOP##########################
    stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=SAMPLING_RATE,
                output=False,
                input=True,
                stream_callback=callback)
    stream.start_stream()
    ani = animation.FuncAnimation(fig, animate, interval=100)
    plt.show() # what is this doing?
	analyze(ani) # todo: this might not work, check items
	# todo: call the simpleColor(strip, color) function, set the color 
	#		according to the analysis of the speech
    while stream.is_active():
        time.sleep(10)
        stream.stop_stream()
    stream.close()

    p.terminate()

def callback(in_data, frame_count, time_info, flag):
    global b,a,fulldata,dry_data,frames,buffer,audio_data 
    audio_data = np.fromstring(in_data, dtype=np.float32)
    dry_data = np.append(dry_data,audio_data)
    #do processing here
    fulldata = np.append(fulldata,audio_data)
    return (audio_data, pyaudio.paContinue)

main()