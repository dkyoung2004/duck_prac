#Wav파일을 스펙트로그램으로 플로팅 하는 코드
import matplotlib.pyplot as plot
from matplotlib.pyplot import figure
from scipy.io import wavfile
a = 1
zero = ''

# Read the wav file (mono)
for i in range(0, 1000):
    if a <= 9:
        zero = '00000'
    if 10 <= a & a <= 99:
        zero = '0000'
    if 100 <= a & a <= 999:
        zero = '000'
    if 1000 <= a & a <= 9999:
        zero = '00'
    if 10000 <= a & a <= 99999:
        zero = '0'
    file = './data/KsponSpeech_' + zero + str(a) + '.pcm.wav'
    samplingFrequency, signalData = wavfile.read(file)

    plot.figure(figsize=(4.76, 4.76))
    plot.specgram(signalData[:,0], Fs=samplingFrequency)
    plot.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    ax = plot.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    name = './image/' + str(a) + '.png'
    plot.savefig(name, bbox_inches='tight', pad_inches=0)
    a = a + 1
