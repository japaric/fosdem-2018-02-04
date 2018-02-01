import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import struct
import sys

# Helpers
def load_control_data(path):
    with open(path, "rb") as f:
        data = f.read()

    ary = []
    arz = []
    gx = []
    kalman = []
    pid = []
    i = 0
    N = 14
    while i < len(data):
        if i + N > len(data):
            break

        ary.append(struct.unpack('<h', data[i:i+2])[0])
        arz.append(struct.unpack('<h', data[i+2:i+4])[0])

        gx.append(struct.unpack('<h', data[i+4:i+6])[0])

        kalman.append(struct.unpack('<f', data[i+6:i+10])[0])

        pid.append(struct.unpack('<f', data[i+10:i+14])[0])

        i += N

    ary = np.array(ary) * K_A
    arz = np.array(arz) * K_A
    gx = np.array(gx) * K_G
    kalman = np.array(kalman)
    tilt = np.arctan2(ary, arz) * 180 / math.pi # (degrees / s)
    pid = np.array(pid) * 100 # duty cycle (%)

    return len(ary), ary, arz, gx, kalman, tilt, pid

def load_cpu_data(path):
    with open(path, "rb") as f:
        data = f.read()

    CPU_FREQ = 64e6
    sleep = []
    i = 0
    N = 14
    while i < len(data):
        if i + N > len(data):
            break

        sleep.append(struct.unpack('<I', data[i:i+4])[0])

        i += N

    cpu = (1 - np.array(sleep) * FREQ / CPU_FREQ) * 100

    return len(cpu), cpu

def load_log_data(path):
    with open(path, "rb") as f:
        data = f.read()

    ary = []
    arz = []
    gx = []
    kalman = []
    i = 0
    N = 10
    while i < len(data):
        if i + N > len(data):
            break

        ary.append(struct.unpack('<h', data[i:i+2])[0])
        arz.append(struct.unpack('<h', data[i+2:i+4])[0])
        gx.append(struct.unpack('<h', data[i+4:i+6])[0])
        kalman.append(struct.unpack('<f', data[i+6:i+10])[0])

        i += N

    ary = np.array(ary) * K_A
    arz = np.array(arz) * K_A
    gx = np.array(gx) * K_G
    kalman = np.array(kalman)
    tilt = np.arctan2(ary, arz) * 180 / math.pi

    return len(ary), ary, arz, gx, kalman, tilt

# Style
sns.set()

# Constants
K_A = 2 / (1 << 15) # accelerometer sensitivity
K_G = 250 / (1 << 15) # gyroscope sensitivity
FREQ = 512
DT = 1 / FREQ

n, ary, arz, gx, kalman, tilt = load_log_data('flat.data')

# accelerometer-still.svg
ax = plt.subplot2grid((2, 2), (0, 0))
plt.plot(np.arange(0, n) * DT, ary)
plt.ylabel('Acceleration (g)')
ax.set_title(r'$G_y$', color='green')
plt.margins(0)

ax = plt.subplot2grid((2, 2), (0, 1))
plt.plot(np.arange(0, n) * DT, arz)
ax.set_title(r'$G_z$', color='blue')
plt.margins(0)

ax= plt.subplot2grid((2, 2), (1, 0), colspan=2)
plt.plot(np.arange(0, n) * DT, tilt)
mu = np.mean(tilt)
plt.plot([0, n * DT], np.ones(2) * mu, label='mean', linewidth=3)
plt.ylabel('Tilt (degrees)')
plt.xlabel('Time (seconds)')
ax.set_title(r'$atan(G_x / G_y)$')
plt.legend()
plt.margins(0)

plt.suptitle('Lying flat and still')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('accelerometer-still.svg')
plt.close()

# gyroscope-still.svg
plt.subplot(211)
plt.plot(np.arange(0, n) * DT, gx)
plt.plot([0, n*DT], np.ones(2) * np.mean(gx), linewidth=3, label='bias')
plt.ylabel('Angular rate (degrees / s)')
plt.xlabel('Time (seconds)')
plt.legend()
plt.margins(0)

plt.subplot(212)
plt.plot(np.arange(0, n) * DT, np.cumsum(gx) * DT)
plt.ylabel('Tilt (degrees)')
plt.xlabel('Time (seconds)')
plt.margins(0)

plt.suptitle('Lying flat and still')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('gyroscope-still.svg')
plt.close()

# gyroscope-calibrated.svg
plt.subplot(211)
plt.plot(np.arange(0, n) * DT, gx - np.mean(gx))
plt.ylabel('Angular rate (degrees / s)')
plt.xlabel('Time (seconds)')
plt.margins(0)

plt.subplot(212)
plt.plot(np.arange(0, n) * DT, np.cumsum(gx - np.mean(gx)) * DT)
plt.ylabel('Tilt (degrees)')
plt.xlabel('Time (seconds)')
plt.margins(0)

plt.suptitle('Lying flat and still (calibrated)')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('gyroscope-calibrated.svg')
plt.close()

# kalman-flat.svg
plt.title('Lying flat and still')
plt.plot(np.arange(0, n) * DT, tilt, '.', label='Accelerometer', markersize=5)
plt.plot(np.arange(0, n) * DT, kalman, label='Kalman filter', linewidth=3)
plt.ylabel('Tilt (degrees)')
plt.xlabel('Time (seconds)')
plt.legend()
plt.margins(0)

plt.tight_layout()
plt.savefig('kalman-still.svg')
plt.close()

# motion.data
n, ary, arz, gx, kalman, tilt = load_log_data('motion.data')

## accelerometer-motion.svg
plt.title('Planar motion')
plt.plot(np.arange(0, n) * DT, tilt)
plt.ylabel('Tilt (degrees)')
plt.xlabel('Time (seconds)')
plt.margins(0)

plt.tight_layout()
plt.savefig('accelerometer-motion.svg')
plt.close()

# rotate.data
n, ary, arz, gx, kalman, tilt = load_log_data('rotate.data')

## kalman-rotate.svg
plt.title('Rotating slowly')
plt.plot(np.arange(0, n) * DT, tilt, label='Accelerometer')
plt.plot(np.arange(0, n) * DT, kalman, label='Kalman filter', linewidth=3)
plt.ylabel('Tilt (degrees)')
plt.xlabel('Time (seconds)')
plt.legend()
plt.margins(0)

plt.tight_layout()
plt.savefig('kalman-rotate.svg')
plt.close()

# rotate-fast.data
n, ary, arz, gx, kalman, tilt = load_log_data('rotate-fast.data')

## kalman-rotate-fast.svg
plt.title('Rotating fast')
plt.plot(np.arange(0, n) * DT, tilt, label='Accelerometer')
plt.plot(np.arange(0, n) * DT, kalman, label='Kalman filter', linewidth=3)
plt.ylabel('Tilt (degrees)')
plt.xlabel('Time (seconds)')
plt.legend()
plt.margins(0)

plt.tight_layout()
plt.savefig('kalman-rotate-fast.svg')
plt.close()

# stable.data
n, ary, arz, gx, kalman, tilt, pid = load_control_data('stable.data')

# stable.svg
ax = plt.subplot(211)
plt.plot(np.arange(0, n) * DT, kalman)
plt.plot([0, n * DT], [10, 10], label='Set point')
plt.ylabel('Tilt (degrees)')
plt.xlabel('Time (seconds)')
plt.legend()
ax.set_title('y(t), process variable')
plt.margins(0)

ax = plt.subplot(212)
plt.plot(np.arange(0, n) * DT, pid)
plt.ylabel('Duty cycle (%)')
plt.xlabel('Time (seconds)')
ax.set_title('u(t), control variable')
plt.margins(0)

plt.suptitle('Stable PID controller')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('stable.svg')
plt.close()

# unstable.data
n, ary, arz, gx, kalman, tilt, pid = load_control_data('unstable.data')

# unstable.svg
ax = plt.subplot(211)
plt.plot(np.arange(0, n) * DT, kalman)
plt.plot([0, n * DT], [10, 10], label='Set point')
plt.ylabel('Tilt (degrees)')
plt.xlabel('Time (seconds)')
plt.legend()
ax.set_title('y(t), process variable')
plt.margins(0)

ax = plt.subplot(212)
plt.plot(np.arange(0, n) * DT, pid)
plt.ylabel('Duty cycle (%)')
plt.xlabel('Time (seconds)')
ax.set_title('u(t), control variable')
plt.margins(0)

plt.suptitle('Unstable PID controller')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('unstable.svg')
plt.close()

# forward.data
n, ary, arz, gx, kalman, tilt, pid = load_control_data('forward.data')

# forward.svg
ax = plt.subplot(211)
plt.plot(np.arange(0, n) * DT, kalman)
plt.plot([0, n * DT], [4, 4], label='Set point')
plt.ylabel('Tilt (degrees)')
plt.xlabel('Time (seconds)')
plt.legend()
ax.set_title('y(t), process variable')
plt.margins(0)

ax = plt.subplot(212)
plt.plot(np.arange(0, n) * DT, pid)
plt.ylabel('Duty cycle (%)')
plt.xlabel('Time (seconds)')
ax.set_title('u(t), control variable')
plt.margins(0)

plt.suptitle('Moving forward')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('forward.svg')
plt.close()

# cpu.data
n, cpu = load_cpu_data('cpu.data')

# cpu.svg
plt.title('CPU monitor')
plt.plot(np.arange(2, n) * DT, cpu[2:])
plt.ylabel('CPU usage (%)')
plt.xlabel('Time (seconds)')
plt.margins(0)

plt.tight_layout()
plt.savefig('cpu.svg')
plt.close()
