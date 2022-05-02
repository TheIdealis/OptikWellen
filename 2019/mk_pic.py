import numpy as np
import pylab as plt
from scipy.fftpack import fft, fftfreq, fftshift, ifft, ifftshift
plt.style.use('default')
from convert_wavelength import wl2color, wl2n
from itertools import product

def get_rays(a, n=1.333):
    if abs(a) >= 1:
        return [[[-2, 2], [a, a]]]
    x1 = np.sqrt(1 - a**2)
    rays = [[[-2, -x1], [a, a]]]
    
    alpha = np.arcsin(a)
    beta = np.arcsin(np.sin(alpha)/n)
    gamma = np.pi/2 -alpha + beta
    d = 2 * np.cos(beta)
    y2 = a - d*np.cos(gamma)
    x2 = d * np.sin(gamma) - x1   
    rays.append([[-x1, x2], [a, y2]])
    
    delta = 2*beta - (np.pi/2 - gamma)
    y3 = y2 - d*np.sin(delta)
    x3 = x2 - d * np.cos(delta)
    rays.append([[x2, x3], [y2, y3]])
    
    tau = 4*beta - 2*alpha
    x4 = -2
    # not working for small refractive indices
    y4 = y3 - (2 - abs(x3)) * np.tan(tau) 
    rays.append([[x3, x4], [y3, y4]])
    
    return rays

n1, n2 = 10, 10

a_s = np.linspace(0., .999, n1)
wl_s = np.linspace(380, 750, n2)

fig, ax = plt.subplots(figsize=(10, 10))
ax.clear()
ax.set_xlim([-2., 2.])
ax.set_ylim([-2., 2.])
ax.axis('off')
x_drop = np.linspace(-1, 1, 100)   
plt.plot(x_drop, np.sqrt(1 - x_drop**2), 'k-', lw=0.5)
plt.plot(x_drop, -np.sqrt(1 - x_drop**2), 'k-', lw=0.5)
ax.patch.set_facecolor("none")
ax.patch.set_edgecolor("none")
fig.canvas.draw()
w, h = fig.canvas.get_width_height()
img = np.float(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, -1).copy())
# img[img[:, :, -1] == 0] = 0

print(img.shape, np.max(img))

for a, wl in product(a_s, wl_s):
    rays = get_rays(a, wl2n(wl))
    for ray in rays:
        ax.clear()
        plt.plot(ray[0], ray[1], color=wl2color(wl), alpha=1)
        ax.set_xlim([-2., 2.])
        ax.set_ylim([-2., 2.])
        ax.axis('off')
        ax.patch.set_facecolor("none")
        ax.patch.set_edgecolor("none")
        fig.canvas.draw()
        img2 = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, -1).copy()
        img2[img2[:, :, -1] == 0] = 0
        img += img2/(n1*n2)
        
fig.clf()
plt.imshow(np.int(img))
# plt.subplots_adjust(0, 0, 1, 1)
plt.axis("off")
plt.show()