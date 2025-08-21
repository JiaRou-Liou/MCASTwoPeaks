# -*- coding: utf-8 -*-

"""

Observe how the parameters determine the behavior of two peaks in this 2 & 3 species 
mass-conserved activator-substrate (MCAS) model. 

Possible scenarios:
Competition: one peak decays, another grows

Coexistence: at least one peak saturates into a high plateau, another grows alongside it.

Equalization: With the addition of an indirect species (3 species model), both peaks grow alongside each other. 
The shorter one grows faster. 


Parameter sets examples: 
    Coexistence: a, b, c, d = 1, 1, 0, 0; 
    M_totalmass, A = 25.2, 4/18
    
    
    Competition: a, b, c, d = 1, 1, 0, 0; 
    M_totalmass, A = 15, 5/18
    
    Equalization: a, b, c, d = 1, 1, 0.1, 1; 
    M_totalmass, A = 25.2, 1/4

"""

#libraries
import numpy as np
import matplotlib.pyplot as py
from scipy.fft import fft, ifft
#import matplotlib.animation as animation       #Commented out when animation is not needed

# Parameters
L = 20         # Domain length
Nx = 128         # Number of grid points
dx = L / Nx
x = np.linspace(0, L, Nx, endpoint=False)
kx = np.fft.fftfreq(Nx,dx)*2*np.pi

Tmax = 100       #Simulation time (in seconds). Better to run for at least longer than 100s 
dt = 0.001      
Nt = int(Tmax / dt)     #Number of time steps

# Reaction parameters
a, b, c, d = 1, 1, 0.1, 1      #c is turned off for two species MCAS model. 
Du, Dv, Dw = 0.01, 1, 1     #Du<<1 is set for diffusion-limited regime

# ---- Initial Conditions ----
sigma = 0.3        #Width of the peaks. Tweak this to determine the peaks' height. 
x1, x2 = 0.25 * L, 0.75 * L         #Location of the two peaks

# Profiles of the unequal peaks (u's mass is set as unity initially)
u1 = np.exp(-((x - x1) ** 2) / (2 * sigma ** 2))
u2 = np.exp(-((x - x2) ** 2) / (2 * sigma ** 2))
u = 0.2 * u1 / np.sum(u1 * dx) + 0.8 * u2 / np.sum(u2 * dx) 

M_totalmass = 18*np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 5])    #Control the total mass M
w = np.zeros_like(x)        # Start with no buffer

def main(u, w, A, M_totalmass):
    """
    Record and evolve the two Gaussian peaks using spectral method. 
    
    WARNING: Make sure the peaks can sustain by setting the initial density of u<1 
    (lateral instability of the peak), and tweaking the height of the peak with sigma. 
    But be careful not to set the peak too sharp (sigma too small). The code might 
    runs into a RunTimeWarning that returns a nan. 
        
    Parameters
    ----------
        u: 1D array
           Profile/ height of the peaks of the species u. 
           
        w: 1D array
           Initilization of the indirect substrate. Maintain at 0 when c is turned off.
           
        A: float 
           Ratio of u over total mass (Indirectly specifies the amount of v).
           
        M_totalmass: float
                     Total mass of all species (M=sum(u+v+w)*dx). 
        
    Returns: 
        history: dict
                 Nested dictionary that records the spatial temporal information of the species
        
        """
    ix1 = int(Nx * x1 / L)      #Grid number of the two peaks
    ix2 = int(Nx * x2 / L)
    
    u = u*(M_totalmass*A)       #Example: A=1/21 (u:v = 1:20)
    u_mass = np.sum(u*dx)       #Initial mass of u is normalized to unity.
    print('hightest peak of u=',np.max(u))
    
    v = np.ones_like(x) * (M_totalmass - u_mass) / L 
    
    # — Storage for diagnostics —
    history = {
        'species': {
            'u1': np.zeros(Nt),
            'u2': np.zeros(Nt),
            'q':  np.zeros((Nt,len(u))),
            'total_mass': np.zeros(Nt),
            'u':np.zeros((Nt,len(u))),
            'v':np.zeros((Nt,len(u))),
            'w':np.zeros((Nt,len(u)))
        },
        'rates': {
            'Nu_x1': np.zeros(Nt),
            'Nw_x1': np.zeros(Nt),
            'Nu_x2': np.zeros(Nt),
            'Nw_x2': np.zeros(Nt),
            'net_growth':np.zeros((Nt,len(u)))}}
    
    # --- Spectral method for PDEs ---
    uk = fft(u)
    vk = fft(v)
    wk = fft(w)
    
    
    Luk = -Du*kx**2
    Lvk = -Dv*kx**2
    Lwk = -Dw*kx**2
    
    Nu = a*(u**2)*v-(b+c)*u
    Nv = -a*(u**2)*v+b*u+d*w
    Nw = c*u-d*w
    
    Nuk = fft(Nu)
    Nvk = fft(Nv)
    Nwk = fft(Nw)
    
    oNuk = Nuk
    oNvk = Nvk
    oNwk = Nwk
    
    history['species']['u1'][0] = u[ix1]
    history['species']['u2'][0] = u[ix2]

    history['species']['total_mass'][0] = np.sum(u + v + w) * dx

    history['species']['u'][0,:] = u
    history['species']['v'][0,:] = v
    history['species']['q'][0,:] = v+(Du/Dv)*u
    history['species']['w'][0,:] = w

    history['rates']['Nu_x1'][0] = Nu[ix1]
    history['rates']['Nw_x1'][0] = Nw[ix1]

    history['rates']['net_growth'][0,:]=0
    
    for i in range(1,Nt):
        uk = ((1+Luk*dt/2)*uk+(1/2)*(3*Nuk-oNuk)*dt)/(1-(1/2)*Luk*dt)
        vk = ((1+Lvk*dt/2)*vk+(1/2)*(3*Nvk-oNvk)*dt)/(1-(1/2)*Lvk*dt)
        wk = ((1+Lwk*dt/2)*wk+(1/2)*(3*Nwk-oNwk)*dt)/(1-(1/2)*Lwk*dt)
        
        u = np.real(ifft(uk))
        v = np.real(ifft(vk))
        w = np.real(ifft(wk))
        
        oNuk = Nuk
        oNvk = Nvk
        oNwk = Nwk
        
        Nu = a*(u**2)*v-(b+c)*u    #Nonlinear reaction rate f(u,v)=a*(u**2)*v-b*u
        Nv = -a*(u**2)*v+b*u+d*w
        Nw = c*u-d*w
        
        Nuk = fft(Nu)
        Nvk = fft(Nv)
        Nwk = fft(Nw)
        
        # net growth
        lap_u = (np.roll(u,-1) - 2*u + np.roll(u,1)) / dx**2
        net_u = Du * lap_u + Nu


        
        history['species']['u1'][i] = u[ix1]
        history['species']['u2'][i] = u[ix2]
        
        
        history['species']['total_mass'][i] = np.sum(u + v + w) * dx

        history['species']['u'][i,:] = u
        history['species']['v'][i,:] = v
        history['species']['q'][i,:] = v+(Du/Dv)*u
        history['species']['w'][i,:] = w
        
        history['rates']['Nu_x1'][i] = Nu[ix1]
        history['rates']['Nw_x1'][i] = Nw[ix1]
        
        
        history['rates']['net_growth'][i,:]=net_u
        
    
    return history

one_instance = main(u,w,1/4,M_totalmass[2])
u1_hist = one_instance['species']['u1']
u2_hist = one_instance['species']['u2']

q_hist = one_instance['species']['q']

Nu_hist = one_instance['rates']['Nu_x1']
Nw_hist = one_instance['rates']['Nw_x1']     


print(q_hist[0,0],q_hist[-1,0])      
    
    

# — Plotting results —
t = np.arange(Nt) * dt
fig, ax = py.subplots(2,1, figsize=(12,8), sharex=True)

ax[0].plot(t, u1_hist, label='u @ x1')
ax[0].plot(t,u2_hist, label='u @ x2')
ax[0].set_ylabel('u peaks')
ax[0].legend()



ax[1].plot(t, Nu_hist, label='rate u @ x1')
ax[1].plot(t, Nw_hist, label='rate w @ x1')
ax[1].set_ylabel('reaction rate')
ax[1].set_xlabel('Time')
ax[1].set_title('peak 2 reaction rate')
ax[1].legend()
fig.tight_layout()
py.show()
print(main.__doc__)

"""
#--*ANIMATION to track time evolution of the substrate in space*--

#u, v, w, q profiles to track the spatial evolution of each quantity
u = history['species']['u'] 
v = history['species']['v']
w = history['species']['w']
q =history['species']['q']
net_growth_rate = history['rates']['net_growth']


# create a figure with two subplots
fig1, (ax1, ax2) = py.subplots(2,1)

# intialize two line objects (one in each axes)
line1, = ax1.plot([], [], lw=2)
line2, = ax2.plot([], [], lw=2, color='r')
line = [line1, line2]

# the same axes initalizations as before (just now we do it for both of them)
#for ax in [ax1, ax2]:
ax1.set_ylim(0,np.max(u)+1)
ax1.set_xlim(0, x[-1])
ax1.grid()


ax2.set_ylim(0,np.max(q[-1,:])+1)
ax2.set_xlim(0, x[-1])
ax2.grid()

# Create a time label in the upper-left corner of the first subplot
time_template = 'Time = {:.2f}'
time_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes)

# initialize the data arrays 
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    return line + [time_text]

def run(i):
    y1 = u[i, :]
    y2 = q[i, :]

    line[0].set_data(x, y1)
    line[1].set_data(x, y2)
    time_text.set_text(time_template.format(t[i]))
    return line + [time_text]

# Build animation, but only every 100th frame
sample_interval = 100
frame_indices = range(0, len(t), sample_interval)
ani = animation.FuncAnimation(fig1, run, frames=frame_indices,init_func=init,blit=True, interval=10) ## interval: ms between frames during playback
fig1.suptitle('plot', fontsize=14)
FFwriter = animation.FFMpegWriter(fps=10)
ani.save('animation.mp4', writer = FFwriter)
    """
    
    
    
    
        
        
        
        
        
    
    








