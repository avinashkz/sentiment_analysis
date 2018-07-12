import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import time

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    data = open('sentiment.txt', 'r').read()
    lines = data.split('\n')
    xar = []
    yar = []
    
    x = 0
    y = 0
    for l in lines:
        x += 1
        if "pos" in l:
            y += 1
        elif "neg" in l:
            y -= 1
        xar.append(x)
        yar.append(y)
    
    ax1.clear()
    ax1.plot(xar, yar)
    
ani = animation.FuncAnimation(fig, animate, interval = 1000)
plt.show()


