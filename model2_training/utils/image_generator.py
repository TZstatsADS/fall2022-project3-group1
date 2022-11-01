"""
Author: Chengming He
A customized image generator for handling 2 inputs and able to perform common augmentations
"""


import numpy as np
from matplotlib import pyplot as plt
import os

try:
    from scipy.ndimage.interpolation import rotate
except ModuleNotFoundError:
    os.system('pip install scipy')
    from scipy.ndimage.interpolation import rotate

class ImageGenerator(object):
    def __init__(self, x, y,c):
        
        self.x = x
        self.y = y
        self.c = c
        self.N, self.height, self.width, self.channels, = x.shape
        self.is_horizontal_flip = False
        self.is_vertical_flip = False
        self.is_add_noise = False
        self.is_bright = False
        self.is_translate = False
        self.is_rotate = False        
        self.translated = None
        self.rotated = None
        self.flipped = None
        self.added = None
        self.bright = None
        self.x_aug = self.x.copy()
        self.y_aug = self.y.copy()
        self.c_aug = self.c.copy()
        self.N_aug = self.N    
    def create_aug_data(self):
        if self.translated:
            self.x_aug = np.vstack((self.x_aug,self.translated[0]))
            self.y_aug = np.vstack((self.y_aug,self.translated[1]))
            self.c_aug = np.hstack((self.c_aug,self.translated[2]))
        if self.rotated:
            self.x_aug = np.vstack((self.x_aug,self.rotated[0]))
            self.y_aug = np.vstack((self.y_aug,self.rotated[1]))
            self.c_aug = np.hstack((self.c_aug,self.rotated[2]))
        if self.flipped:
            self.x_aug = np.vstack((self.x_aug,self.flipped[0]))
            self.y_aug = np.vstack((self.y_aug,self.flipped[1]))
            self.c_aug = np.hstack((self.c_aug,self.flipped[2]))
        if self.added:
            self.x_aug = np.vstack((self.x_aug,self.added[0]))
            self.y_aug = np.vstack((self.y_aug,self.added[1]))
            self.c_aug = np.hstack((self.c_aug,self.added[2]))
        if self.bright:
            self.x_aug = np.vstack((self.x_aug,self.bright[0]))
            self.y_aug = np.vstack((self.y_aug,self.bright[1]))
            self.c_aug = np.hstack((self.c_aug,self.bright[2]))
        self.N_aug = len(self.x_aug)
        print("Size of training data:{}".format(self.N_aug))
        
    def next_batch_gen(self, batch_size, shuffle=True):
        num_batch = self.N // batch_size
        batch_count = 0
        while True:
            if batch_count < num_batch:
                batch_count += 1
                yield [self.x_aug[(batch_count-1)*batch_size:batch_count*batch_size,],self.y_aug[(batch_count-1)*batch_size:batch_count*batch_size]],self.c_aug[(batch_count-1)*batch_size:batch_count*batch_size]
            else:
                mask = np.random.permutation(self.x_aug.shape[0])
                self.x_aug, self.y_aug ,self.c_aug  = self.x_aug[mask], self.y_aug[mask],self.c_aug[mask]
                batch_count = 0



    def show(self, images):

        fig = plt.figure(figsize=(10, 10))
        for i in range(16):
            ax = fig.add_subplot(4, 4, i+1)
            ax.imshow(images[i, :], 'gray')
            ax.axis('off')



    def translate(self, shift_height, shift_width):

        if not self.is_translate:
            self.is_translate = True
        translated = self.x.copy()
        translated = np.roll(translated,shift_height,axis=1)
        translated = np.roll(translated,shift_width,axis=2)
        self.translated = (translated,self.y.copy(),self.c.copy())
        return translated


    def rotate(self, angle=0.0):

        
        if not self.is_rotate:
            self.is_rotate = True
        rotated = self.x.copy()
        for i in range(self.N):
            rotated[i] = rotate(rotated[i],angle,reshape=False)
        self.rotated = (rotated,self.y.copy(),self.c.copy())
        return rotated
        

    def flip(self, mode='h'):
        
        assert mode == 'h' or 'v' or 'hv'
        if mode == 'h':
            flipped = np.flip(self.x.copy(), axis=2)
            self.is_horizontal_flip = not self.is_horizontal_flip
        elif mode == 'v':
            flipped = np.flip(self.x.copy(), axis=1)
            self.is_vertical_flip = not self.is_vertical_flip
        elif mode == 'hv':
            flipped = np.flip(np.flip(self.x.copy(), axis=0), axis=1)
            self.is_horizontal_flip = not self.is_horizontal_flip
            self.is_vertical_flip = not self.is_vertical_flip
        else:
            raise ValueError('Mode should be \'h\' or \'v\' or \'hv\'')
        print('Vertical flip: ', self.is_vertical_flip, 'Horizontal flip: ', self.is_horizontal_flip)
    
        self.flipped = (flipped,self.y.copy(),self.c.copy())
        return flipped

    
    def add_noise(self, portion, amplitude):
        
        assert (portion >= 0 ) & (portion <= 1)
        if not self.is_add_noise:
            self.is_add_noise = True
        added = self.x.copy()
        mask = np.random.choice(self.N,int(portion*self.N))
        
        added[mask] += np.random.normal(scale=amplitude,size=(len(mask),self.height, self.width,self.channels)).astype(int)
        self.added = (added[mask],self.y.copy()[mask],self.c.copy()[mask])
        return added



    def brightness(self, factor):

        assert factor >= 1
        if not self.is_bright:
            self.is_bright = True
        bright = self.x.copy()
        for i in range(bright.shape[0]):
            bright[i, :, :, :] = (bright[i,:,:,:] * factor).astype(int)
            bright[i, :, :, :][bright[i,:,:,:] >= 255] = 255
            
        self.bright = (bright, self.y.copy(),self.c.copy())
        print("Brightness increased by a factor of:", factor)
        return bright

      