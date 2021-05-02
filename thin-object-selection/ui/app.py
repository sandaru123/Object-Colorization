import tkinter as tk
import tkinter.filedialog as filedialog
from PIL import Image, ImageTk,ImageChops
import cv2
import numpy as np
import copy
import ui.seg_model as seg_model
import dataloaders.helpers as helpers
import imageio
import os, os.path
from os import path
import colorizer as colorizers

class InteractiveDemo:
    def __init__(self, master=tk.Tk()):
        self.master = master
        self.master.title("Deep Interactive Thin Object Selection Demo")
        self.max_size = 1200

        self.mask_image_objectName= ''
        self.filename = 'Test Images/dambassuu.jpeg'
        self.image = np.array(Image.open(self.filename))
        self.clicks = []        # store user clicks
        self.mask = []          # store segmentation mask
        self._resize_image()    # resize image to fit the canvas
        self.canvas_image = self.resize_image.copy()

        self.canvas = ImageTk.PhotoImage(Image.fromarray(self.canvas_image))
        self.label = tk.Label(self.master, image=self.canvas)
        self.label.pack(side=tk.BOTTOM)
        self.label.bind('<Button-1>', self.on_click)

        self.btn_open_image = tk.Button(self.master, text='Open Image', command=self._open_image)
        self.btn_open_image.pack(side=tk.LEFT)
        self.btn_reset = tk.Button(self.master, text='Reset', command=self._reset)
        self.btn_reset.pack(side=tk.LEFT)
        self.btn_save_mask = tk.Button(self.master, text='Save Mask', command=self._save_mask)
        self.btn_save_mask.pack(side=tk.LEFT)

        self.mask_file = ''

    def _resize_image(self):
        """ Resize image to fit canvas. """
        h, w = self.image.shape[:2]
        max_side = np.maximum(h, w)
        if max_side > self.max_size:
            self.sc = self.max_size / max_side
            self.resize_image = cv2.resize(self.image, (0,0), fx=self.sc, fy=self.sc, interpolation=cv2.INTER_LINEAR)
        else:
            self.sc = 1
            self.resize_image = self.image.copy()

    def _init_seg_model(self, net, cfg, device):
        """ Initialize segmentation model. """
        self.model = seg_model.SegModel(net, cfg, device)

    def _reset(self):
        self.clicks = [] # store user clicks
        self.mask = []   # store segmentation mask
        self.canvas_image = self.resize_image.copy() # rest image
        self._update_canvas()

    def _open_image(self):
        """ Open an image. """
        def _open_image_dialog():
            filename = filedialog.askopenfilename(title='Open')
            return filename
        self.filename = _open_image_dialog()
        self.image = np.array(Image.open(self.filename))
        if len(self.image.shape) > 2 and self.image.shape[2] == 4:
            # convert the image from RGBA2RGB
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR)
        self._resize_image()
        self._reset()
        self._update_canvas()

    def _update_canvas(self):
        """ Update the canvas. """
        self.canvas = ImageTk.PhotoImage(Image.fromarray(self.canvas_image))
        self.label.config(image=self.canvas)
        self.master.update_idletasks()

    def on_click(self, event):
        self._draw_click([int(event.x), int(event.y)]) # draw the latest click
        self.clicks.append([int(event.x / self.sc), int(event.y / self.sc)])
        # Automatically perform segmentation when there are 4 clicks
        if len(self.clicks) == 6:
            self.segment()

    def segment(self):
        """ Perform segmentation. """
        clicks = np.array(self.clicks).astype(np.int)
        self.mask = self.model._segment(self.image, clicks)
        self._visualize_mask()

    def _draw_click(self, click):
        """ Draw user clicks on the canvas image. """
        cv2.circle(self.canvas_image, (click[0], click[1]), 8, [255,255,255], -1)
        self._update_canvas()

    def _visualize_mask(self):
        """ Visualize segmentation mask. """
        mask = cv2.resize(self.mask, (0,0), fx=self.sc, fy=self.sc, interpolation=cv2.INTER_LINEAR)
        if len(mask.shape) > 2 and mask.shape[2] == 4:
            # convert the image from RGBA2RGB
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR)
        self.canvas_image = helpers.mask_image(self.canvas_image, mask>0.5, color=[0,255,0], alpha=0.3)
        self._update_canvas()

    def _save_mask_img(self):
        """ Save segmentation mask. """
        masked_image = self.mask_file
        o_image = np.array(Image.open(self.filename))
        mask_ = np.array(Image.open(masked_image).resize(o_image.shape[1::-1], Image.BILINEAR))
        print(mask_.dtype, mask_.min(), mask_.max())
        # uint8 0 255
        mask_ = mask_ / 255
        print(mask_.dtype, mask_.min(), mask_.max())
        # float64 0.0 1.0
        mask_re = mask_.reshape(*mask_.shape, 1)
        dst = o_image * mask_re
        image_name_save = 'images/Object_image.png'
        print(image_name_save)
        Image.fromarray(dst.astype(np.uint8)).save(image_name_save)

        while True:
            maskImg_exist = path.exists(image_name_save)
            print(maskImg_exist)
            if maskImg_exist == False:
                print
                "not saved yet"
            else:
                print
                "saved"
                break
        #maskname
        mask1 = cv2.imread(image_name_save)
        diff_im = self.image - mask1
        bg_save_path = 'images/imgWithoutMask.png'
        cv2.imwrite(bg_save_path, diff_im)
        colorizers.colorizeImage(image_name_save, diff_im, bg_save_path)

    def _save_mask(self):
        """ Save segmentation mask. """
        mask = (self.mask * 255).astype(np.uint8)
        names = self.filename.split('/')
        img_name = names[-1].split('.')[0] + '_mask.png'
        filename = names[:-1]
        filename.append(img_name)
        filename = '/'.join(filename)
        imageio.imwrite(filename, mask)

        self.mask_file = filename

        print('Mask is saved to {}!'.format(filename))

        while True:
            m_image = 'images/'.join(filename)
            mask_exist = path.exists(m_image)
            if mask_exist == True:
                print
                "not saved yet"
            else:
                print
                "saved"
                break
        self._save_mask_img()




    def mainloop(self):
        self.master.mainloop()