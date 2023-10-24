"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2 as cv

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

from models.percentlayer import PercentLayer

def rescale2rgb(target):
    # rescale original to 8 bit values [0,255]
    x0 = np.min(target)
    x1 = np.max(target)
    y0 = 0
    y1 = 255.0
    i8 = ((target - x0) * ((y1 - y0) / (x1 - x0))) + y0

    # # create new array with rescaled values and unsigned 8 bit data type
    o8 = i8.astype(np.uint8)

    # calculate porosity
    img = cv.medianBlur(o8, 5)
    return img

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    porosityList = np.array([])
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        
        percents = model.percent.clone().detach().squeeze()
        fake_img = visuals['fake_B'].detach().cpu().numpy()
        
        # img process for calculate
        img= rescale2rgb(fake_img[0,0])
        cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)


         # area finding
        # Threshold the image to create a binary image
        ret, thresh = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, 2, 1)

        cnt = contours
        big_contour = []
        max = 0
        for j in cnt:
            area = cv.contourArea(j)  #--- find the contour having biggest area ---
            if (area > max):
                max = area
                big_contour = j
        if len(big_contour) != 0:
            solid_percent = percents[0].cpu().numpy()
            spercent_list = np.array([])
            for idx, j in np.ndenumerate(solid_percent):
                if (cv.pointPolygonTest(big_contour,(idx[1], idx[0]), False) > 0):
                    spercent_list = np.append(spercent_list, 1-solid_percent[idx[0], idx[1]])
            porosity = spercent_list.sum() / spercent_list.size
            print(f"Porosity >>>> {porosity}")
        else:
            porosity = 0
            print("Empty")

        
        porosityList = np.append(porosityList, porosity)

        ###### DEBUG #####
        # cv.drawContours(cimg, big_contour, -1, (0, 255, 0), 2)
        # cv.imwrite('test.jpg', cimg)

        if opt.visual or opt.save_percent:
            # save percent info and image for plotly visualizing
            percents_np = percents.view(3, -1).detach().cpu().numpy()
            target = percents_np
            target = np.array(list(zip(target[0], target[1], target[2])))

            np.save(f'./percentOutput/percent_np/percent_{i}.npy', target)
            np.save(f'./percentOutput/image_np/img_{i}.npy', fake_img)

    df = pd.DataFrame(porosityList)
    df.to_csv('./percentOutput/porosityTmpOutput.csv')

    webpage.save()  # save the HTML

    if opt.visual:
        from percentOutput.visualize import plotly_visual 
        plotly_visual(opt.solid_ct, opt.water_ct, opt.air_ct)