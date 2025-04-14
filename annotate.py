#!/usr/bin/env python

# Python
import argparse
import glob
import json
import os
import sys


# 3rd party
import exifread
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tabulate import tabulate
import datetime
import pandas as pd


class Photo:

    def __init__(self, filename, verbose=False):
        if not os.path.isfile(filename):
            raise ValueError("Could not find the file: %s" % filename)
        self._filename = filename
        self._read_and_downsample(verbose)
        self._choice = None

    def data(self):
        return self._data
    
    def choice(self):
        return self._choice
 
    def filename(self):
        return self._filename

    def __eq__(self, rhs):
        return self._filename == rhs._filename 

    def _read_and_downsample(self, verbose=False):
        """
        Reads the image, performs rotation, and downsamples.
        """

        #----------------------------------------------------------------------
        # read image

        data = mpimg.imread(self._filename)

        #----------------------------------------------------------------------
        # downsample

        # the point of downsampling is so the images can be redrawn by the
        # display as fast as possible, this is so one can iterate though the
        # image set as quickly as possible.  No one want"s to wait around for
        # the fat images to be loaded over and over.

        # dump downsample, just discard columns-n-rows
        M, N = data.shape[0:2]
        MN = max([M,N])
        step = int(MN / 800)
        if step == 0: step = 1

        if data.ndim == 3:
            data = data[ 0:M:step, 0:N:step, :]
        elif data.ndim == 2:
            data = data[ 0:M:step, 0:N:step]
        else:
            raise RuntimeError(f"Don't know how to deal with {data.ndim} dimensions!")

        #----------------------------------------------------------------------
        # rotate

        # read orientation with exifread
        with open(self._filename, "rb") as fin:
            tags = exifread.process_file(fin)

        r = "Horizontal (normal)"
        try:
            r = str(tags["Image Orientation"])
        except:
            pass

        # rotate as necessary

        if r == "Horizontal (normal)":
            pass
        elif r == "Rotated 90 CW":
            data = np.rot90(data, 3)
        elif r == "Rotated 90 CCW":
            data = np.rot90(data, 1)
        elif r == "Rotated 180":
            data = np.rot90(data, 2)
        else:
            print("Ignoring unhandled rotation '{r}'")

        self._data = data

        if verbose:
            sys.stdout.write(".")
            sys.stdout.flush()

def position_figure(fig):
    # put figure in center of the screen.
    backend = mpl.get_backend().lower()
    
    if backend not in {"tkagg"}:
        return

    # Get the screen size
    fig_manager = plt.get_current_fig_manager()

    # Set the position.
    if backend == "tkagg":
        fig.canvas.manager.window.wm_geometry(f"+150+90")


class Display(object):

    LEFT = 0
    RIGHT = 1
    SKIP = 2

    def __init__(self, f1, title = None, figsize = None):
        self._choice = None
        assert isinstance(f1, Photo)
        if figsize is None:
            figsize = [20,8]

        fig = plt.figure(figsize=figsize)

        h = 10
        image_viewer = plt.subplot2grid((h, 1), (1, 0), rowspan = h - 2)
        left_button = plt.subplot2grid((h, 3), (h - 1, 0))
        skip_button = plt.subplot2grid((h, 3), (h - 1, 1))
        right_button = plt.subplot2grid((h, 3), (h - 1, 2))

        kwargs = dict(ha = "center", va = "center", fontsize=18)
        left_button.text(0.5, 0.5, s = "No (left)", **kwargs)
        skip_button.text(0.5, 0.5, s = "Skip (down)", **kwargs)
        right_button.text(0.5, 0.5, s = "Yes (right)", **kwargs)
        self._fig = fig
        self._ax_select_left = left_button
        self._ax_select_skip = skip_button
        self._ax_select_right = right_button

        fig.subplots_adjust(
            left = 0.02,
            bottom = 0.02,
            right = 0.98,
            top = 0.98,
            wspace = 0.05,
            hspace = 0,
        )

        image_viewer.imshow(f1.data())

        for ax in [image_viewer, left_button, skip_button, right_button]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        self._attach_callbacks()

        if title:
            fig.suptitle(title, fontsize=14)

        position_figure(fig)
        plt.show()

    def _on_click(self, event):

        if event.inaxes == self._ax_select_left:
            self._choice = Display.LEFT
            plt.close(self._fig)

        elif event.inaxes == self._ax_select_right:
            self._choice = Display.RIGHT
            plt.close(self._fig)

        elif event.inaxes == self._ax_select_skip:
            self._choice = Display.SKIP
            plt.close(self._fig)

    def _on_key_press(self, event):

        if event.key == "left":
            self._choice = Display.LEFT

            plt.close(self._fig)
        elif event.key == "right":
            self._choice = Display.RIGHT
            plt.close(self._fig)
        elif event.key == "down":
            self._choice = Display.SKIP
            plt.close(self._fig)

    def _attach_callbacks(self):
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("key_release_event", self._on_key_press)

class ImageTable:
    def __init__(self, question):
        self._question = question
        self._photos = {}

    def add_photo(self, filename_or_photo, verbose=False):
        if isinstance(filename_or_photo, str):
            filename = filename_or_photo
            if filename not in self._photos:
                self._photos[filename] = Photo(filename, verbose=verbose)

        elif isinstance(filename_or_photo, Photo):
            photo = filename_or_photo
            if photo.filename() not in self._photos:
                self._photos[photo.filename()] = photo

    def annotate(self, figsize):

        n_photos = len(self._photos)
        keys = list(self._photos.keys())
        
        np.random.shuffle(keys)
        for j in range(0, n_photos):
            filename = keys[j]
            title = "Round %d / %d\n%s\n%s" % (
                j + 1, n_photos, filename.split("/")[-1], self._question)
            
            img = self._photos[filename]
            d = Display(img, title, figsize)
            if d._choice == Display.LEFT:
                
                img._choice = "no"
                pass

            elif d._choice == Display.RIGHT:
                img._choice = "yes"
                pass 

            elif d._choice == Display.SKIP:
                img._choice = None
                print("Skipping", filename)
                pass

            elif d._choice == None:
                print("Aborting!")
                break
            else:
                raise RuntimeError(f"oops, found a bug!  d._choice == '{d._choice}'")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "photo_dir",
        help = "The photo directory to scan for .jpg images"
    )
    parser.add_argument(
        "-q",
        "--question",
        type= str,
        required= True,
        help = "The question to ask about the images"
    )
    parser.add_argument(
        "-rq",
        "--reverse_question",
        type= str,
        required= False,
        default= None,
        help = "(Optional) The reverse question to ask about the images"
    )
    parser.add_argument(
        "-a",
        "--annotator",
        type= str,
        required= False,
        default= None,
        help = "(Optional) The name of the annotator"
    )
    parser.add_argument(
        "-f",
        "--figsize",
        nargs = 2,
        type = int,
        default = [20, 10],
        help = "Specifies width and height of the Matplotlib figsize (20, 10)"
    )
    
    args = parser.parse_args()
    assert os.path.isdir(args.photo_dir)
    os.chdir(args.photo_dir)

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the ranking table and add photos to it.
    table = ImageTable(args.question)

    #--------------------------------------------------------------------------
    # Read in table .json if present

    sys.stdout.write("Reading in photos and downsampling ...")
    sys.stdout.flush()

    #--------------------------------------------------------------------------
    # glob for files, to include newly added files
    filelist = glob.glob("*.jpg")
    sys.stdout.write(f" found {len(filelist)} files ")
    sys.stdout.flush()
    for filename in filelist:
        table.add_photo(filename, verbose=True)

    # Done reading & downsampling.
    print(" done!")

    #--------------------------------------------------------------------------
    # Rank the photos!
    table.annotate(args.figsize)

    #--------------------------------------------------------------------------
    # save the table
    rows = []
    for filename, photo in table._photos.items():
        if photo.choice() is not None:
            rows.append({
                "filename": filename,
                "annotator": args.annotator,
                "question": args.question,
                "choice": photo.choice()
            })

            if args.reverse_question:
                rows.append({
                    "filename": filename,
                    "annotator": args.annotator,
                    "question": args.reverse_question,
                    "choice": "no" if photo.choice() == "yes" else "yes"
                })
        
    output_file = f"{session_id}_annotate.csv"

    sys.stdout.write(f"Writing out {len(rows)} rows to {output_file} ...")

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
