"""
Contains all the types of interaction related to the GUI
Not related to automatic evaluation in the DAVIS dataset

You can inherit the Interaction class to create new interaction types
undo is (sometimes partially) supported
"""


import torch
import numpy as np
import cv2
import time
from model.aggregate import aggregate_sbg, aggregate_wbg
from util.tensor_util import pad_divide_by, unpad
from collections import deque
from copy import deepcopy
from interact.interactive_utils import color_map


max_history = 50

class Interaction:
    def __init__(self, image, prev_mask, true_size, controller):
        self.image = image # This image is PADDED
        self.prev_mask = prev_mask.clone() # This is also PADDED
        self.controller = controller
        self.start_time = time.time()
        self.history = deque(maxlen=max_history)

        self.h, self.w = true_size

        self.out_prob = None
        self.out_mask = None

    def undo(self):
        pass

    def can_undo(self):
        return len(self.history) > 0

    def predict(self):
        pass

class LocalInteraction(Interaction):
    # This interaction compress all the interactions done in the local stage into one
    # Performs stitching
    def __init__(self, image, prev_mask, true_size, bounding_box, region_prob, pad, local_pad):
        super().__init__(image, prev_mask, true_size, None)
        lx, ux, ly, uy = bounding_box
        self.out_prob = unpad(self.prev_mask, pad)
        region_prob = unpad(region_prob, local_pad)

        # Trim the margin since results at the boundary are not that trustworthy
        if (ux-lx) > 6 and (uy-ly) > 6:
            lx += 3
            ux -= 3
            ly += 3
            uy -= 3
            self.out_prob[:,:,ly:uy+1, lx:ux+1] = region_prob[:,:,3:-3,3:-3]
        else:
            self.out_prob[:,:,ly:uy+1, lx:ux+1] = region_prob
        self.out_prob, _ = pad_divide_by(self.out_prob, 16, self.out_prob.shape[-2:])
        self.out_mask = aggregate_sbg(self.out_prob, keep_bg=True)
        self.storage = None # Might be used outside

    def can_undo(self):
        return False

    def predict(self):
        return self.out_mask

class CropperInteraction(Interaction):
    # Turns a global map into a local map through cropping
    def __init__(self, image, prev_mask, pad, bounding_box):
        lx, ux, ly, uy = bounding_box
        true_size = (uy-ly+1, ux-lx+1)
        super().__init__(image, prev_mask, true_size, None)

        self.bounding_box = bounding_box # UN-PADDED
        unpad_prev_mask = unpad(self.prev_mask, pad)
        self.out_prob = unpad_prev_mask[:, :, ly:uy+1, lx:ux+1]
        self.out_prob, self.pad = pad_divide_by(self.out_prob, 16, self.out_prob.shape[-2:])
        self.out_mask = aggregate_sbg(self.out_prob, keep_bg=True)

        unpad_image = unpad(self.image, pad)
        self.im_crop = unpad_image[:, :, ly:uy+1, lx:ux+1]
        self.im_crop, _ = pad_divide_by(self.im_crop, 16, self.im_crop.shape[-2:])

    def can_undo(self):
        return False

    def predict(self):
        return self.out_mask

class FreeInteraction(Interaction):
    def __init__(self, image, prev_mask, true_size, num_objects, pad):
        """
        prev_mask should be in probabilities 
        """
        super().__init__(image, prev_mask, true_size, None)

        self.K = num_objects

        self.drawn_map = unpad(self.prev_mask, pad).detach().cpu().numpy()
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths = [self.curr_path]

        self.size = None
        self.surplus_history = False

    def set_size(self, size):
        self.size = size

    """
    k - object id
    vis - a tuple (visualization map, pass through alpha). None if not needed.
    """
    def push_point(self, x, y, k, vis=None):
        if vis is not None:
            vis_map, vis_alpha = vis
        selected = self.curr_path[k]
        selected.append((x, y))
        if len(selected) >= 2:
            for i in range(self.K):
                self.drawn_map[i,0] = cv2.line(self.drawn_map[i,0], 
                    (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                    (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                    int((i+1)==k), thickness=self.size)

            # Plot visualization
            if vis is not None:
                # Visualization for drawing
                if k == 0:
                    vis_map = cv2.line(vis_map, 
                        (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                        (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                        color_map[k], thickness=self.size)
                else:
                    vis_map = cv2.line(vis_map, 
                        (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                        (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                        color_map[k], thickness=self.size)
                # Visualization on/off boolean filter
                vis_alpha = cv2.line(vis_alpha, 
                    (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                    (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                    0.75, thickness=self.size)

        if vis is not None:
            return vis_map, vis_alpha

    def end_path(self):
        # Complete the drawing
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths.append(self.curr_path)
        self.history.append(self.drawn_map.copy())
        self.surplus_history = True

    def predict(self):
        self.out_prob = torch.from_numpy(self.drawn_map).float().cuda()
        self.out_prob, _ = pad_divide_by(self.out_prob, 16, self.out_prob.shape[-2:])
        self.out_mask = aggregate_sbg(self.out_prob, keep_bg=True)
        return self.out_mask

    def undo(self):
        if self.surplus_history:
            self.history.pop()
            self.surplus_history = False
        self.drawn_map = self.history.pop()
        # pop the current path (which is empty) and the last path
        self.all_paths = self.all_paths[:-2]
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths.append(self.curr_path)
        return self.predict()

    def can_undo(self):
        return (len(self.history) > 0) and not (
            self.surplus_history and (len(self.history) < 2))


class ScribbleInteraction(Interaction):
    def __init__(self, image, prev_mask, true_size, controller, num_objects):
        """
        prev_mask should be in an indexed form
        """
        super().__init__(image, prev_mask, true_size, controller)

        self.K = num_objects

        self.drawn_map = np.empty((self.h, self.w), dtype=np.uint8)
        self.drawn_map.fill(255)
        # background + k
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths = [self.curr_path]
        self.size = 3
        self.surplus_history = False

    """
    k - object id
    vis - a tuple (visualization map, pass through alpha). None if not needed.
    """
    def push_point(self, x, y, k, vis=None):
        if vis is not None:
            vis_map, vis_alpha = vis
        selected = self.curr_path[k]
        selected.append((x, y))
        if len(selected) >= 2:
            self.drawn_map = cv2.line(self.drawn_map, 
                (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                k, thickness=self.size)

            # Plot visualization
            if vis is not None:
                # Visualization for drawing
                if k == 0:
                    vis_map = cv2.line(vis_map, 
                        (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                        (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                        color_map[k], thickness=self.size)
                else:
                    vis_map = cv2.line(vis_map, 
                            (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                            (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                            color_map[k], thickness=self.size)
                # Visualization on/off boolean filter
                vis_alpha = cv2.line(vis_alpha, 
                        (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                        (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                        0.75, thickness=self.size)

        # Optional vis return
        if vis is not None:
            return vis_map, vis_alpha

    def end_path(self):
        # Complete the drawing
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths.append(self.curr_path)
        self.history.append(self.drawn_map.copy())
        self.surplus_history = True

    def predict(self):
        self.out_prob = self.controller.interact(self.image, self.prev_mask, self.drawn_map)
        self.out_mask = aggregate_wbg(self.out_prob, keep_bg=True, hard=True)
        return self.out_mask

    def undo(self):
        if self.surplus_history:
            self.history.pop()
            self.surplus_history = False
        self.drawn_map = self.history.pop()
        # pop the current path (which is empty) and the last path
        self.all_paths = self.all_paths[:-2]
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths.append(self.curr_path)
        return self.predict()

    def can_undo(self):
        return (len(self.history) > 0) and not (
            self.surplus_history and (len(self.history) < 2))


class ClickInteraction(Interaction):
    def __init__(self, image, prev_mask, true_size, controller, tar_obj, pad):
        """
        prev_mask in a prob. form
        """
        super().__init__(image, prev_mask, true_size, controller)
        self.tar_obj = tar_obj
        self.pad = pad

        # negative/positive for each object
        self.pos_clicks = []
        self.neg_clicks = []

        self.out_prob = self.prev_mask.clone()
        self.surplus_history = False

    """
    neg - Negative interaction or not
    vis - a tuple (visualization map, pass through alpha). None if not needed.
    """
    def push_point(self, x, y, neg, vis=None):
        # Clicks
        if neg:
            self.neg_clicks.append((x, y))
        else:
            self.pos_clicks.append((x, y))

        # Do the prediction, note that the image is padded
        self.obj_mask = self.controller.interact(self.image, x+self.pad[0], y+self.pad[2], not neg)
        self.history.append(deepcopy((self.pos_clicks, self.neg_clicks)))
        self.surplus_history = True

        # Plot visualization
        if vis is not None:
            vis_map, vis_alpha = vis
            # Visualization for clicks
            if neg:
                vis_map = cv2.circle(vis_map, 
                        (int(round(x)), int(round(y))),
                        2, color_map[0], thickness=-1)
            else:
                vis_map = cv2.circle(vis_map, 
                        (int(round(x)), int(round(y))),
                        2, color_map[self.tar_obj], thickness=-1)

            vis_alpha = cv2.circle(vis_alpha, 
                        (int(round(x)), int(round(y))),
                        2, 1, thickness=-1)

            # Optional vis return
            return vis_map, vis_alpha

    def predict(self):
        if self.obj_mask is None:
            self.out_prob = self.prev_mask.clone()
        else:
            self.out_prob[self.tar_obj-1] = self.obj_mask
        self.out_mask = aggregate_sbg(self.out_prob, keep_bg=True, hard=True)
        return self.out_mask

    def undo(self):
        if self.surplus_history:
            self.history.pop()
            self.surplus_history = False
        self.pos_clicks, self.neg_clicks = self.history.pop()
        self.obj_mask = self.controller.undo()
        return self.predict()

    def can_undo(self):
        return (len(self.history) > 0) and not (
            self.surplus_history and (len(self.history) < 2))