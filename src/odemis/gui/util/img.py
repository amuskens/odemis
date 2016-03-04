# -*- coding: utf-8 -*-
"""
Created on 10 Jan 2014

@author: Éric Piel

Copyright © 2014 Éric Piel, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License version 2 as published by the Free Software Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.

"""

# Some helper functions to convert/manipulate images (DataArray and wxImage)

from __future__ import division

import cairo
import logging
import math
import numpy
from odemis import model
from odemis.gui import BLEND_SCREEN, BLEND_DEFAULT
from odemis.gui.comp.overlay.base import Label
import odemis.model
from odemis.model._dataflow import DataArray
from odemis.util import units
from odemis.acq import stream
import time
import wx


BAR_PLOT_COLOUR = (0.75, 0.75, 0.75)
MAX_RES_FACTOR = 5  # upper limit resolution factor to exported image


# @profile
# TODO: rename to *_bgra_*
def format_rgba_darray(im_darray, alpha=None):
    """ Reshape the given numpy.ndarray from RGB to BGRA format

    If an alpha value is provided it will be set in the '4th' byte and used to scale the other RGB
    values within the array.

    """

    if im_darray.shape[-1] == 3:
        h, w, _ = im_darray.shape
        rgba_shape = (h, w, 4)
        rgba = numpy.empty(rgba_shape, dtype=numpy.uint8)
        # Copy the data over with bytes 0 and 2 being swapped (RGB becomes BGR through the -1)
        rgba[:, :, 0:3] = im_darray[:, :, ::-1]
        if alpha is not None:
            rgba[:, :, 3] = alpha
            rgba = scale_to_alpha(rgba)
        new_darray = odemis.model.DataArray(rgba)

        return new_darray

    elif im_darray.shape[-1] == 4:
        if hasattr(im_darray, 'metadata'):
            if im_darray.metadata.get('byteswapped', False):
                logging.warning("Trying to convert to BGRA an array already in BGRA")
                return im_darray

        rgba = numpy.empty(im_darray.shape, dtype=numpy.uint8)
        rgba[:, :, 0] = im_darray[:, :, 2]
        rgba[:, :, 1] = im_darray[:, :, 1]
        rgba[:, :, 2] = im_darray[:, :, 0]
        rgba[:, :, 3] = im_darray[:, :, 3]
        new_darray = odemis.model.DataArray(rgba)
        new_darray.metadata['byteswapped'] = True
        return new_darray
    else:
        raise ValueError("Unsupported colour depth!")


def min_type(data):
    """Find the minimum type code needed to represent the elements in `data`.
    """

    if numpy.issubdtype(data.dtype, numpy.integer):
        types = [numpy.int8, numpy.uint8, numpy.int16, numpy.uint16, numpy.int32,
                 numpy.uint32, numpy.int64, numpy.uint64]
    else:
        types = [numpy.float16, numpy.float32, numpy.float64]

    data_min, data_max = data.min(), data.max()

    for t in types:
        if numpy.all(data_min >= numpy.iinfo(t).min) and numpy.all(data_max <= numpy.iinfo(t).max):
            return t
    else:
        raise ValueError("Could not find suitable dtype.")


def apply_rotation(ctx, rotation, b_im_rect):
    """
    Applies rotation to the given cairo context

    ctx: (cairo.Context) Cairo context to draw on
    rotation: (float) in rads
    b_im_rect: (float, float, float, float) top, left, width, height rectangle
        containing the image in buffer coordinates
    """
    if rotation is not None and abs(rotation) >= 0.008:  # > 0.5°
        x, y, w, h = b_im_rect

        rot_x = x + w / 2
        rot_y = y + h / 2
        # Translate to the center of the image (in buffer coordinates)
        ctx.translate(rot_x, rot_y)
        # Rotate
        ctx.rotate(-rotation)
        # Translate back, so the origin is at the top left position of the image
        ctx.translate(-rot_x, -rot_y)


def apply_shear(ctx, shear, b_im_rect):
    """
    Applies shear to the given cairo context

    ctx: (cairo.Context) Cairo context to draw on
    shear: (float) shear to be applied
    b_im_rect: (float, float, float, float) top, left, width, height rectangle
        containing the image in buffer coordinates
    """
    # Shear if needed
    if shear is not None and abs(shear) >= 0.0005:
        # Shear around the center of the image data. Shearing only occurs on the x axis
        x, y, w, h = b_im_rect
        shear_x = x + w / 2
        shear_y = y + h / 2

        # Translate to the center x of the image (in buffer coordinates)
        ctx.translate(shear_x, shear_y)
        shear_matrix = cairo.Matrix(1.0, shear, 0.0, 1.0)
        ctx.transform(shear_matrix)
        ctx.translate(-shear_x, -shear_y)


def apply_flip(ctx, flip, b_im_rect):
    """
    Applies flip to the given cairo context

    ctx: (cairo.Context) Cairo context to draw on
    flip: (boolean) apply flip if True
    b_im_rect: (float, float, float, float) top, left, width, height rectangle
        containing the image in buffer coordinates
    """
    if flip:
        fx = fy = 1.0

        if flip & wx.HORIZONTAL == wx.HORIZONTAL:
            fx = -1.0

        if flip & wx.VERTICAL == wx.VERTICAL:
            fy = -1.0

        x, y, w, h = b_im_rect

        flip_x = x + w / 2
        flip_y = y + h / 2

        flip_matrix = cairo.Matrix(fx, 0.0, 0.0, fy)

        ctx.translate(flip_x, flip_y)

        ctx.transform(flip_matrix)
        ctx.translate(-flip_x, -flip_y)


def fit_to_content(images, client_size):
    """
    Adapt the scale to fit to the current content

    images (list of model.DataArray)
    client_size (wx._core.Size)

    returns (tuple of floats): scale to fit
    """

    # Find bounding box of all the content
    bbox = [None, None, None, None]  # ltrb in wu
    for im in images:
        if im is None:
            continue
        im_scale = im.metadata['dc_scale']
        w, h = im.shape[1] * im_scale[0], im.shape[0] * im_scale[1]
        c = im.metadata['dc_center']
        bbox_im = [c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2]
        if bbox[0] is None:
            bbox = bbox_im
        else:
            bbox = (min(bbox[0], bbox_im[0]), min(bbox[1], bbox_im[1]),
                    max(bbox[2], bbox_im[2]), max(bbox[3], bbox_im[3]))

    if bbox[0] is None:
        return  # no image => nothing to do

    # compute mpp so that the bbox fits exactly the visible part
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]  # wu
    if w == 0 or h == 0:
        logging.warning("Weird image size of %fx%f wu", w, h)
        return  # no image
    cw = max(1, client_size[0])  # px
    ch = max(1, client_size[1])  # px
    scale = min(ch / h, cw / w)  # pick the dimension which is shortest

    return scale


def ar_create_tick_labels(client_size, ticksize, num_ticks, tau):
    """
    Create list of tick labels for AR polar representation

    client_size (wx._core.Size)
    ticksize (int): size of tick in pixels
    num_ticks (int): number of ticks
    returns (list of Labels)
            (tuple of floats): center
            (float): inner radius
            (float): radius
            (float): tau
    """

    # Calculate the characteristic values
    center_x = client_size.x / 2
    center_y = client_size.y / 2
    inner_radius = min(center_x, center_y)
    radius = inner_radius + (ticksize / 1.5)
    ticks = []

    # Top middle
    for i in range(num_ticks):
        # phi needs to be rotated 90 degrees counter clockwise, otherwise
        # 0 degrees will be at the right side of the circle
        phi = (tau / num_ticks * i) - (math.pi / 2)
        deg = round(math.degrees(phi))

        cos = math.cos(phi)
        sin = math.sin(phi)

        # Tick start and end point (outer and inner)
        ox = center_x + radius * cos
        oy = center_y + radius * sin
        ix = center_x + (radius - ticksize) * cos
        iy = center_y + (radius - ticksize) * sin

        # Tick label positions
        lx = center_x + (radius + 5) * cos
        ly = center_y + (radius + 5) * sin

        label = Label(
            text=u"%d°" % (deg + 90),
            pos=(lx, ly),
            font_size=12,
            flip=True,
            align=wx.ALIGN_CENTRE_HORIZONTAL | wx.ALIGN_BOTTOM,
            colour=(0.8, 0.8, 0.8),
            opacity=1.0,
            deg=deg - 90
        )

        ticks.append((ox, oy, ix, iy, label))
    return ticks, (center_x, center_y), inner_radius, radius


def write_label(ctx, l, font_name, canvas_padding=None, view_width=None, view_height=None):
    """
    Draws label to given context

    ctx: (cairo.Context) Cairo context to draw on
    l: (Label) label to draw
    font_name (string): font name
    canvas_padding (int): canvas padding if exists
    view_width (int): window view width
    view_height (int): window view height
    """

    # No text? Do nothing
    if not l.text:
        return

    # Cache the current context settings
    ctx.save()

    # TODO: Look at ScaledFont for additional caching
    ctx.select_font_face(font_name, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)

    # For some reason, fonts look a little bit smaller when Cairo
    # plots them at an angle. We compensate for that by increasing the size
    # by 1 point in that case, so the size visually resembles that of
    # straight text.
    if l.deg not in (0.0, 180.0, None):
        ctx.set_font_size(l.font_size + 1)
    else:
        ctx.set_font_size(l.font_size)

    # Rotation always happens at the plot coordinates
    if l.deg is not None:
        phi = math.radians(l.deg)
        rx, ry = l.pos

        if l.flip:
            phi -= math.pi

        ctx.translate(rx, ry)
        ctx.rotate(phi)
        ctx.translate(-rx, -ry)

    # Take care of newline characters
    parts = l.text.split("\n")

    # Calculate the rendering position
    if not l.render_pos:
        x, y = l.pos

        lw, lh = 0, 0
        plh = l.font_size  # default to font size, but should always get updated
        for p in parts:
            plw, plh = ctx.text_extents(p)[2:4]
            lw = max(lw, plw)
            lh += plh

        # Cairo renders text from the bottom left, but we want to treat
        # the top left as the origin. So we need to add the height (lower the
        # render point), to make the given position align with the top left.
        y += plh

        if canvas_padding is not None:
            # Apply padding
            x = max(min(x, view_width - canvas_padding), canvas_padding)
            y = max(min(y, view_height - canvas_padding), canvas_padding)

        # Horizontally align the label
        if l.align & wx.ALIGN_RIGHT:
            x -= lw
        elif l.align & wx.ALIGN_CENTRE_HORIZONTAL:
            x -= lw / 2.0

        # Vertically align the label
        if l.align & wx.ALIGN_BOTTOM:
            y -= lh
        elif l.align & wx.ALIGN_CENTER_VERTICAL:
            y -= lh / 2.0

        # When we rotate text, flip gets a different meaning
        if l.deg is None and l.flip:
            if canvas_padding is not None:
                width = view_width
                height = view_height

                # Prevent the text from running off screen
                if x + lw + canvas_padding > width:
                    x = width - lw
                elif x < canvas_padding:
                    x = canvas_padding
                if y + lh + canvas_padding > height:
                    y = height - lh
                elif y < lh:
                    y = lh

        l.render_pos = x, y
        l.text_size = lw, lh
    else:
        x, y = l.render_pos
        lw, lh = l.text_size

    # Draw Shadow
    if l.colour:
        ctx.set_source_rgba(0.0, 0.0, 0.0, 0.7 * l.opacity)
        ofst = 0
        for part in parts:
            ctx.move_to(x + 1, y + 1 + ofst)
            ofst += l.font_size
            ctx.show_text(part)

    # Draw Text
    if l.colour:
        if len(l.colour) == 3:
            ctx.set_source_rgba(*(l.colour + (l.opacity,)))
        else:
            ctx.set_source_rgba(*l.colour)

    ofst = 0
    for part in parts:
        ctx.move_to(x, y + ofst)
        ofst += l.font_size + 1
        ctx.show_text(part)

    ctx.restore()


def draw_ar_frame(ctx, client_size, ticks, font_name, center_x, center_y, inner_radius, radius, tau):
    """
    Draws AR frame on the given context

    ctx (cairo.Context): Cairo context to draw on
    client_size (wx._core.Size): client window size
    ticks (list of Labels): list of tick labels to draw
    font_name (string): font name
    center_x (float): center x axis
    center_y (float): center y axis
    inner_radius (float): inner radius
    radius (float): radius
    tau (float): tau
    """
    # Draw frame that covers everything outside the center circle
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.set_source_rgb(0.2, 0.2, 0.2)

    ctx.rectangle(0, 0, client_size.x, client_size.y)
    ctx.arc(center_x, center_y, inner_radius, 0, tau)
    ctx.fill()

    # Draw Azimuth degree circle
    ctx.set_line_width(2)
    ctx.set_source_rgb(0.5, 0.5, 0.5)
    ctx.arc(center_x, center_y, radius, 0, tau)
    ctx.stroke()

    # Draw Azimuth degree ticks
    ctx.set_line_width(1)
    for sx, sy, lx, ly, _ in ticks:
        ctx.move_to(sx, sy)
        ctx.line_to(lx, ly)
    ctx.stroke()

    # Draw tick labels
    for _, _, _, _, label in ticks:
        write_label(ctx, label, font_name)


def set_images(im_args):
    """ Set (or update) image

    im_args: (list of tuples): Each element is either None or
        (im, w_pos, scale, keepalpha, rotation, name, blend_mode)

        0. im (wx.Image): the image
        1. w_pos (2-tuple of float): position of the center of the image (in world units)
        2. scale (float, float): scale of the image
        3. keepalpha (boolean): whether the alpha channel must be used to draw
        4. rotation (float): clockwise rotation in radians on the center of the image
        5. shear (float): horizontal shear relative to the center of the image
        6. flip (int): Image horz or vert flipping. 0 for no flip, wx.HORZ and wx.VERT otherwise
        7. blend_mode (int): blend mode to use for the image. Defaults to `source` which
                just overrides underlying layers.
        8. name (str): name of the stream that the image originated from
        9. date (int): seconds since epoch
        10. stream (object): just needed to identify the image in case of dublicated name

    returns (list of wx.Image)
    """

    images = []

    for args in im_args:
        if args is None:
            images.append(None)
        else:
            im, w_pos, scale, keepalpha, rotation, shear, flip, blend_mode, name, date, stream = args

            if not blend_mode:
                blend_mode = BLEND_DEFAULT

            try:
                depth = im.shape[2]

                if depth == 3:
                    im = add_alpha_byte(im)
                elif depth != 4:  # Both ARGB32 and RGB24 need 4 bytes
                    raise ValueError("Unsupported colour byte size (%s)!" % depth)
            except IndexError:
                # Handle grayscale images pretending they are rgb
                pass

            im.metadata['dc_center'] = w_pos
            im.metadata['dc_scale'] = scale
            im.metadata['dc_rotation'] = rotation
            im.metadata['dc_shear'] = shear
            im.metadata['dc_flip'] = flip
            im.metadata['dc_keepalpha'] = keepalpha
            im.metadata['blend_mode'] = blend_mode
            im.metadata['name'] = name
            im.metadata['date'] = date
            im.metadata['stream'] = stream

            images.append(im)

    return images


def calc_img_buffer_rect(im_data, im_scale, w_im_center, buffer_center, buffer_scale, buffer_size):
    """ Compute the rectangle containing the image in buffer coordinates

    The (top, left) value are relative to the 0,0 top left of the buffer.

    im_data (DataArray): image data
    im_scale (float, float): The x and y scales of the image
    w_im_center (float, float): The center of the image in world coordinates
    buffer_center (float, float): The buffer center
    buffer_scale (float, float): The buffer scale
    buffer_size (float, float): The buffer size

    returns (float, float, float, float) top, left, width, height

    """

    # Scale the image
    im_h, im_w = im_data.shape[:2]
    scale_x, scale_y = im_scale
    scaled_im_size = (im_w * scale_x, im_h * scale_y)

    # Calculate the top left
    w_topleft = (w_im_center[0] - (scaled_im_size[0] / 2),
                 w_im_center[1] - (scaled_im_size[1] / 2))

    b_topleft = (round(((w_topleft[0] - buffer_center[0]) / buffer_scale[0]) + (buffer_size[0] / 2)),
                 round(((w_topleft[1] + buffer_center[1]) / buffer_scale[1]) + (buffer_size[1] / 2)))

    final_size = (scaled_im_size[0] / buffer_scale[0], scaled_im_size[1] / buffer_scale[1])
    return b_topleft + final_size


def draw_image(ctx, im_data, w_im_center, buffer_center, buffer_scale,
               buffer_size, opacity=1.0, im_scale=(1.0, 1.0), rotation=None,
               shear=None, flip=None, blend_mode=BLEND_DEFAULT, interpolate_data=True):
    """ Draw the given image to the Cairo context

    The buffer is considered to have it's 0,0 origin at the top left

    ctx (cairo.Context): Cario context to draw on
    im_data (DataArray): Image to draw
    w_im_center (2-tuple float)
    buffer_center (float, float): The buffer center
    buffer_scale (float, float): The buffer scale
    buffer_size (float, float): The buffer size
    opacity (float) [0..1] => [transparent..opaque]
    im_scale (float, float)
    rotation (float): Clock-wise rotation around the image center in radians
    shear (float): Horizontal shearing of the image data (around it's center)
    flip (wx.HORIZONTAL | wx.VERTICAL): If and how to flip the image
    blend_mode (int): Graphical blending type used for transparency

    """

    # Fully transparent image does not need to be drawn
    if opacity < 1e-8:
        logging.debug("Skipping draw: image fully transparent")
        return

    # Determine the rectangle the image would occupy in the buffer
    b_im_rect = calc_img_buffer_rect(im_data, im_scale, w_im_center, buffer_center, buffer_scale, buffer_size)

    # print b_im_rect
    x, y, w, h = b_im_rect
    # Rotate if needed
    ctx.save()

    # apply transformations if needed
    apply_rotation(ctx, rotation, b_im_rect)
    apply_shear(ctx, shear, b_im_rect)
    apply_flip(ctx, flip, b_im_rect)

    width_ratio = float(im_scale[0]) / float(buffer_scale[0])
    height_ratio = float(im_scale[1]) / float(buffer_scale[1])
    intersection = (0, 0, buffer_size[0], buffer_size[1])
    total_scale = total_scale_x, total_scale_y = (width_ratio, height_ratio)

    # in case of small floating errors
    if abs(total_scale_x - 1) < 1e-8 or abs(total_scale_y - 1) < 1e-8:
        total_scale = (1.0, 1.0)

    if total_scale_x > 1.0 or total_scale_y > .0:
        # logging.debug("Up scaling required")

        # If very little data is trimmed, it's better to scale the entire image than to create
        # a slightly smaller copy first.
        if b_im_rect[2] > intersection[2] * 1.1 or b_im_rect[3] > intersection[3] * 1.1:
            im_data, tl = get_sub_img(intersection, b_im_rect, im_data, total_scale)
            b_im_rect = (tl[0], tl[1], b_im_rect[2], b_im_rect[3],)
            x, y, _, _ = b_im_rect

    if im_data.metadata.get('dc_keepalpha', True):
        im_format = cairo.FORMAT_ARGB32
    else:
        im_format = cairo.FORMAT_RGB24

    height, width, _ = im_data.shape

    # Note: Stride calculation is done automatically when no stride parameter is provided.
    stride = cairo.ImageSurface.format_stride_for_width(im_format, width)

    imgsurface = cairo.ImageSurface.create_for_data(im_data, im_format, width, height, stride)

    # In Cairo a pattern is the 'paint' that it uses to draw
    surfpat = cairo.SurfacePattern(imgsurface)
    # Set the filter, so we get best quality but slow scaling
    # In opposition to the GUI gallery tab, here we care more about the
    # quality of the exported image than being fast.
    if interpolate_data:
        surfpat.set_filter(cairo.FILTER_BEST)
    else:
        # In case of "raw" export try to maintain the original data
        surfpat.set_filter(cairo.FILTER_NEAREST)

    ctx.translate(x, y)
    ctx.scale(total_scale_x, total_scale_y)

    ctx.set_source(surfpat)
    ctx.set_operator(blend_mode)

    if opacity < 1.0:
        ctx.paint_with_alpha(opacity)
    else:
        ctx.paint()

    # Restore the cached transformation matrix
    ctx.restore()


def ar_to_export_data(streams, client_size, raw=False):
    """
    Creates either raw or WYSIWYG representation for the AR projection

    streams (list of Stream objects): streams displayed in the current view
    client_size (wx._core.Size)
    raw (boolean): if True returns raw representation

    returns (model.DataArray)
    """

    if raw:
        # TODO implement raw export
        raise ValueError("Raw export is unsupported for AR data")
    else:
        # we expect just one stream
        wim = format_rgba_darray(streams[0].image.value)
        # image is always centered, fitting the whole canvass
        images = set_images([(wim, (0, 0), (1, 1), False, None, None, None, None, streams[0].name.value, None, None)])
        scale = fit_to_content(images, client_size)

        # Make surface based on the maximum resolution
        data_to_draw = numpy.zeros((client_size.y, client_size.x, 4), dtype=numpy.uint8)
        surface = cairo.ImageSurface.create_for_data(
            data_to_draw, cairo.FORMAT_ARGB32, client_size.x, client_size.y)
        ctx = cairo.Context(surface)

        im = images[0]
        buffer_center = (0, 0)
        buffer_scale = (im.metadata['dc_scale'][0] / scale,
                        im.metadata['dc_scale'][1] / scale)
        buffer_size = client_size.x, client_size.y

        draw_image(
            ctx,
            im,
            im.metadata['dc_center'],
            buffer_center,
            buffer_scale,
            buffer_size,
            1.0,
            im_scale=im.metadata['dc_scale'],
            rotation=im.metadata['dc_rotation'],
            shear=im.metadata['dc_shear'],
            flip=im.metadata['dc_flip'],
            blend_mode=im.metadata['blend_mode'],
            interpolate_data=True
        )

        font_name = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT).GetFaceName()
        tau = 2 * math.pi
        ticksize = 10
        num_ticks = 6
        ticks_info = ar_create_tick_labels(client_size, ticksize, num_ticks, tau)
        ticks, (center_x, center_y), inner_radius, radius = ticks_info
        draw_ar_frame(ctx, client_size, ticks, font_name, center_x, center_y, inner_radius, radius, tau)
        ar_plot = model.DataArray(data_to_draw)
        ar_plot.metadata[model.MD_DIMS] = 'YXC'
        return ar_plot


def value_to_pixel(value, pixel_space, vtp_ratio, value_range, orientation):
    """
    Map range value to legend pixel position

    value (float): value to map
    pixel_space (int): pixel space
    vtp_ratio (float):  value to pixel ratio
    value_range (tuple of floats): value range
    orientation (int): legend orientation

    returns (float): pixel position
    """
    if pixel_space is None:
        return None
    elif None not in (vtp_ratio, value_range):
        pixel = (value - value_range[0]) * vtp_ratio
        pixel = int(round(pixel))
    else:
        pixel = 0
    return pixel if orientation == wx.HORIZONTAL else pixel_space - pixel


def calculate_ticks(value_range, client_size, orientation, tick_spacing):
    """
    Calculate which values in the range to represent as ticks on the axis

    value_range (tuple of floats): value range
    client_size (wx._core.Size)
    orientation (int): legend orientation
    tick_spacing (float): space between ticks

    returns (list of tuples of floats): list of pixel position and value pairs
            (float): value to pixel ratio

    """

    if value_range is None:
        return

    min_val, max_val = value_range

    # Get the horizontal/vertical space in pixels
    pixel_space = client_size[orientation != wx.HORIZONTAL]

    if orientation == wx.HORIZONTAL:
        min_pixel = 0
    else:
        # Don't display ticks too close from the left border
        min_pixel = 10

    # Range width
    value_space = max_val - min_val
    if value_space == 0:
        logging.info("Trying to compute legend tick with empty range %s", value_range)
        vtp_ratio = None
        # Just one tick, at the origin
        pixel = max(min_pixel, value_to_pixel(min_val, pixel_space, vtp_ratio,
                                              value_range, orientation))
        tick_list = [(pixel, min_val)]
        return tick_list

    vtp_ratio = pixel_space / value_space

    num_ticks = pixel_space // tick_spacing
    # Calculate the best step size in powers of 10, so it will cover at
    # least the distance `val_dist`
    value_step = 1e-12

    # Increase the value step tenfold while it fits more than num_ticks times
    # in the range
    while value_step and value_space / value_step > num_ticks:
        value_step *= 10
    # logging.debug("Value step is %s after first iteration with range %s",
    #               value_step, value_space)

    # Divide the value step by two,
    while value_step and value_space / value_step < num_ticks:
        value_step /= 2
    # logging.debug("Value step is %s after second iteration with range %s",
    #               value_step, value_space)

    first_val = (int(min_val / value_step) + 1) * value_step if value_step else 0
    # logging.debug("Setting first tick at value %s", first_val)

    tick_values = [min_val]
    cur_val = first_val

    while cur_val < max_val:
        tick_values.append(cur_val)
        cur_val += value_step

    ticks = []
    for tick_value in tick_values:
        pixel = value_to_pixel(tick_value, pixel_space, vtp_ratio, value_range,
                               orientation)
        pix_val = (pixel, tick_value)
        if pix_val not in ticks:
            if min_pixel <= pixel <= pixel_space:
                ticks.append(pix_val)

    tick_list = ticks

    return tick_list, vtp_ratio


def draw_scale(ctx, value_range, client_size, orientation, tick_spacing, fill_colour, unit, scale_width):
    """
    Draws horizontal and vertical scale bars

    value_range (tuple of floats): value range
    client_size (wx._core.Size)
    orientation (int): legend orientation
    tick_spacing (float): space between ticks
    fill_colour (tuple of floats): colour to fill bars
    unit (string): scale unit
    scale_width: scale bar width
    """

    if value_range is None:
        return

    tick_list, _ = calculate_ticks(value_range, client_size, orientation, tick_spacing)
    csize = client_size

    # Set Font
    font = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)

    ctx.select_font_face(font.GetFaceName(), cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(font.GetPointSize())

    ctx.set_source_rgb(*fill_colour)
    ctx.set_line_width(2)
    ctx.set_line_join(cairo.LINE_JOIN_MITER)

    max_width = 0
    prev_lpos = 0 if orientation == wx.HORIZONTAL else csize.y

    for i, (pos, val) in enumerate(tick_list):
        label = units.readable_str(val, unit, 3)
        _, _, lbl_width, lbl_height, _, _ = ctx.text_extents(label)

        if orientation == wx.HORIZONTAL:
            lpos = pos - (lbl_width // 2)
            lpos = max(min(lpos, csize.x - lbl_width - 2), 2)
            # print (i, prev_right, lpos)
            if prev_lpos < lpos:
                ctx.move_to(lpos, lbl_height + 8)
                ctx.show_text(label)
                ctx.move_to(pos, 5)
                ctx.line_to(pos, 0)
            prev_lpos = lpos + lbl_width
        else:
            max_width = max(max_width, lbl_width)
            lpos = pos + (lbl_height // 2)
            lpos = max(min(lpos, csize.y), 2)

            if prev_lpos >= lpos + 20 or i == 0:
                ctx.move_to(scale_width - lbl_width - 9, lpos)
                ctx.show_text(label)
                ctx.move_to(scale_width - 5, pos)
                ctx.line_to(scale_width, pos)
            prev_lpos = lpos + lbl_height

        ctx.stroke()


def val_x_to_pos_x(val_x, client_size, data_width=None, range_x=None, data_prop=None):
    """ Translate an x value to an x position in pixels
    The minimum x value is considered to be pixel 0 and the maximum is the canvas width. The
    parameter will be clipped if it's out of range.
    val_x (float): The value to map
    client_size (wx._core.Size)
    data_prop (int, int, int, int)
    returns (float)
    """
    range_x = range_x or data_prop[1]
    data_width = data_width or data_prop[0]

    if data_width:
        # Clip val_x
        x = min(max(range_x[0], val_x), range_x[1])
        perc_x = (x - range_x[0]) / data_width
        return perc_x * client_size.x
    else:
        return 0


def val_y_to_pos_y(val_y, client_size, data_height=None, range_y=None, data_prop=None):
    """ Translate an y value to an y position in pixels
    The minimum y value is considered to be pixel 0 and the maximum is the canvas width. The
    parameter will be clipped if it's out of range.
    val_y (float): The value to map
    client_size (wx._core.Size)
    data_prop (int, int, int, int)
    returns (float)
    """
    range_y = range_y or data_prop[3]
    data_height = data_height or data_prop[2]

    if data_height:
        y = min(max(range_y[0], val_y), range_y[1])
        perc_y = (range_y[1] - y) / data_height
        return perc_y * client_size.y
    else:
        return 0


def bar_plot(ctx, data, data_width, range_x, data_height, range_y, client_size, fill_colour):
    """ Do a bar plot of the current `_data` """

    if len(data) < 2:
        return

    vx_to_px = val_x_to_pos_x
    vy_to_py = val_y_to_pos_y

    line_to = ctx.line_to
    ctx.set_source_rgb(*fill_colour)

    diff = (data[1][0] - data[0][0]) / 2.0
    px = vx_to_px(data[0][0] - diff, client_size, data_width, range_x)
    py = vy_to_py(0, client_size, data_height, range_y)

    ctx.move_to(px, py)
    # print "-", px, py

    for i, (vx, vy) in enumerate(data[:-1]):
        py = vy_to_py(vy, client_size, data_height, range_y)
        # print "-", px, py
        line_to(px, py)
        px = vx_to_px((data[i + 1][0] + vx) / 2.0, client_size, data_width, range_x)
        # print "-", px, py
        line_to(px, py)

    py = vy_to_py(data[-1][1], client_size, data_height, range_y)
    # print "-", px, py
    line_to(px, py)

    diff = (data[-1][0] - data[-2][0]) / 2.0
    px = vx_to_px(data[-1][0] + diff, client_size, data_width, range_x)
    # print "-", px, py
    line_to(px, py)

    py = vy_to_py(0, client_size, data_height, range_y)
    # print "-", px, py
    line_to(px, py)

    ctx.close_path()
    ctx.fill()


def spectrum_to_export_data(spectrum, client_size, raw, unit, spectrum_range):
    """
    Creates either raw or WYSIWYG representation for the spectrum data plot

    spectrum (list of float): spectrum values
    client_size (wx._core.Size)
    raw (boolean): if True returns raw representation
    unit (string): wavelength unit
    spectrum_range (list of float): spectrum range

    returns (model.DataArray)
    """

    if raw:
        return spectrum
    else:
        # Draw spectrumbar plot
        data = zip(spectrum_range, spectrum)
        fill_colour = BAR_PLOT_COLOUR
        data_to_draw = numpy.zeros((client_size.y, client_size.x, 4), dtype=numpy.uint8)
        surface = cairo.ImageSurface.create_for_data(
            data_to_draw, cairo.FORMAT_ARGB32, client_size.x, client_size.y)
        ctx = cairo.Context(surface)
        # calculate data characteristics
        horz, vert = zip(*data)
        min_x = min(horz)
        max_x = max(horz)
        min_y = min(vert)
        max_y = max(vert)
        range_x = (min_x, max_x)
        data_width = max_x - min_x
        range_y = (min_y, max_y)
        data_height = max_y - min_y
        bar_plot(ctx, data, data_width, range_x, data_height, range_y, client_size, fill_colour)

        # Draw horizontal scale legend
        value_range = (spectrum_range[0], spectrum_range[-1])
        orientation = wx.HORIZONTAL
        tick_spacing = 120
        scale_width = 40
        scale_height = 30
        scale_x_draw = numpy.zeros((scale_height, client_size.x, 4), dtype=numpy.uint8)
        scale_x_draw.fill(25)
        surface = cairo.ImageSurface.create_for_data(
            scale_x_draw, cairo.FORMAT_ARGB32, client_size.x, scale_height)
        ctx = cairo.Context(surface)
        draw_scale(ctx, value_range, client_size, orientation, tick_spacing, fill_colour, unit, scale_width)
        data_with_legend = numpy.append(data_to_draw, scale_x_draw, axis=0)

        # Draw vertical scale legend
        orientation = wx.VERTICAL
        tick_spacing = 80
        value_range = (min(spectrum), max(spectrum))
        unit = None
        scale_y_draw = numpy.zeros((client_size.y, scale_width, 4), dtype=numpy.uint8)
        scale_y_draw.fill(25)
        surface = cairo.ImageSurface.create_for_data(
            scale_y_draw, cairo.FORMAT_ARGB32, scale_width, client_size.y)
        ctx = cairo.Context(surface)
        draw_scale(ctx, value_range, client_size, orientation, tick_spacing, fill_colour, unit, scale_width)

        # Extend y scale bar to fit the height of the bar plot with the x
        # scale bar attached
        extend = numpy.empty((scale_height, scale_width, 4), dtype=numpy.uint8)
        extend.fill(25)
        scale_y_draw = numpy.append(scale_y_draw, extend, axis=0)
        data_with_legend = numpy.append(scale_y_draw, data_with_legend, axis=1)

        spec_plot = model.DataArray(data_with_legend)
        spec_plot.metadata[model.MD_DIMS] = 'YXC'
        return spec_plot


def draw_export_legend(legend_ctx, images, buffer_size, mpp, mag=None, hfw=None,
                       scale_bar_width=None, scale_actual_width=None, date=None, streams_data=None, stream=None):
    """
    Draws legend to be attached to the exported image
    """
    init_x_pos = buffer_size[0] // 25
    upper_part = 0.25
    middle_part = 0.5
    lower_part = 0.85
    large_font = buffer_size[0] // 60  # used for general data
    small_font = buffer_size[0] // 80  # used for stream data
    n = len(images)
    # Just make cell dimensions analog to the image buffer dimensions
    big_cell_height = buffer_size[1] // 12
    small_cell_height = buffer_size[1] // 24
    cell_x_step = buffer_size[0] // 5
    legend_ctx.set_source_rgb(0, 0, 0)
    legend_ctx.rectangle(0, 0, buffer_size[0], n * small_cell_height + big_cell_height)
    legend_ctx.fill()
    legend_ctx.set_source_rgb(1, 1, 1)
    legend_ctx.set_line_width(2)

    # draw separation lines
    legend_y_pos = big_cell_height
    legend_ctx.move_to(0, legend_y_pos)
    legend_ctx.line_to(buffer_size[0], legend_y_pos)
    legend_ctx.stroke()
    for i in range(n - 1):
        legend_y_pos += small_cell_height
        legend_ctx.move_to(0, legend_y_pos)
        legend_ctx.line_to(buffer_size[0], legend_y_pos)
        legend_ctx.stroke()

    # write Magnification
    # TODO: Don't rely on a Microsoft font, just use DejaVu or something basic
    legend_ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL)
    legend_ctx.set_font_size(large_font)
    legend_x_pos = init_x_pos
    legend_y_pos = middle_part * big_cell_height
    legend_ctx.move_to(legend_x_pos, legend_y_pos)
    mag_text = u"Mag: × %s" % units.readable_str(units.round_significant(mag, 3))
#         if n == 1:
#             mag_dig = images[0].metadata['dc_scale'][0] / mpp
#             label = mag_text + u" (Digital: × %s)" % units.readable_str(units.round_significant(mag_dig, 2))
#         else:
    label = mag_text
    legend_ctx.show_text(label)

    # write HFW
    legend_x_pos += cell_x_step
    legend_ctx.move_to(legend_x_pos, legend_y_pos)
    hfw = units.round_significant(hfw, 4)
    label = u"HFW: %s" % units.readable_str(hfw, "m", sig=3)
    legend_ctx.show_text(label)

    # Draw scale bar
    legend_x_pos += cell_x_step
    legend_y_pos = upper_part * big_cell_height
    legend_ctx.move_to(legend_x_pos, legend_y_pos)
    legend_y_pos = middle_part * big_cell_height
    legend_ctx.line_to(legend_x_pos, legend_y_pos)
    legend_y_pos = lower_part * big_cell_height
    legend_ctx.move_to(legend_x_pos, legend_y_pos)
    label = units.readable_str(scale_actual_width, "m", sig=2)
    legend_ctx.show_text(label)
    legend_y_pos = middle_part * big_cell_height
    legend_ctx.move_to(legend_x_pos, legend_y_pos)
    legend_x_pos += scale_bar_width
    legend_ctx.line_to(legend_x_pos, legend_y_pos)
    legend_y_pos = upper_part * big_cell_height
    legend_ctx.line_to(legend_x_pos, legend_y_pos)
    legend_ctx.stroke()
    legend_x_pos += buffer_size[0] // 20
    legend_y_pos = middle_part * big_cell_height
    legend_ctx.move_to(legend_x_pos, legend_y_pos)

    # write acquisition date
    if date is not None:
        label = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(date))
        legend_ctx.show_text(label)

    # write stream data
    legend_y_pos = 0.75 * big_cell_height
    for s, data in streams_data.iteritems():
        legend_ctx.set_font_size(small_font)
        if s == stream:
            # in case of multifile, spot this particular stream with a
            # circle next to the stream name
            legend_x_pos = init_x_pos / 2
            legend_y_pos += small_cell_height
            legend_ctx.move_to(legend_x_pos, legend_y_pos)
            legend_ctx.arc(legend_x_pos, legend_y_pos, 10, 0, 2 * math.pi)
            legend_ctx.fill()
            legend_ctx.stroke()
            legend_x_pos = init_x_pos
            legend_ctx.move_to(legend_x_pos, legend_y_pos)
        else:
            legend_x_pos = init_x_pos
            legend_y_pos += small_cell_height
            legend_ctx.move_to(legend_x_pos, legend_y_pos)
        legend_ctx.show_text(s.name.value)
        legend_ctx.set_font_size(small_font)
        legend_x_pos += cell_x_step
        legend_y_pos_store = legend_y_pos
        for i, d in enumerate(data[:-1]):
            legend_ctx.move_to(legend_x_pos, legend_y_pos)
            legend_ctx.show_text(d)
            if (i % 2 == 1):
                legend_x_pos += cell_x_step
                legend_y_pos -= 0.4 * small_cell_height
            else:
                legend_y_pos += 0.4 * small_cell_height
        legend_y_pos = legend_y_pos_store


def get_ordered_images(streams, rgb=True):
    """ Return the list of images to display, ordered bottom to top (=last to draw)

    The last image of the list will have the merge ratio applied (as opacity)

    """

    images_opt = []
    images_spc = []
    images_std = []
    streams_data = {}

    im_min_type = numpy.uint8
    for s in streams:
        if not s:
            # should not happen, but let's not completely fail on this
            logging.error("StreamTree has a None stream")
            continue

        if not hasattr(s, "image") or s.image.value is None:
            continue

        # FluoStreams are merged using the "Screen" method that handles colour
        # merging without decreasing the intensity.
        data_raw = s.raw[0]
        if rgb:
            data = s.image.value
        else:
            # Pretend to be rgb
            if numpy.can_cast(im_min_type, min_type(data_raw)):
                im_min_type = min_type(data_raw)

            # Split the bits in R,G,B,A
            data = model.DataArray(numpy.zeros((data_raw.shape[0], data_raw.shape[1], 4), dtype=numpy.uint8),
                                   data_raw.metadata)
            data[:, :, 0] = numpy.right_shift(data_raw[:, :], 8) & 255
            data[:, :, 1] = data_raw[:, :] & 255
            data[:, :, 2] = numpy.right_shift(data_raw[:, :], 16) & 255
            data[:, :, 3] = numpy.right_shift(data_raw[:, :], 24) & 255

        if isinstance(s, stream.OpticalStream):
            images_opt.append((data, BLEND_SCREEN, s))
        elif isinstance(s, (stream.SpectrumStream, stream.CLStream)):
            images_spc.append((data, BLEND_DEFAULT, s))
        else:
            images_std.append((data, BLEND_DEFAULT, s))

        # metadata useful for the legend
        stream_data = []
        if data_raw.metadata.get(model.MD_EXP_TIME, None):
            stream_data.append(u"Exp. time: %s" % units.readable_str(data_raw.metadata[model.MD_EXP_TIME], "s", sig=3))
        if data_raw.metadata.get(model.MD_LIGHT_POWER, None):
            stream_data.append(units.readable_str(data_raw.metadata[model.MD_LIGHT_POWER], "W", sig=3))
        if data_raw.metadata.get(model.MD_EBEAM_VOLTAGE, None):
            stream_data.append(units.readable_str(data_raw.metadata[model.MD_EBEAM_VOLTAGE], "V", sig=3))
        if data_raw.metadata.get(model.MD_EBEAM_CURRENT, None):
            stream_data.append(units.readable_str(data_raw.metadata[model.MD_EBEAM_CURRENT], "A", sig=3))
        if data_raw.metadata.get(model.MD_DWELL_TIME, None):
            stream_data.append(u"dwelltime: %s" % units.readable_str(data_raw.metadata[model.MD_DWELL_TIME], "s"))
        if data_raw.metadata.get(model.MD_FILTER_NAME, None):
            stream_data.append(data_raw.metadata[model.MD_FILTER_NAME])
        if data_raw.metadata.get(model.MD_IN_WL, None):
            stream_data.append(u"ex.: %s" % units.readable_str(numpy.average(data_raw.metadata[model.MD_IN_WL]), "m", sig=3))
        if data_raw.metadata.get(model.MD_OUT_WL, None):
            stream_data.append(u"em.: %s" % units.readable_str(numpy.average(data_raw.metadata[model.MD_OUT_WL]), "m", sig=3))
        if isinstance(s, stream.OpticalStream):
            baseline = data_raw.metadata.get(model.MD_BASELINE, 0)
        else:
            baseline = numpy.min(data_raw)
        stream_data.append(baseline)
        streams_data[s] = stream_data

    # Sort by size, so that the biggest picture is first drawn (no opacity)
    def get_area(d):
        return numpy.prod(d[0].shape[0:2]) * d[0].metadata[model.MD_PIXEL_SIZE][0]

    images_opt.sort(key=get_area, reverse=True)
    images_spc.sort(key=get_area, reverse=True)
    images_std.sort(key=get_area, reverse=True)

    # Reset the first image to be drawn to the default blend operator to be
    # drawn full opacity (only useful if the background is not full black)
    if images_opt:
        images_opt[0] = (images_opt[0][0], BLEND_DEFAULT, images_opt[0][2])

    return images_opt + images_std + images_spc, streams_data, im_min_type


def physical_to_world_pos(phy_pos):
    """ Translate physical coordinates into world coordinates.
    Works both for absolute and relative values.

    phy_pos (float, float): "physical" coordinates in m
    returns (float, float)
    """
    # The y value needs to be flipped between physical and world coordinates.
    return phy_pos[0], -phy_pos[1]


def convert_streams_to_images(streams, images_cache, rgb=True):
    """ Temporary function to convert the StreamTree to a list of images as
    the export function currently expects.

    """
    images, streams_data, im_min_type = get_ordered_images(streams, rgb)

    # add the images in order
    ims = []
    im_cache = {}
    for rgbim, blend_mode, stream in images:
        # TODO: convert to RGBA later, in canvas and/or cache the conversion
        # On large images it costs 100 ms (per image and per canvas)

        if not rgb:
            # TODO use another method to fake rgba format
            rgba_im = format_rgba_darray(rgbim)
        else:
            # Get converted RGBA image from cache, or create it and cache it
            im_id = id(rgbim)
            if im_id in images_cache:
                rgba_im = images_cache[im_id]
                im_cache[im_id] = rgba_im
            else:
                rgba_im = format_rgba_darray(rgbim)
                im_cache[im_id] = rgba_im

        keepalpha = False
        date = rgbim.metadata.get(model.MD_ACQ_DATE, None)
        scale = rgbim.metadata[model.MD_PIXEL_SIZE]
        pos = physical_to_world_pos(rgbim.metadata[model.MD_POS])
        rot = rgbim.metadata.get(model.MD_ROTATION, 0)
        shear = rgbim.metadata.get(model.MD_SHEAR, 0)
        flip = rgbim.metadata.get(model.MD_FLIP, 0)

        ims.append((rgba_im, pos, scale, keepalpha, rot, shear, flip, blend_mode,
                    stream.name.value, date, stream))

    # Replace the old cache, so the obsolete RGBA images can be garbage collected
    images_cache = im_cache
    images = set_images(ims)

    return images, streams_data, images_cache, im_min_type


def get_sub_img(b_intersect, b_im_rect, im_data, total_scale):
    """ Return the minimial image data that will cover the intersection

    :param b_intersect: (rect) Intersection of the full image and the buffer
    :param b_im_rect: (rect) The area the full image would occupy in the
        buffer
    :param im_data: (DataArray) The original image data
    :param total_scale: (float, float) The scale used to convert the image data to
        buffer pixels. (= image scale * buffer scale)

    :return: (DataArray, (float, float))

    Since trimming the image will possibly change the top left buffer
    coordinates it should be drawn at, an adjusted (x, y) tuple will be
    returned as well.

    TODO: Test if scaling a sub image really has performance benefits while rendering with
    Cairo (i.e. Maybe Cairo is smart enough to render big images without calculating the pixels
    that are not visible.)

    """
    im_h, im_w = im_data.shape[:2]

    # No need to get sub images from small image data
    if im_h <= 4 or im_w <= 4:
        logging.debug("Image too small to intersect...")
        return im_data, b_im_rect[:2]

    # where is this intersection in the original image?
    unsc_rect = (
        (b_intersect[0] - b_im_rect[0]) / total_scale[0],
        (b_intersect[1] - b_im_rect[1]) / total_scale[1],
        b_intersect[2] / total_scale[0],
        b_intersect[3] / total_scale[1]
    )

    # Round the rectangle values to whole pixel values
    # Note that the width and length get "double rounded":
    # The bottom left gets rounded up to match complete pixels and that
    # value is adjusted by a rounded down top/left.
    unsc_rnd_rect = [
        int(unsc_rect[0]),  # rounding down origin
        int(unsc_rect[1]),  # rounding down origin
        math.ceil(unsc_rect[0] + unsc_rect[2]) - int(unsc_rect[0]),
        math.ceil(unsc_rect[1] + unsc_rect[3]) - int(unsc_rect[1])
    ]

    # Make sure that the rectangle fits inside the image
    if (unsc_rnd_rect[0] + unsc_rnd_rect[2] > im_w or
            unsc_rnd_rect[1] + unsc_rnd_rect[3] > im_h):
        # sometimes floating errors + rounding leads to one pixel too
        # much => just crop.
        assert(unsc_rnd_rect[0] + unsc_rnd_rect[2] <= im_w + 1)
        assert(unsc_rnd_rect[1] + unsc_rnd_rect[3] <= im_h + 1)
        unsc_rnd_rect[2] = im_w - unsc_rnd_rect[0]  # clip width
        unsc_rnd_rect[3] = im_h - unsc_rnd_rect[1]  # clip height

    # New top left origin in buffer coordinates to account for the clipping
    b_new_x = (unsc_rnd_rect[0] * total_scale[0]) + b_im_rect[0]
    b_new_y = (unsc_rnd_rect[1] * total_scale[1]) + b_im_rect[1]

    # Calculate slicing parameters
    sub_im_x, sub_im_y = unsc_rnd_rect[:2]
    sub_im_w, sub_im_h = unsc_rnd_rect[-2:]
    sub_im_w = max(sub_im_w, 2)
    sub_im_h = max(sub_im_h, 2)

    # We need to copy the data, since cairo.ImageSurface.create_for_data expects a single
    # segment buffer object (i.e. the data must be contiguous)
    im_data = im_data[sub_im_y:sub_im_y + sub_im_h,
                      sub_im_x:sub_im_x + sub_im_w].copy()

    return im_data, (b_new_x, b_new_y)


def images_to_export_data(images, view_hfw, min_res, view_pos, im_min_type, streams_data, draw_merge_ratio, rgb=True, interpolate_data=True):
    # The list of images to export
    data_to_export = []

    # meters per pixel for the focussed window
    mpp_screen = 1e-3 * wx.DisplaySizeMM()[0] / wx.DisplaySize()[0]

    # Find min pixel size
    min_pxs = min([im.metadata['dc_scale'] for im in images])

    # Check that resolution of all images remains within limits if we use
    # the smallest pixel size, otherwise adjust it
    new_res = view_hfw[0] // min_pxs[0], view_hfw[1] // min_pxs[1]
    max_res = MAX_RES_FACTOR * min_res[0], MAX_RES_FACTOR * min_res[1]
    clipped_res = tuple(numpy.clip(new_res, min_res, max_res))
    if clipped_res != new_res:
        min_pxs = tuple([a / b for a, b in zip(view_hfw, clipped_res)])

    clipped_res = int(clipped_res[0]), int(clipped_res[1])
    mag = mpp_screen / min_pxs[0]

    # Make surface based on the maximum resolution
    data_to_draw = numpy.zeros((clipped_res[0], clipped_res[1], 4), dtype=numpy.uint8)
    surface = cairo.ImageSurface.create_for_data(
        data_to_draw, cairo.FORMAT_ARGB32, clipped_res[1], clipped_res[0])
    ctx = cairo.Context(surface)

    # The buffer center is the same as the view window's center
    buffer_center = tuple(view_pos)
    buffer_scale = (min_pxs[0], min_pxs[1])
    buffer_size = clipped_res[1], clipped_res[0]

    # scale bar details
    bar_width = buffer_size[0] // 4
    actual_width = bar_width * buffer_scale[0]
    actual_width = units.round_significant(actual_width, 1)

    n = len(images)
    last_image = images.pop()
    # For every image, except the last
    i = 0
    for i, im in enumerate(images):
        if im.metadata['blend_mode'] == BLEND_SCREEN or (not rgb):
            # No transparency in case of "raw" export
            merge_ratio = 1.0
        else:
            merge_ratio = 1 - i / n

        draw_image(
            ctx,
            im,
            im.metadata['dc_center'],
            buffer_center,
            buffer_scale,
            buffer_size,
            merge_ratio,
            im_scale=im.metadata['dc_scale'],
            rotation=im.metadata['dc_rotation'],
            shear=im.metadata['dc_shear'],
            flip=im.metadata['dc_flip'],
            blend_mode=im.metadata['blend_mode'],
            interpolate_data=interpolate_data
        )
        if not rgb:
            # Create legend
            legend_to_draw = numpy.zeros((n * (buffer_size[1] // 24) + (buffer_size[1] // 12), buffer_size[0], 4), dtype=numpy.uint8)
            legend_surface = cairo.ImageSurface.create_for_data(
                legend_to_draw, cairo.FORMAT_ARGB32, buffer_size[0], n * (buffer_size[1] // 24) + (buffer_size[1] // 12))
            legend_ctx = cairo.Context(legend_surface)
            draw_export_legend(legend_ctx, images + [last_image], buffer_size, min_pxs[0], mag,
                               view_hfw[1], bar_width, actual_width, last_image.metadata['date'], streams_data, im.metadata['stream'])

            new_data_to_draw = numpy.zeros((data_to_draw.shape[0], data_to_draw.shape[1]), dtype=numpy.uint32)
            new_data_to_draw[:, :] = numpy.left_shift(data_to_draw[:, :, 2], 8, dtype=numpy.uint32) | data_to_draw[:, :, 1]
            new_data_to_draw[:, :] = new_data_to_draw[:, :] | numpy.left_shift(data_to_draw[:, :, 0], 16, dtype=numpy.uint32)
            new_data_to_draw[:, :] = new_data_to_draw[:, :] | numpy.left_shift(data_to_draw[:, :, 3], 24, dtype=numpy.uint32)
            new_data_to_draw = new_data_to_draw.astype(im_min_type)
            # Turn legend to grayscale
            new_legend_to_draw = legend_to_draw[:, :, 0] + legend_to_draw[:, :, 1] + legend_to_draw[:, :, 2]
            new_legend_to_draw = new_legend_to_draw.astype(im_min_type)
            new_legend_to_draw = numpy.where(new_legend_to_draw == 0, numpy.min(new_data_to_draw), numpy.max(new_data_to_draw))
            data_with_legend = numpy.append(new_data_to_draw, new_legend_to_draw, axis=0)
            # Clip background to baseline
            baseline = streams_data[im.metadata['stream']][-1]
            data_with_legend = numpy.clip(data_with_legend, baseline, numpy.max(new_data_to_draw))
            data_to_export.append(model.DataArray(data_with_legend, im.metadata))

            data_to_draw = numpy.zeros((clipped_res[0], clipped_res[1], 4), dtype=numpy.uint8)
            surface = cairo.ImageSurface.create_for_data(
                data_to_draw, cairo.FORMAT_ARGB32, clipped_res[1], clipped_res[0])
            ctx = cairo.Context(surface)

    if not images or last_image.metadata['blend_mode'] == BLEND_SCREEN or (not rgb):
        merge_ratio = 1.0
    else:
        merge_ratio = draw_merge_ratio

    draw_image(
        ctx,
        last_image,
        last_image.metadata['dc_center'],
        buffer_center,
        buffer_scale,
        buffer_size,
        merge_ratio,
        im_scale=last_image.metadata['dc_scale'],
        rotation=last_image.metadata['dc_rotation'],
        shear=last_image.metadata['dc_shear'],
        flip=last_image.metadata['dc_flip'],
        blend_mode=last_image.metadata['blend_mode'],
        interpolate_data=interpolate_data
    )
    # Create legend
    legend_to_draw = numpy.zeros((n * (buffer_size[1] // 24) + (buffer_size[1] // 12), buffer_size[0], 4), dtype=numpy.uint8)
    legend_surface = cairo.ImageSurface.create_for_data(
        legend_to_draw, cairo.FORMAT_ARGB32, buffer_size[0], n * (buffer_size[1] // 24) + (buffer_size[1] // 12))
    legend_ctx = cairo.Context(legend_surface)
    draw_export_legend(legend_ctx, images + [last_image], buffer_size, min_pxs[0], mag,
                       view_hfw[1], bar_width, actual_width, last_image.metadata['date'], streams_data, last_image.metadata['stream'] if (not rgb) else None)
    if not rgb:
        new_data_to_draw = numpy.zeros((data_to_draw.shape[0], data_to_draw.shape[1]), dtype=numpy.uint32)
        new_data_to_draw[:, :] = numpy.left_shift(data_to_draw[:, :, 2], 8, dtype=numpy.uint32) | data_to_draw[:, :, 1]
        new_data_to_draw[:, :] = new_data_to_draw[:, :] | numpy.left_shift(data_to_draw[:, :, 0], 16, dtype=numpy.uint32)
        new_data_to_draw[:, :] = new_data_to_draw[:, :] | numpy.left_shift(data_to_draw[:, :, 3], 24, dtype=numpy.uint32)
        new_data_to_draw = new_data_to_draw.astype(im_min_type)
        # Turn legend to grayscale
        new_legend_to_draw = legend_to_draw[:, :, 0] + legend_to_draw[:, :, 1] + legend_to_draw[:, :, 2]
        new_legend_to_draw = new_legend_to_draw.astype(im_min_type)
        new_legend_to_draw = numpy.where(new_legend_to_draw == 0, numpy.min(new_data_to_draw), numpy.max(new_data_to_draw))
        data_with_legend = numpy.append(new_data_to_draw, new_legend_to_draw, axis=0)
        # Clip background to baseline
        baseline = streams_data[last_image.metadata['stream']][-1]
        data_with_legend = numpy.clip(data_with_legend, baseline, numpy.max(new_data_to_draw))
    else:
        data_with_legend = numpy.append(data_to_draw, legend_to_draw, axis=0)
        data_with_legend[:, :, [2, 0]] = data_with_legend[:, :, [0, 2]]
        last_image.metadata[model.MD_DIMS] = 'YXC'
    data_to_export.append(model.DataArray(data_with_legend, last_image.metadata))
    return data_to_export


def add_alpha_byte(im_darray, alpha=255):

    height, width, depth = im_darray.shape

    if depth == 4:
        return im_darray
    elif depth == 3:
        new_im = numpy.empty((height, width, 4), dtype=numpy.uint8)
        new_im[:, :, -1] = alpha
        new_im[:, :, :-1] = im_darray

        if alpha != 255:
            new_im = scale_to_alpha(new_im)

        if isinstance(im_darray, DataArray):
            return DataArray(new_im, im_darray.metadata)
        else:
            return new_im
    else:
        raise ValueError("Unexpected colour depth of %d bytes!" % depth)


def scale_to_alpha(im_darray):
    """ Scale the R, G and B values to the alpha value present """

    if im_darray.shape[2] != 4:
        raise ValueError("DataArray needs to have 4 byte RGBA values!")

    im_darray[:, :, 0] *= im_darray[:, :, 3] / 255
    im_darray[:, :, 1] *= im_darray[:, :, 3] / 255
    im_darray[:, :, 2] *= im_darray[:, :, 3] / 255

    return im_darray


# Note: it's also possible to directly generate a wx.Bitmap from a buffer, but
# always implies a memory copy.
def NDImage2wxImage(image):
    """
    Converts a NDImage into a wxImage.
    Note, the copy of the data will be avoided whenever possible.
    image (ndarray of uint8 with shape YX3 or YX4): original image,
     order of last dimension is RGB(A)
    return (wxImage)
    """
    assert(len(image.shape) == 3)
    size = image.shape[1::-1]
    if image.shape[2] == 3: # RGB
        wim = wx.ImageFromBuffer(*size, dataBuffer=image) # 0 copy
        return wim
    elif image.shape[2] == 4: # RGBA
        # 2 copies
        return wx.ImageFromDataWithAlpha(*size,
                             data=numpy.ascontiguousarray(image[:, :, 0:3]),
                             alpha=numpy.ascontiguousarray(image[:, :, 3]))
    else:
        raise ValueError("image is of shape %s" % (image.shape,))

# Untested
def NDImage2wxBitmap(image):
    """
    Converts a NDImage into a wxBitmap.
    Note, the copy of the data will be avoided whenever possible.
    image (ndarray of uint8 with shape YX3 or YX4): original image,
     order of last dimension is RGB(A)
    return (wxImage)
    """
    assert(len(image.shape) == 3)
    size = image.shape[1::-1]
    if image.shape[2] == 3: # RGB
        bim = wx.EmptyBitmap(size[0], size[1], 24)
        bim.CopyFromBuffer(image, wx.BitmapBufferFormat_RGB)
        # bim = wx.BitmapFromBuffer(size[0], size[1], image)
    elif image.shape[2] == 4: # RGBA
        bim = wx.BitmapFromBufferRGBA(size[0], size[1], image)
    else:
        raise ValueError("image is of shape %s" % (image.shape,))

    return bim


def wxImage2NDImage(image, keep_alpha=True):
    """
    Converts a wx.Image into a numpy array.
    image (wx.Image): the image to convert of size MxN
    keep_alpha (boolean): keep the alpha channel when converted
    returns (numpy.ndarray): a numpy array of shape NxMx3 (RGB) or NxMx4 (RGBA)
    Note: Alpha not yet supported.
    """
    if keep_alpha and image.HasAlpha():
        shape = image.Height, image.Width, 4
        raise NotImplementedError()
    else:
        shape = image.Height, image.Width, 3

    return numpy.ndarray(buffer=image.DataBuffer, shape=shape, dtype=numpy.uint8)


def wxImageScaleKeepRatio(im, size, quality=wx.IMAGE_QUALITY_NORMAL):
    """
    Scales (down) an image so that if fits within a given bounding-box without
      changing the aspect ratio, and filling up with black bands
    im (wxImage): the image to scale
    size (int, int): the size (width, height) of the bounding box
    quality (int): scaling quality, same as image.Scale()
    return (wxImage): an image scaled to fit the size within at least one
      dimension. The other dimension will be of the requested size, but with
      only a subset containing the data.
    """
    ratio = min(size[0] / im.Width, size[1] / im.Height)
    rw = max(1, int(im.Width * ratio))
    rh = max(1, int(im.Height * ratio))
    sim = im.Scale(rw, rh, quality)

    # Add a (black) border on the small dimension
    lt = ((size[0] - rw) // 2, (size[1] - rh) // 2)
    sim.Resize(size, lt, 0, 0, 0)

    return sim
