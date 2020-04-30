# -*- coding: utf-8 -*-
"""
Created on 22 Feb 2013

@author: Rinze de Laat

Copyright © 2013 Rinze de Laat, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License version 2 as published by the Free Software
Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.


### Purpose ###

Combobox and similar controls.

"""
from __future__ import division

from builtins import str
import logging
from odemis.gui import img
import odemis.gui
from odemis.gui.comp.buttons import ImageButton, darken_image
from odemis.gui.conf.data import COLORMAPS
import wx.adv
from matplotlib import cm
import matplotlib.colors as colors
import numpy
from odemis.util.img import getColorbar, RGBTintToColormap
from PIL.ImageColor import getcolor


class ComboBox(wx.adv.OwnerDrawnComboBox):
    """ A simple sub class of OwnerDrawnComboBox that prevents a white border
    from showing around the combobox and allows for left/right caret
    navigation with the arrow keys.

    OwnerDrawnComboBox consists of a ComboCtrl, which a child TextCtrl. The
    problem with the white border comes from the fact that the background colour
    of the ComboCtrl cannot be set. Any call to that method will only result in
    the TextCtrl changing colour.

    Getting rid of margins also didn't help, since the top margin is 'stuck' at
    -1, which causes the ComboCtrl's white background colour to show.

    In the end, the solution was to draw the background ourselves, using the correct colour.

    """

    def __init__(self, *args, **kwargs):

        labels = kwargs.pop('labels', [])
        choices = kwargs.pop('choices', [])

        super(ComboBox, self).__init__(*args, **kwargs)
        # SetMargins allow the left margin to be set to 0, but the top
        # margin won't move and stays at the default -1.
        self.SetMargins(0, 0)

        self.SetForegroundColour(odemis.gui.FG_COLOUR_EDIT)
        # Use the same colour as the parent (by default)
        # HACK: there seems to be a bug in wxWidgets (v3.0.2), where
        # OwnerDrawnComboBox.GetBackgroundColour() always returns the same fixed
        # colour (in init?). So we cannot rely on it.
        bckcol = self.Parent.GetBackgroundColour()
        self.SetBackgroundColour(bckcol)

        icon = img.getBitmap("icon/arr_down_s.png")
        icon_x = 16 // 2 - icon.GetWidth() // 2
        icon_y = 16 // 2 - icon.GetHeight() // 2 - 1

        bmpLabel = ImageButton._create_bitmap(img.getImage("button/btn_def_16.png"),
                                              (16, 16), bckcol)
        dc = wx.MemoryDC()
        dc.SelectObject(bmpLabel)
        dc.DrawBitmap(icon, icon_x, icon_y)
        dc.SelectObject(wx.NullBitmap)

        hover_image = bmpLabel.ConvertToImage()
        darken_image(hover_image, 1.1)

        dis_image = bmpLabel.ConvertToImage()
        darken_image(dis_image, 0.8)

        orig_image = bmpLabel.ConvertToImage()
        darken_image(orig_image, 1.0)

        self.SetButtonBitmaps(orig_image.ConvertToBitmap(),
                              bmpHover=hover_image.ConvertToBitmap(),
                              bmpDisabled=dis_image.ConvertToBitmap(),
                              pushButtonBg=False)

        # Convert losing the focus into accepting the new value typed in
        # (generates EVT_TEXT_ENTER).
        self._prev_text = None
        self._text_changed = False
        self.Bind(wx.EVT_TEXT, self._on_text)
        self.Bind(wx.EVT_COMBOBOX, self._on_text_enter)
        self.Bind(wx.EVT_TEXT_ENTER, self._on_text_enter)
        self.Bind(wx.EVT_KILL_FOCUS, self._on_focus)

        self.Bind(wx.EVT_PAINT, self.on_paint)

        # If no labels are provided, create them from the choices
        if not labels and choices:
            labels = [str(c) for c in choices]

        for label, choice in zip(labels, choices):
            self.Append(label, choice)

        def _eat_event(evt):
            """ Quick and dirty empty function used to 'eat' mouse wheel events """

            # TODO: This solution only makes sure that the control's value
            # doesn't accidentally get altered when it gets hit by a mouse
            # wheel event. However, it also stop the event from propagating
            # so the containing scrolled window will not scroll either.
            # (If the event is skipped, the control will change value again)
            # No easy fix found in wxPython 3.0.
            pass

        self.Bind(wx.EVT_MOUSEWHEEL, _eat_event)

    def _on_text(self, evt):
        text = self.GetValue()
        if self._prev_text != text:
            self._text_changed = True
            self._prev_text = text

    def _on_text_enter(self, evt):
        self._text_changed = False
        self._prev_text = self.GetValue()

    def _on_focus(self, evt):
        # When showing/hiding the drop-down, the KILL_FOCUS event is sent,
        # although we didn't really lose the focus. In such case, no need to
        # report a change of value. Also send the event only if the text has
        # (probably) changed, to avoid sending too many events.
        if evt.GetWindow() != self and self._text_changed:
            entevt = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_ENTER, self.Id)
            wx.PostEvent(self, entevt)
        else:
            logging.debug("No sending event as focus is still on combobox or text unchanged")
        evt.Skip()  # pass it on

    def on_paint(self, evt):
        """ Handle the paint event

        Because OwnerDrawnComboBox showed the white background 'behind' the text control (1px
        at the bottom and to the right), which could not be gotten rid off, we are forced to
        paint the background in the correct colour ourselves.

        """

        dc = wx.BufferedPaintDC(self)
        self.draw(dc)
        evt.Skip()  # Make sure the event propagates, so the drop-down button will be drawn

    def draw(self, dc):
        """ Clear the widget with the correct background colour """
        back_colour = self.Parent.GetBackgroundColour()
        back_brush = wx.Brush(back_colour, wx.BRUSHSTYLE_SOLID)
        dc.SetBackground(back_brush)
        dc.Clear()

        
ITEM_WIDTH = 200
ITEM_HEIGHT = 28
COLOBAR_WITH_RATIO = 0.6


class ColorMapComboBox(ComboBox):
    """
    A special combo box that is designed for allowing the user to select a color map
    """

    def __init__(self, *args, **kwargs):
        """
        colormap_dict: dict of str -> matplotlib colormap. Corresponds to a
        name of the color map to a color map object
        """
        self.colormap_dict = kwargs['colormap_dict']
        self.choices = self.colormap_dict.keys()
        kwargs['choices'] = self.choices
        del kwargs['colormap_dict']
        super(ColorMapComboBox, self).__init__(*args, **kwargs)
        # self.SetForegroundColour(wx.Colour(0, 0, 0, alpha=wx.ALPHA_TRANSPARENT))
        # self.SetBackgroundColour(wx.Colour(0, 0, 0, alpha=wx.ALPHA_TRANSPARENT))
        self.Bind(wx.EVT_ERASE_BACKGROUND, self._on_erase_background)
        

    def OnMeasureItemWidth(self, item):
        return ITEM_WIDTH

    def OnMeasureItem(self, item):
        return ITEM_HEIGHT

    def SetCustomTintValue(self, value):
        self._tint_value = value
        self.Refresh(eraseBackground=True)

    def OnDrawItem(self, dc, rect, item, flags):

        # Draw a rectangle of the color
        item_name = self.choices[item]
        color_map = self.colormap_dict[item_name]
        
        if not color_map:
            return

        color_map = RGBTintToColormap(color_map, item_name)

        # Draw color map
        colorbar_width = rect.width * COLOBAR_WITH_RATIO
        gradient = getColorbar(color_map, colorbar_width, rect.height)
        bmp = wx.Bitmap(*gradient.shape[1::-1])
        bmp.CopyFromBuffer(gradient, format=wx.BitmapBufferFormat_RGB)
        # image = wx.ImageFromBuffer(*gradient.shape[1::-1], gradient)
        dc.DrawBitmap(bmp, 0, item * rect.height)
        dc.DrawText(item_name.title(), colorbar_width + 5, item * rect.height + 5)

    def on_paint(self, evt):
        """ Handle the paint event

        Because OwnerDrawnComboBox showed the white background 'behind' the text control (1px
        at the bottom and to the right), which could not be gotten rid off, we are forced to
        paint the background in the correct colour ourselves.

        """
        dc = wx.BufferedPaintDC(self)
        self.draw(dc)
        evt.Skip()  # Make sure the event propagates, so the drop-down button will be drawn

    def _on_erase_background(self, evt):
        dc = wx.BufferedPaintDC(self)
        self.draw(dc)
        evt.Skip()

    def draw(self, dc):
        """
        Override drawing of the combox box to show the colour
        """

        w = self.Size.GetWidth()
        h = self.Size.GetHeight()
        
        color_map = self.colormap_dict.values()[self.GetSelection()]
        color_map = RGBTintToColormap(color_map)

        gradient = getColorbar(color_map, w, h)
        bmp = wx.EmptyBitmap(*gradient.shape[1::-1])
        bmp.CopyFromBuffer(gradient, format=wx.BitmapBufferFormat_RGB)
        dc.DrawBitmap(bmp, 0, 0)

