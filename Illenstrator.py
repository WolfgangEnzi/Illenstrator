"""
================================================================================

Illenstrator v1.0 - Wolfgang Enzi 2021

Illenstrator is a python program designed to illustrate some of the
basic concepts of strong gravitational lensing in an interactive manner.

I wrote this program for my supervisor Dr. Simona Vegetti as a thank you
for all her guidance and support during my time at the Max Planck Institute
for Astrophysics and I hope that this small tool will benefit students
both new and old.

Notes on this version:
- The Textbox tool of matplotlib is quite slow, but an improvement may require
restucturing the code significantly. Potentially this can be solved with
Tkinter
- It would be nice to add more precise caustics + critical curves

================================================================================
"""

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Slider, Button, RadioButtons
import sys
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.ndimage import gaussian_filter as gauf
import matplotlib as mpl
import time

sys.setrecursionlimit(200)
mpl.rc('image', cmap='Spectral_r')

# ==============================================================================
if __name__ == "__main__":

    # function describing the deflection angle of an SIE lens which has its
    # major axis aligned with the x axis
    def SIE_defl(xx, yy, f, re):

        phi = np.arctan2(yy, xx)
        fp = np.sqrt(1.0 - f * f)
        da = np.arcsinh(fp / f * np.cos(phi)) * np.sqrt(f) / (fp)
        db = np.arcsin(fp * np.sin(phi)) * np.sqrt(f) / (fp)

        return da * re * np.sqrt(f), db * re * np.sqrt(f)

    # function describing the deflection angle of an SIE lens
    def RT_SIE_defl(gamma, x0, y0, xx, yy, a, b):

        if a > b:
            f = b / a
            re = a
            gamma += np.pi / 2.0

        else:
            f = a / b
            re = b

        xxp = np.cos(gamma) * (xx - x0) + np.sin(gamma) * (yy - y0)
        yyp = - np.sin(gamma) * (xx - x0) + np.cos(gamma) * (yy - y0)
        da, db = SIE_defl(xxp, yyp, f, re)
        dx = np.cos(gamma) * da - np.sin(gamma) * db
        dy = np.sin(gamma) * da + np.cos(gamma) * db

        return dx, dy

    # function describing the convergence of an SIE lens
    def RT_SIE_kappa(gamma, x0, y0, xx, yy, a, b):

        if a > b:
            f = b / a
            re = a

        else:
            f = a / b
            re = b
            gamma += np.pi / 2.0

        xxp = np.cos(gamma) * (xx - x0) + np.sin(gamma) * (yy - y0)
        yyp = - np.sin(gamma) * (xx - x0) + np.cos(gamma) * (yy - y0)
        psi = np.sqrt(xxp * xxp + yyp * yyp / f / f)
        kappa = re / (2.0 * psi)

        return kappa

    # function describing the brightness of a Gaussian shaped source
    def Gauss_source(gamma, x0, y0, xx, yy, a, b):

        if a > b:
            f = b / a
            re = a

        else:
            f = a/b
            re = b
            gamma += np.pi/2.0

        xxp = np.cos(gamma) * (xx - x0) + np.sin(gamma) * (yy - y0)
        yyp = - np.sin(gamma) * (xx - x0) + np.cos(gamma) * (yy - y0)
        psi2 = (xxp * xxp + yyp * yyp / f / f) / re / re
        bright = np.exp(- 0.5 * psi2)

        return bright

    # ==========================================================================

    # function that updates the arrays and colorbar limits of plots
    def update_pcm(plot, cbar, data):
        plot.set_array(data.ravel())
        plot.set_clim(np.min(data), np.max(data))
       
    # ==========================================================================

    # function that draws an ellipse into a given subplot ax
    def draw_ell(ax, a, b, gamma, x, y):
        ell_radius_x = a
        ell_radius_y = b
        gamma_deg = gamma * 180.0 / np.pi
        ellipse = Ellipse((x, y), width=ell_radius_x * 2.0,
                          height=ell_radius_y * 2.0, edgecolor="k",
                          ls=":", facecolor="None", zorder=10, angle=gamma_deg)
        u = ax.add_patch(ellipse)
        return u

    # ==========================================================================

    # parameters that are used throughout the code

    Nx = 150
    Ny = 120
    xmin = -2.0+1e-30
    xmax = 2.0
    ymin = -2.0+1e-30
    ymax = 2.0

    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny
    x = ymin + np.arange(Nx) * dx + dx/2.0
    y = ymin + np.arange(Ny) * dy + dy/2.0

    xx, yy = np.meshgrid(x, y)

    lensed = np.zeros((Ny, Nx))
    source = np.zeros((Ny, Nx))
    deflx = np.zeros((Ny, Nx))
    defly = np.zeros((Ny, Nx))
    kappa = np.zeros((Ny, Nx))
    mag = np.zeros((Ny, Nx))

    rad_flag = 0

    # ==========================================================================

    # parameters that are used to draw ellipses and lines to indicate the
    # structures to be added to the convergence or source

    lensed_sx01 = np.nan
    lensed_sy01 = np.nan
    lensed_sx02 = np.nan
    lensed_sy02 = np.nan
    lensed_sx03 = np.nan
    lensed_sy03 = np.nan
    lensed_lines = []
    lensed_pars = []
    state_lensed = -1

    a = 0
    b = 0
    gamma0 = 0

    source_sx01 = np.nan
    source_sy01 = np.nan
    source_sx02 = np.nan
    source_sy02 = np.nan
    source_sx03 = np.nan
    source_sy03 = np.nan
    source_lines = []
    source_pars = []
    state_source = -1

    # ==========================================================================

    # function that updates the plots if the convergence has been changed
    def update_kappa():

        global state_lensed, lensed_sx01, lensed_sy01, lensed_sx02, lensed_sy02
        global lensed_sx03, lensed_sy03, kappa, deflx, defly, mag
        global xx, yy, a, b, gamma0, lensed_lines, plot_lensed, ax_lensed
        global ax_bl, ax_br, plot_bl, plot_br, lensed_pars, source_pars
        global cbar_lensed, cbar_kappa, cbar_bl, cbar_br, lensed, source

        deflx = np.zeros((Ny, Nx))
        defly = np.zeros((Ny, Nx))
        kappa = np.zeros((Ny, Nx))
        mag = np.zeros((Ny, Nx))

        for i in range(len(lensed_pars)):
            p = lensed_pars[i]
            kappa += RT_SIE_kappa(p[2], p[3], p[4], xx, yy, p[0], p[1])
            deflx_, defly_ = RT_SIE_defl(p[2], p[3], p[4], xx, yy, p[0], p[1])
            deflx += deflx_
            defly += defly_

        xx_source = xx - deflx
        yy_source = yy - defly

        if len(lensed_pars) > 0:
            imag = np.fabs(np.gradient(xx_source, x, axis=1)
                           * np.gradient(yy_source, y, axis=0)
                           - np.gradient(xx_source, y, axis=0)
                           * np.gradient(yy_source, x, axis=1))
            mag = 1.0 / (1e-15 + gauf(imag, 0.5))
        else:
            kappa = np.ones((Ny, Nx))
            mag = np.ones((Ny, Nx))

        lensed = np.zeros((Ny, Nx))

        for i in range(len(source_pars)):
            p = source_pars[i]
            lensed += Gauss_source(p[2], p[3], p[4],
                                   xx - deflx, yy - defly, p[0], p[1])

        update_pcm(plot_lensed, cbar_lensed, lensed)

        if rad_flag == 0:
            update_pcm(plot_bl, cbar_bl, deflx)
            update_pcm(plot_br, cbar_br, defly)

        if rad_flag == 1:
            update_pcm(plot_bl, cbar_bl, np.log10(mag))
            update_pcm(plot_br, cbar_br, np.log10(kappa))

    # ==========================================================================

    # function that updates the plots if the source has been changed
    def update_source():

        source = np.zeros((Ny, Nx))
        lensed = np.zeros((Ny, Nx))
        for i in range(len(source_pars)):
            p = source_pars[i]
            source += Gauss_source(p[2], p[3], p[4], xx, yy, p[0], p[1])
            lensed += Gauss_source(p[2], p[3], p[4], xx - deflx,
                                   yy - defly, p[0], p[1])

        update_pcm(plot_source, cbar_source, source)
        update_pcm(plot_lensed, cbar_lensed, lensed)

    # ==========================================================================

    # function that determines the state and what to do with clicks when
    # placing a SIE lens
    def check_state_lensed(state, event):

        global state_lensed, lensed_sx01, lensed_sy01, lensed_sx02, lensed_sy02
        global lensed_sx03, lensed_sy03, kappa, deflx, defly, mag
        global xx, yy, a, b, gamma0, lensed_lines, plot_lensed, ax_lensed
        global ax_bl, ax_br, plot_bl, plot_br, lensed_pars, source_pars
        global cbar_lensed, cbar_kappa, cbar_bl, cbar_br, lensed, source

        if state_lensed == 3:

            update_kappa()

            state_lensed = -1
            for i in range(len(lensed_lines))[::-1]:
                lensed_lines[i][0].remove()
            lensed_lines = []

        if state_lensed == 2:
            lensed_sx03 = event.xdata
            lensed_sy03 = event.ydata
            v1 = np.array([lensed_sx02-lensed_sx01, lensed_sy02-lensed_sy01])
            v2 = np.array([lensed_sx03-lensed_sx01, lensed_sy03-lensed_sy01])
            v3 = (np.identity(2) * 1.0 - np.outer(v1, v1)
                  * 1.0 / np.sum(v1 * v1)).dot(v2)
            q1 = np.sqrt(v1.dot(v1))
            gamma0 = np.arctan2(v1[1], v1[0])
            q2 = np.sqrt(v3.dot(v3))
            a = q1
            b = q2
            lensed_pars += [[a, b, gamma0, lensed_sx01, lensed_sy01]]
            lensed_lines += [ax_lensed.plot([lensed_sx01, lensed_sx03],
                                            [lensed_sy01, lensed_sy03],
                                            lw=1, ls="--", color="k",
                                            zorder=10),
                             [draw_ell(ax_lensed, a, b, gamma0,
                                       lensed_sx01, lensed_sy01)],
                             ax_lensed.plot([lensed_sx01, lensed_sx01 + v3[0]],
                                            [lensed_sy01, lensed_sy01 + v3[1]],
                                            lw=1, color="k", zorder=10)]

            state_lensed += 1

        if state_lensed == 1:
            lensed_sx02 = event.xdata
            lensed_sy02 = event.ydata
            lensed_lines += [ax_lensed.plot([lensed_sx01, lensed_sx02],
                                            [lensed_sy01, lensed_sy02],
                                            lw=1, color="k", zorder=10)]
            state_lensed += 1

        if state_lensed == 0:
            lensed_sx01 = event.xdata
            lensed_sy01 = event.ydata
            lensed_lines += [ax_lensed.plot([lensed_sx01], [lensed_sy01],
                                            marker=".", lw=1, color="k",
                                            zorder=10)]
            state_lensed += 1

        fig.canvas.draw_idle()

    # ==========================================================================

    # function that determines the state and what to do with clicks when
    # placing a Gaussian source
    def check_state_source(state, event):

        global state_source, source_sx01, source_sy01, source_sx02, source_sy02
        global source_sx03, source_sy03, kappa, deflx, defly, mag
        global xx, yy, a, b, gamma0, source_lines, plot_source, ax_source
        global ax_bl, ax_br, plot_bl, plot_br, source_pars, source, lensed
        global cbar_source, cbar_kappa, cbar_bl, cbar_br, lensed_pars

        if state_source == 3:

            update_source()

            state_source = -1
            for i in range(len(source_lines))[::-1]:
                source_lines[i][0].remove()
            source_lines = []

        if state_source == 2:
            source_sx03 = event.xdata
            source_sy03 = event.ydata
            v1 = np.array([source_sx02 - source_sx01,
                           source_sy02 - source_sy01])
            v2 = np.array([source_sx03 - source_sx01,
                           source_sy03 - source_sy01])
            v3 = (np.identity(2) * 1.0 - np.outer(v1, v1)
                  * 1.0 / np.sum(v1 * v1)).dot(v2)
            q1 = np.sqrt(v1.dot(v1))
            gamma0 = np.arctan2(v1[1], v1[0])
            q2 = np.sqrt(v3.dot(v3))
            a = q1
            b = q2
            source_pars += [[a / 2.0, b / 2.0, gamma0,
                            source_sx01, source_sy01]]
            source_lines += [ax_source.plot([source_sx01, source_sx03],
                                            [source_sy01, source_sy03], lw=1,
                                            ls="--", color="k", zorder=10),
                             [draw_ell(ax_source, a, b, gamma0,
                                       source_sx01, source_sy01)],
                             ax_source.plot([source_sx01, source_sx01 + v3[0]],
                                            [source_sy01, source_sy01 + v3[1]],
                                            lw=1, color="k", zorder=10)]

            state_source += 1

        if state_source == 1:
            source_sx02 = event.xdata
            source_sy02 = event.ydata
            source_lines += [ax_source.plot([source_sx01, source_sx02],
                                            [source_sy01, source_sy02],
                                            lw=1, color="k", zorder=10)]
            state_source += 1

        if state_source == 0:
            source_sx01 = event.xdata
            source_sy01 = event.ydata
            source_lines += [ax_source.plot([source_sx01], [source_sy01],
                                            marker=".", lw=1,
                                            color="k", zorder=10)]
            state_source += 1

        fig.canvas.draw_idle()

    # ==========================================================================

    fig = plt.figure(figsize=(8, 6))
    fig.canvas.set_window_title('Illenstrator')

    # ==========================================================================

    # placing the plots and adding limits / labels

    ax_lensed = plt.subplot2grid((2, 2), (0, 0))
    plot_lensed = plt.pcolormesh(x, y, lensed)
    cbar_lensed = plt.colorbar(plot_lensed, format='%.1f')
    ax_lensed.set_title("lensed")
    plt.ylabel("y [as]")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    ax_source = plt.subplot2grid((2, 2), (0, 1))
    plot_source = plt.pcolormesh(x, y, source)
    ax_source.set_title("source")
    cbar_source = plt.colorbar(plot_source, format='%.1f')

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    ax_bl = plt.subplot2grid((2, 2), (1, 0))
    plot_bl = plt.pcolormesh(x, y, deflx)
    cbar_bl = plt.colorbar(plot_bl, format='%.1f')
    plt.xlabel("x [as]")
    plt.ylabel("y [as]")
    ax_bl.set_title(r"$\alpha_x$")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    ax_br = plt.subplot2grid((2, 2), (1, 1))
    plot_br = plt.pcolormesh(x, y, defly)
    cbar_br = plt.colorbar(plot_br, format='%.1f')
    ax_br.set_title(r"$\alpha_y$")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("x [as]")

    # ==========================================================================

    # handling of the general clicking on subplots
    def onclick(event):

        if (event.inaxes == ax_lensed):
            check_state_lensed(state_lensed, event)

        if (event.inaxes == ax_source):
            check_state_source(state_source, event)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # ==========================================================================

    # function that removes all SIE lenses
    def reset_Kappa(event):

        global lensed_pars
        del lensed_pars
        lensed_pars = []
        update_kappa()

        fig.canvas.draw_idle()

    # ==========================================================================

    # function that removes all Gaussian sources
    def reset_Source(event):

        global source_pars
        del source_pars
        source_pars = []
        update_source()

        fig.canvas.draw_idle()

    # ==========================================================================

    # box determining the prefix for saved/loaded data
    initial_text = "./prefix_"
    sl_text = initial_text

    # update filename prefix for load/save
    def submit(text):
        global sl_text
        sl_text = text

    # sleep
    def on_change(text):
        time.sleep(0.0001)

    axbox = plt.axes([0.8, 0.8, 0.15, 0.04])
    text_box = TextBox(axbox, '', initial=initial_text)
    text_box.on_submit(submit)
    text_box.on_text_change(on_change)

    # ==========================================================================

    # button to save the sources and lenses to files with
    Saveax = plt.axes([0.825, 0.725, 0.1, 0.04])
    button8 = Button(Saveax, 'Save', hovercolor='0.975')

    # function that stores the lens and source parameters
    def save_all(event):
        np.savetxt(sl_text + "lens.txt", np.array(lensed_pars))
        np.savetxt(sl_text + "source.txt", np.array(source_pars))

    button8.on_clicked(save_all)

    # ==========================================================================

    # button to load the sources and lenses from a file
    Loadax = plt.axes([0.825, 0.675, 0.1, 0.04])
    button9 = Button(Loadax, 'Load', hovercolor='0.975')

    # function that loads all lens and source parameters from files
    def load_all(event):
        global lensed_pars, source_pars

        lensed_pars = np.loadtxt(sl_text + "lens.txt").tolist()
        source_pars = np.loadtxt(sl_text + "source.txt").tolist()

        update_kappa()
        update_source()

        fig.canvas.draw_idle()

    button9.on_clicked(load_all)

    # ==========================================================================

    # button to reset everything
    resetax = plt.axes([0.825, 0.525, 0.1, 0.04])
    button = Button(resetax, 'Reset all', hovercolor='0.975')

    # function that resets all source and lens parameters
    def reset_all(event):

        reset_Kappa(event)
        reset_Source(event)

    button.on_clicked(reset_all)

    # ==========================================================================

    # button to remove the last SIE lens
    removeSIEax = plt.axes([0.2, 0.92, 0.04, 0.04])
    button7 = Button(removeSIEax, '-', hovercolor='0.975')

    # function that removes the last SIE added to the lens model
    def remove_SIE(event):

        global lensed_pars

        if len(lensed_pars) > 0:
            lensed_pars = lensed_pars[:-1]
            update_kappa()
            fig.canvas.draw_idle()

    button7.on_clicked(remove_SIE)

    # ==========================================================================

    # button to add an SIE lens
    addSIEax = plt.axes([0.125, 0.92, 0.04, 0.04])
    button2 = Button(addSIEax, '+', hovercolor='0.975')

    # function enables to add an SIE lens the model
    def addSIEax(event):

        global state_lensed
        if state_lensed == -1:
            state_lensed = 0

    button2.on_clicked(addSIEax)

    # ==========================================================================

    # button to remove all SIE lenses
    resetKappa = plt.axes([0.275, 0.92, 0.04, 0.04])
    button3 = Button(resetKappa, '0', hovercolor='0.975')

    button3.on_clicked(reset_Kappa)

    # ==========================================================================

    # button to remove the last Gaussian source
    removeSourceax = plt.axes([0.575, 0.92, 0.04, 0.04])
    button6 = Button(removeSourceax, '-', hovercolor='0.975')

    # function that removes the last Gaussian source
    def remove_Source(event):

        global source_pars
        if len(source_pars) > 0:
            source_pars = source_pars[:-1]
            update_source()
            fig.canvas.draw_idle()

    button6.on_clicked(remove_Source)

    # ==========================================================================

    # button to add a Gaussian source
    addSource = plt.axes([0.5, 0.92, 0.04, 0.04])
    button4 = Button(addSource, '+', hovercolor='0.975')

    # function that adds a Gaussian source to the model
    def addSource(event):
        global state_source
        if state_source == -1:
            state_source = 0

    button4.on_clicked(addSource)

    # ==========================================================================

    # button to remove all Gaussian sources
    resetSource = plt.axes([0.65, 0.92, 0.04, 0.04])
    button5 = Button(resetSource, '0', hovercolor='0.975')

    button5.on_clicked(reset_Source)

    # ==========================================================================

    # radio buttons to choose what is plotted in the bottom panels
    rax1 = plt.axes([0.80, 0.125, 0.15, 0.20])
    lab = (r'$\alpha_x$' + '\n' + r'$\alpha_y$', r'$\log_{10}\mu_{\rm s}$'
           + '\n' + r'$\log_{10}\kappa$')
    radio1 = RadioButtons(rax1, labels=lab, active=0, activecolor="gray")

    # function that allows to switch between different quantities
    # that are displayed in the lower two panels
    def select_disp1(label):

        global rad_flag
        rad_flag = lab.index(label)

        if rad_flag == 0:
            update_pcm(plot_bl, cbar_bl, deflx)
            update_pcm(plot_br, cbar_br, defly)
            ax_bl.set_title(r"$\alpha_x$")
            ax_br.set_title(r"$\alpha_y$")

        if rad_flag == 1:

            update_pcm(plot_bl, cbar_bl, np.log10(mag))
            update_pcm(plot_br, cbar_br, np.log10(kappa))
            ax_bl.set_title(r"$\log_{10} \mu_{\rm s}$")
            ax_br.set_title(r"$\log_{10} \kappa$")

        fig.canvas.draw_idle()

    radio1.on_clicked(select_disp1)

    # ==========================================================================

    plt.tight_layout()
    plt.subplots_adjust(right=0.75, top=0.85, wspace=0.25)

    plt.show()
