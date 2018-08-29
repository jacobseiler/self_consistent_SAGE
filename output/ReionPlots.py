#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from astropy import cosmology

from scipy import stats

import ReadScripts as rs
import PlotScripts as ps
import CollectiveStats as collective
import ObservationalData as Obs

from mpi4py import MPI

output_format = "png"


def plot_history(z_array_reion_allmodels, 
                 lookback_array_reion_allmodels, cosmo_allmodels,
                 t_bigbang_allmodels, mass_frac_allmodels, duration_definition, 
                 model_tags, output_dir, output_tag, passed_ax=None):

    if passed_ax:
        ax1 = passed_ax
    else:
        fig1 = plt.figure(figsize = (8,8))
        ax1 = fig1.add_subplot(111)

    # When we plot the models, we also want to determine the duration of
    # reionization for each.
    duration_z = []
    duration_t = []
    for model_number in range(len(z_array_reion_allmodels)):
        # However only want to do this if we haven't passed another axis (i.e.,
        # if `plot_history` was called within `ReionData.py`.
        if not passed_ax:
            duration_z.append([]) 
            duration_t.append([])
            for val in duration_definition:
                idx = (np.abs(mass_frac_allmodels[model_number] - val)).argmin()
                duration_z[model_number].append(z_array_reion_allmodels[model_number][idx]) 
                duration_t[model_number].append(lookback_array_reion_allmodels[model_number][idx])


            print("Model {0}: Start {1:.2f} \tMid {2:.2f}\tEnd {3:.2f}\t"
                  "dz {4:.2f}\tdt {5:.2f}Myr" \
                  .format(model_number, duration_z[model_number][0],
                          duration_z[model_number][1], duration_z[model_number][-1],
                          duration_z[model_number][-1]-duration_z[model_number][0],
                          duration_t[model_number][-1]-duration_t[model_number][0]))

        ax1.plot(lookback_array_reion_allmodels[model_number], 
                 mass_frac_allmodels[model_number], 
                 color = ps.colors[model_number],
                 ls = ps.linestyles[model_number],
                 label = model_tags[model_number])

    ax1.set_xlabel(r"$\mathbf{Time \: since \: Big \: Bang \: [Myr]}$", 
                   fontsize = ps.global_labelsize)
    tick_locs = np.arange(200.0, 1000.0, 100.0)
    ax1.xaxis.set_minor_locator(mtick.MultipleLocator(ps.time_tickinterval))
    tick_labels = [r"$\mathbf{%d}$" % x for x in tick_locs]
    ax1.xaxis.set_major_locator(mtick.MultipleLocator(100))
    ax1.set_xticklabels(tick_labels, fontsize = ps.global_fontsize)

    ax1.set_ylabel(r'$\mathbf{\langle \chi_{HI} \rangle}$',
                   fontsize = ps.global_labelsize)

    tick_locs = np.arange(-0.2, 1.1, 0.2)
    ax1.set_yticklabels([r"$\mathbf{%.2f}$" % x for x in tick_locs],
                        fontsize = ps.global_fontsize)
    ax1.yaxis.set_minor_locator(mtick.MultipleLocator(0.05))
    ax1.set_ylim([-0.05, 1.05])

    ax1.set_xlim(ps.time_xlim)

    ax2 = ps.add_time_z_axis(ax1, cosmo_allmodels[0],
                             t_bigbang_allmodels[0]/1.0e3)

    ax1 = ps.adjust_axis(ax1, ps.global_axiswidth, ps.global_tickwidth,
                         ps.global_major_ticklength, ps.global_minor_ticklength) 

    if not passed_ax:
        leg = ax1.legend(loc='lower right', numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize(ps.global_legendsize)

    if passed_ax:
        return ax1
    else:
        outputFile1 = "{0}/{1}.{2}".format(output_dir, output_tag, output_format)
        fig1.savefig(outputFile1, bbox_inches='tight')  # Save the figure
        print('Saved file to {0}'.format(outputFile1))
        plt.close(fig1)

        return None


def plot_nion(z_array_reion_allmodels, lookback_array_reion_allmodels,
              cosmo_allmodels, t_bigbang_allmodels, nion_allmodels,
              model_tags, output_dir, output_tag):

    fig1 = plt.figure(figsize = (8,8))
    ax1 = fig1.add_subplot(111)

    for model_number in range(len(z_array_reion_allmodels)):

        # Units for photons are 1.0e50 so move to log properly.
        log_sum = 50 + np.log10(nion_allmodels[model_number])

        ax1.plot(lookback_array_reion_allmodels[model_number], 
                 log_sum, 
                 color = ps.colors[model_number],
                 ls = ps.linestyles[model_number],
                 label = model_tags[model_number])

    ax1 = ps.plot_bouwens2015(cosmo_allmodels[0], t_bigbang_allmodels[0], ax1)

    ax1.set_xlabel(r"$\mathbf{Time \: since \: Big \: Bang \: [Myr]}$", 
                   fontsize = ps.global_labelsize)
    tick_locs = np.arange(200.0, 1000.0, 100.0)
    ax1.xaxis.set_minor_locator(mtick.MultipleLocator(ps.time_tickinterval))
    tick_labels = [r"$\mathbf{%d}$" % x for x in tick_locs]
    ax1.xaxis.set_major_locator(mtick.MultipleLocator(100))
    ax1.set_xticklabels(tick_labels, fontsize = ps.global_fontsize)

    ax1.set_ylabel(r'$\mathbf{log_{10} \sum f_{esc}\dot{N}_\gamma \: [s^{-1} Mpc^{-3}]}$',
                   fontsize = ps.global_labelsize)
    tick_locs = np.arange(48.0, 51.25, 0.5)
    ax1.set_yticklabels([r"$\mathbf{%.1f}$" % x for x in tick_locs],
                        fontsize = ps.global_fontsize)
    ax1.set_ylim([48.4, 51.25])
    ax1.yaxis.set_minor_locator(mtick.MultipleLocator(0.25))

    ax1.set_xlim(ps.time_xlim)

    ax2 = ps.add_time_z_axis(ax1, cosmo_allmodels[0],
                             t_bigbang_allmodels[0]/1.0e3)

    ax1.text(350, 50.0, r"$\mathbf{68\%}$", horizontalalignment='center', 
             verticalalignment = 'center', fontsize = ps.global_labelsize-4)
    ax1.text(350, 50.5, r"$\mathbf{95\%}$", horizontalalignment='center',
             verticalalignment = 'center', fontsize = ps.global_labelsize-4)

    ax1 = ps.adjust_axis(ax1, ps.global_axiswidth, ps.global_tickwidth,
                         ps.global_major_ticklength, ps.global_minor_ticklength) 

    leg = ax1.legend(loc='lower right', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize(ps.global_legendsize)

    outputFile1 = "{0}/{1}.{2}".format(output_dir, output_tag, output_format)
    fig1.savefig(outputFile1, bbox_inches='tight')  # Save the figure
    print('Saved file to {0}'.format(outputFile1))
    plt.close(fig1)


def plot_ps_fixed_XHI(k, P21, PHII, fixed_XHI_values, model_tags, output_dir,
                      output_tag):

    fig, ax = plt.subplots(nrows = 1, ncols = len(fixed_XHI_values),
                           sharey='row', figsize=(16, 6))

    for model_number in range(len(k)):
        for fraction in range(len(fixed_XHI_values)):

            this_ax = ax[fraction]

            w = np.where(k[model_number][fraction] > 9e-2)[0]
            label = model_tags[model_number]
 
            this_ax.plot(k[model_number][fraction][w],
                         P21[model_number][fraction][w],
                         color = ps.colors[model_number],
                         ls = ps.linestyles[model_number],
                         lw = 2, rasterized=True, label = label)

            if model_number != 0:
                continue

            this_ax = ps.adjust_axis(this_ax, ps.global_axiswidth,
                                     ps.global_tickwidth,
                                     ps.global_major_ticklength,
                                     ps.global_minor_ticklength)


            this_ax.set_xlabel(r'$\mathbf{k \: \left[Mpc^{-1}h\right]}$',
                               size = ps.global_labelsize)
            this_ax.set_xscale('log')
            this_ax.set_xlim([7e-2, 5.5])

            HI_string = "{0:.2f}".format(fixed_XHI_values[fraction])

            label = r"$\mathbf{\langle \chi_{HI}\rangle = " +HI_string + r"}$"
            this_ax.text(0.05, 0.9, label, transform = this_ax.transAxes,
                         fontsize = ps.global_fontsize)


            tick_locs = np.arange(-3, 2.5, 1.0)
            this_ax.set_xticklabels([r"$\mathbf{10^{%d}}$" % x for x in tick_locs],
                                    fontsize = ps.global_fontsize)

    ax[0].set_ylabel(r'$\mathbf{\Delta_{21}^2 \left[mK^2\right]}$',
                     size = ps.global_labelsize)
    ax[0].set_yscale('log', nonposy='clip')

    tick_locs = np.arange(-1.0, 5.5, 1.0)
    ax[0].set_yticklabels([r"$\mathbf{10^{%d}}$" % x for x in tick_locs],
                          fontsize = ps.global_fontsize)

    leg = ax[0].legend(loc='lower right', numpoints=1,
                       labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize(ps.global_legendsize-2)

    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.0, hspace = 0.0)


    outputFile = "{0}/{1}.{2}".format(output_dir,
                                     output_tag,
                                     output_format)
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to {0}'.format(outputFile))
    plt.close()


def  plot_duration_contours(z_array_reion_allmodels,
                            lookback_array_reion_allmodels, cosmo_allmodels,
                            t_bigbang_allmodels, mass_frac_allmodels,
                            duration_contours_limits, duration_definition,
                            model_tags, output_dir, output_tag, passed_ax=None):

    fig1 = plt.figure(figsize = (8,8))
    ax1 = fig1.add_subplot(111)

    # First find the duration of reionization for each model.
    duration_z = []
    duration_t = []
    dt = []

    # Need to get rid of the (alpha, beta) = (0.0, 0.0) model because it has no
    # reionization (duration is not defined).
    if duration_contours_limits[0][0] == duration_contours_limits[1][0] == 0:
        dt.append(np.nan)
 
    # We need to be careful here. For low values of alpha + beta, reionization
    # won't actually complete.  Hence we need to check `duration_z` and see
    # those models in which reionization is 'completed' at the last snapshot.
    reion_completed = []
    for model_number in range(len(mass_frac_allmodels)):
        mass_frac_thismodel = mass_frac_allmodels[model_number]

        duration_z.append([]) 
        duration_t.append([])
        for val in duration_definition:
            idx = (np.abs(mass_frac_thismodel - val)).argmin()
            duration_z[model_number].append(z_array_reion_allmodels[model_number][idx]) 
            duration_t[model_number].append(lookback_array_reion_allmodels[model_number][idx]) 

            if (val == duration_definition[-1]) and \
               (idx == len(mass_frac_thismodel)):
                reion_completed.append(0)
            else:
                reion_completed.append(1)

        dt.append(duration_t[model_number][-1] - duration_t[model_number][0])

    dt = np.array(dt)
    reion_completed = np.array(reion_completed)

    alpha = np.arange(duration_contours_limits[0][0],
                      duration_contours_limits[0][1] + duration_contours_limits[0][2],
                      duration_contours_limits[0][2])

    beta = np.arange(duration_contours_limits[1][0],
                     duration_contours_limits[1][1] + duration_contours_limits[1][2],
                     duration_contours_limits[1][2])

    X, Y = np.meshgrid(beta, alpha)
    dt = dt.reshape(len(X), len(Y))
    reion_completed = reion_completed.reshape(len(X), len(Y))

    print(reion_completed)
    print(reion_completed.shape)

    CS = ax1.contour(Y, X, dt)
    ax1.clabel(CS, inline=1, fontsize = ps.global_fontsize)

    ax1.set_xlabel(r"$\mathbf{\alpha}$", size = ps.global_fontsize)
    ax1.set_ylabel(r"$\mathbf{\beta}$", size = ps.global_fontsize)

    ax1 = ps.adjust_axis(ax1, ps.global_axiswidth, ps.global_tickwidth,
                         ps.global_major_ticklength, ps.global_minor_ticklength) 

    ax1.xaxis.set_major_locator(mtick.MultipleLocator(0.1))
    ax1.yaxis.set_major_locator(mtick.MultipleLocator(0.05))

    tick_locs = np.arange(min(alpha)-0.1, max(alpha)+0.1, 0.1)
    ax1.set_xticklabels([r"$\mathbf{%.1f}$" % x for x in tick_locs],
                        fontsize = ps.global_fontsize)

    tick_locs = np.arange(min(beta)-0.05, max(beta)+0.05, 0.05)
    ax1.set_yticklabels([r"$\mathbf{%.2f}$" % x for x in tick_locs],
                        fontsize = ps.global_fontsize)

    outputFile = "{0}/{1}.{2}".format(output_dir, output_tag, output_format)
    plt.savefig(outputFile, bbox_inches='tight')  # Save the figure
    print('Saved file to {0}'.format(outputFile))
    plt.close()


def plot_tau(z_array_reion_allmodels, lookback_array_reion_allmodels,
             cosmo_allmodels, t_bigbang_allmodels, tau_allmodels,
             model_tags, output_dir, output_tag, passed_ax=None):
   
 
    if passed_ax:
        ax1 = passed_ax
    else:
        fig1 = plt.figure(figsize = (8,8))
        ax1 = fig1.add_subplot(111)

    for model_number in range(len(tau_allmodels)):
        ax1.plot(lookback_array_reion_allmodels[model_number], 
                 tau_allmodels[model_number], 
                 color = ps.colors[model_number],
                 ls = ps.linestyles[model_number],
                 label = model_tags[model_number])
        
    ax1.set_xlabel(r"$\mathbf{Time \: since \: Big \: Bang \: [Myr}]$",
                  size = ps.global_labelsize)

    ax1.set_ylabel(r"$\mathbf{\tau}$",
                 size = ps.global_labelsize)
    ax1.set_ylim([0.042, 0.072])
    tick_locs = np.arange(0.04, 0.075, 0.005)
    ax1.set_yticklabels([r"$\mathbf{%.3f}$" % x for x in tick_locs],
                        fontsize = ps.global_fontsize)

    ax1 = ps.adjust_axis(ax1, ps.global_axiswidth, ps.global_tickwidth,
                         ps.global_major_ticklength, ps.global_minor_ticklength) 
    ax2 = ps.add_time_z_axis(ax1, cosmo_allmodels[0],
                             t_bigbang_allmodels[0]/1.0e3)

    ax1.fill_between(np.arange(200, 1200, 0.01), 0.058 - 0.012, 0.058 + 0.012,
                     color = 'k', alpha = 0.3)
    ax1.axhline(y = 0.058, xmin = 0, xmax = 20, color = 'k', alpha = 0.3)
    ax1.text(850, 0.0570, r"$\mathrm{Planck \: 2016}$")

    if not passed_ax:
        leg = ax1.legend(loc='upper right', numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize(ps.global_legendsize)
 
    if passed_ax:
        return ax1
    else:
        outputFile = "{0}/{1}.{2}".format(output_dir, output_tag, output_format)
        plt.savefig(outputFile, bbox_inches='tight')  # Save the figure
        print('Saved file to {0}'.format(outputFile))
        plt.close()


def plot_combined_history_tau(z_array_reion_allmodels,
                              lookback_array_reion_allmodels,                              
                              cosmo_allmodels, t_bigbang_allmodels, 
                              mass_frac_allmodels, tau_allmodels, model_tags,
                              output_dir, output_tag):

    fig, ax = plt.subplots(nrows = 1, ncols = 2, 
                           sharex=False, sharey=False, figsize=(16, 8))

    # Note: The `None` here is the duration definition.  We don't want to
    # calculate the duration as it was already done when `plot_history` was
    # called from `ReionData.py`.
    ax[0] = plot_history(z_array_reion_allmodels, 
                         lookback_array_reion_allmodels, cosmo_allmodels,
                         t_bigbang_allmodels, mass_frac_allmodels, None, 
                         model_tags, output_dir, output_tag, passed_ax=ax[0])

    ax[1] = plot_tau(z_array_reion_allmodels, lookback_array_reion_allmodels,
                     cosmo_allmodels, t_bigbang_allmodels, tau_allmodels,
                     model_tags, output_dir, output_tag, passed_ax=ax[1])

    leg = ax[0].legend(loc='upper right', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize(ps.global_legendsize)

    plt.tight_layout()    

    outputFile = "{0}/{1}.{2}".format(output_dir, output_tag, output_format)
    plt.savefig(outputFile, bbox_inches='tight')  # Save the figure
    print('Saved file to {0}'.format(outputFile))
    plt.close()

