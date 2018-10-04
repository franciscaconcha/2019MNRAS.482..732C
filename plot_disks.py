"""
Script to reproduce Figures 1, 2, and 3 of the paper
For the rest of the figures use find_models.py
"""

from amuse.plot import *
from amuse import io
from amuse.lab import *

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import matplotlib.lines as mlines

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 24, })
rc('text', usetex=True)
rc('axes', labelsize=26)  # fontsize of the x and y labels
mpl.rcParams['xtick.major.pad'] = 8  # to avoid overlapping x/y labels
mpl.rcParams['ytick.major.pad'] = 8  # to avoid overlapping x/y labels

# Had to do this for now as a workaround, will try to get rid of it soon
global plot_colors
plot_colors = {"gas": "#ca5670", "no_gas": "#638ccc", "gas_expulsion": "#72a555"}


# To manage plot legends	
class NoGasObject(object):
    pass


class GasObject(object):
    pass


class GasExpObject(object):
    pass


class IsolatedObject(object):
    pass


class NoGasObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5],
                           [0.7 * height, 0.7 * height],
                           lw=3,
                           color=plot_colors["no_gas"])
        l2 = mlines.Line2D([x0, y0 + width + 5],
                           [0.2 * height, 0.2 * height],
                           linestyle='--',
                           lw=3,
                           color=plot_colors["no_gas"])
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class GasObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5],
                           [0.7 * height, 0.7 * height],
                           lw=3,
                           color=plot_colors["gas"])
        l2 = mlines.Line2D([x0, y0 + width + 5],
                           [0.2 * height, 0.2 * height],
                           linestyle='--',
                           lw=3,
                           color=plot_colors["gas"])
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class GasExpObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5],
                           [0.7 * height, 0.7 * height],
                           lw=3,
                           color=plot_colors["gas_expulsion"])
        l2 = mlines.Line2D([x0, y0 + width + 5],
                           [0.2 * height, 0.2 * height],
                           linestyle='--',
                           lw=3,
                           color=plot_colors["gas_expulsion"])
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class IsolatedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5],
                           [0.7 * height, 0.7 * height],
                           lw=2,
                           color='k')
        l2 = mlines.Line2D([x0, y0 + width + 5],
                           [0.2 * height, 0.2 * height],
                           linestyle='--',
                           lw=2,
                           color='k')
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


def disk_characteristic_radius(star, t):
    """ Return the characteristic radius of a circumstellar disk on a given moment.
        Eq. (1) of paper.

    :param star: star with the disk
    :param t: moment of time at which the characteristic radius is to be evaluated
    :return: characteristic disk radius
    """
    gamma = 1.
    T = 1 + (t - star.last_encounter) / star.viscous_timescale
    return (T ** (1 / (2 - gamma))) * star.initial_characteristic_disk_radius


def disk_mass(star, t):
    """ Return the total mass of a circumstellar disk on a given moment.
        Eq. (2) of paper.

    :param star: star with the disk
    :param t: moment of time at which the disk mass is to be evaluated
    :return: total disk mass at moment t
    """
    gamma = 1.
    T = 1 + (t - star.last_encounter) / star.viscous_timescale
    return T ** (1 / (2 * gamma - 4)) * star.initial_disk_mass


def disk_accretion_rate(star, t):
    """ Return the disk accretion rate of a circumstellar disk on a given moment.
        Eq. (5) of paper.

    :param star: star with the disk
    :param t: moment of time at which the accretion rate is to be evaluated
    :return: disk accretion rate
    """
    gamma = 1.
    T = 1 + (t - star.last_encounter) / star.viscous_timescale
    return T ** ((5 - 2 * gamma) / (2 * gamma - 4)) * star.initial_disk_mass / (4 - 2 * gamma) / star.viscous_timescale


def viscous_timescale(star, alpha):
    """ Compute the viscous timescale of the circumstellar disk.
        Eq. (9) of paper.

    :param star: star with the circumstellar disk
    :return: viscous timescale of the circumstellar disk
    """
    # Negative of the temperature profile exponent, i.e. q in $T \propto R^{-q}$
    temperature_profile = 0.5

    # Reference distance from the star at which the disk temperature is given
    Rref = 1.0 | units.AU

    # Disk temperature at the reference distance for a star with solar luminosity
    Tref = 280.0 | units.K

    # Molar mass of the gas
    mu = 2.3 | units.g / units.mol

    # Viscosity exponent
    gamma = 1.0

    stellar_evolution = SeBa()
    stellar_evolution.particles.add_particles(Particles(mass=star.stellar_mass))
    stellar_luminosity = stellar_evolution.particles.luminosity.value_in(units.LSun)
    stellar_evolution.stop()

    R = star.initial_characteristic_disk_radius
    T = Tref * (stellar_luminosity) ** 0.25
    q = temperature_profile
    M = constants.G * star.stellar_mass

    return mu * (R ** (0.5 + q)) * (M ** 0.5) / 3 / alpha / ((2 - gamma) ** 2) / constants.molar_gas_constant / T / (
            Rref ** q)


def plot_initial_viscous_timescales(save=False):
    """ Plot viscous timescale at t=0 for different stellar masses.
        Generates Figure 1 of paper.

    :param save: True to save the figure to png file
    """
    fig = plt.figure(figsize=(10, 8), dpi=90)
    ax = plt.subplot(111)
    alphas = [1e-2, 5e-3, 1e-4]
    plot_labels = [r'$\alpha = 1 \times {10}^{-2}$', r'$\alpha = 5 \times {10}^{-3}$', r'$\alpha = 1 \times {10}^{-4}$']
    line_styles = ['-', '--', ':']

    Delta = 0.05
    M = np.arange(-1.0, 2.0 + Delta, Delta)
    M = 10 ** M
    stars = Particles(len(M))
    stars.stellar_mass = M | units.MSun
    stars.initial_characteristic_disk_radius = M ** 0.5 * 30.0 | units.AU

    for alpha, label, linestyle in zip(alphas, plot_labels, line_styles):
        plt.semilogx(M,
                     viscous_timescale(stars, alpha).value_in(units.Myr),
                     lw=2,
                     ls=linestyle,
                     color='black',
                     label=label)

    global plot_colors

    plt.plot(0.5,
             8E4 / 1000000,
             'bo',
             markersize=10,
             markeredgewidth=0.0,
             label='Hartmann et al. (1998)',
             color="#638ccc")

    plt.plot([0.54, 0.54],
             [0.1, 0.3],
             'ro-',
             lw=2,
             markersize=10,
             markeredgewidth=0.0,
             label='Isella et al. (2009)',
             color="#ca5670")

    plt.legend(loc='lower right', fontsize=18, numpoints=1, ncol=2)
    ax.set_yscale('log')
    plt.xlabel(r'$\log(\mathrm{M_{\star}/M_\odot})$')
    plt.ylabel(r'$\log(\tau_{\mathrm visc})$ [Myr]')

    plt.xlim([1E-1, 1E2])
    plt.ylim([1E-2, 1E2])

    if save:
        plt.savefig('../viscous-disks/figures/redo/viscous_timescales.png')


def CDF_disk_size(N, path, snapshot, runs, Rvir, title, gas_scenario, save=False):
    """ Plot the cumulative distribution of disk sizes for different gas scenarios.
        Top panel of Figure 2 on paper.

    :param N: number of stars
    :param path: path to file to plot
    :param snapshot: moment in time in which to plot
    :param runs: number of runs to consider for mean
    :param Rvir: virial radius of the cluster
    :param title: plot title
    :param gas_scenario: gas scenario to plot
    :param save: True to save figure to png file
    """
    fig = plt.figure(figsize=(10, 8), dpi=90)
    ax = plt.subplot(111)
    global plot_colors

    gas_scenarios = gas_scenario.split(' ')
    plot_labels = ['No gas', 'Gas', 'Gas expulsion']

    alphas = [0.005, 0.01]
    line_styles = ['--', '-']
    Rc = 30.0

    for alpha, line in zip(alphas, line_styles):
        for scenario, label in zip(gas_scenarios, plot_labels):
            all_runs = []

            # Calculate mean of all runs
            for r in range(runs):
                file_path = '{0}/{1}/{2}/{3}/{4}/{5}/R{6}_{7}.hdf5'.format(path, scenario, N, alpha, Rc, r, Rvir,
                                                                           snapshot)
                stars = io.read_set_from_file(file_path, 'hdf5')
                disk_size = np.log10(2 * stars.initial_characteristic_disk_radius.value_in(units.AU))
                sorted_disk_size = np.sort(disk_size)
                all_runs.append(sorted_disk_size)

            all_runs = np.array(all_runs)
            sorted_disk_sizes = np.mean(all_runs, axis=0)
            sorted_disk_size_errors = np.std(all_runs, axis=0)
            cumulative = np.array([float(x) for x in np.arange(sorted_disk_sizes.size + 1)])

            plt.plot(np.concatenate([sorted_disk_sizes, sorted_disk_sizes[[-1]]]),
                     cumulative / len(cumulative),
                     ls=line, lw=2, color=plot_colors[scenario], label=label)

            # Add results range to no_gas curve
            if scenario == 'no_gas':
                low = np.concatenate([sorted_disk_sizes, sorted_disk_sizes[[-1]]]) - np.concatenate(
                    [sorted_disk_size_errors, sorted_disk_size_errors[[-1]]])
                high = np.concatenate([sorted_disk_sizes, sorted_disk_sizes[[-1]]]) + np.concatenate(
                    [sorted_disk_size_errors, sorted_disk_size_errors[[-1]]])
                plt.fill_betweenx(cumulative / len(cumulative), low, high, color=plot_colors[scenario], alpha='0.2')

            # Obtain disk size for isolated evoution
            all_runs_isolated = []

            for r in range(runs):
                files_path = '{0}/{1}/{2}/{3}/{4}/{5}/'.format(path, "no_gas", N, alpha, Rc, r)
                initial_state = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, 0), 'hdf5',
                                                      close_file=True)
                radii = np.log10(2 * disk_characteristic_radius(initial_state, snapshot | units.yr).value_in(units.AU))
                all_runs_isolated.append(radii)

            sorted_disk_sizes_iso = np.sort(np.mean(all_runs_isolated, axis=0))
            cumulative_iso = np.array([float(x) for x in np.arange(sorted_disk_sizes_iso.size + 1)])

            plt.plot(np.concatenate([sorted_disk_sizes_iso, sorted_disk_sizes_iso[[-1]]]),
                     cumulative_iso / len(cumulative_iso),
                     ls=line,
                     lw=1, color='black', label=label)

    plt.xlim([1.0, 3.5])
    plt.ylim([0.0, 1.0])
    if title == 1:
        plt.title('Disk diameter')

    plt.legend([NoGasObject(), GasObject(), GasExpObject(), IsolatedObject()],
               ['No gas', 'Gas', 'Gas expulsion', 'Isolated evolution,\nno gas'],
               handler_map={NoGasObject: NoGasObjectHandler(), GasObject: GasObjectHandler(),
                            GasExpObject: GasExpObjectHandler(), IsolatedObject: IsolatedObjectHandler()},
               loc='upper left', fontsize=20)

    plt.xlabel(r'$\log{d} \left[\mathrm{au}\right]$')
    plt.ylabel(r'$f < d$')
    if save:
        plt.savefig(
            '../viscous-disks/figures/redo/CDF_size_R0{0}_N{1}_a0{2}_{3}Myr.png'.format(str(Rvir).split('.')[1], N,
                                                                                        str(alpha).split('.')[1],
                                                                                        int(snapshot / 1000000)))


def CDF_disk_mass(N, path, snapshot, runs, Rvir, title, gas_scenario, save=False):
    """ Plot the cumulative distribution of disk masses for different gas scenarios.
        Middle panel of Figure 2 on paper.

    :param N: number of stars
    :param path: path to file to plot
    :param snapshot: moment in time in which to plot
    :param runs: number of runs to consider for mean
    :param Rvir: virial radius of the cluster
    :param title: plot title
    :param gas_scenario: gas scenario to plot
    :param save: True to save figure to png file
    """
    fig = plt.figure(figsize=(10, 8), dpi=90)
    ax = plt.subplot(111)
    global plot_colors

    gas_scenarios = gas_scenario.split(' ')
    plot_labels = ['No gas', 'Gas', 'Gas expulsion']

    alphas = [0.005, 0.01]
    line_styles = ['--', '-']
    Rc = 30.0

    for alpha, line in zip(alphas, line_styles):
        for scenario, label in zip(gas_scenarios, plot_labels):
            all_runs = []

            # Calculate mean of all runs
            for r in range(runs):
                file_path = '{0}/{1}/{2}/{3}/{4}/{5}/R{6}_{7}.hdf5'.format(path, scenario, N, alpha, Rc, r, Rvir,
                                                                           snapshot)
                stars = io.read_set_from_file(file_path, 'hdf5')
                disk_masses = np.log10(disk_mass(stars, snapshot | units.yr).value_in(units.MJupiter))
                sorted_disk_mass = np.sort(disk_masses)
                all_runs.append(sorted_disk_mass)

                files_path = '{0}/{1}/{2}/{3}/{4}/{5}/'.format(path, "no_gas", N, alpha, Rc, r)
                initial_state = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, 0), 'hdf5',
                                                      close_file=True)

            all_runs = np.array(all_runs)
            sorted_disk_masses = np.mean(all_runs, axis=0)
            sorted_disk_masses_errors = np.std(all_runs, axis=0)
            cumulative = np.array([float(x) for x in np.arange(sorted_disk_masses.size + 1)])

            ax.plot(np.concatenate([sorted_disk_masses, sorted_disk_masses[[-1]]]), cumulative / len(cumulative),
                    ls=line, lw=2, color=plot_colors[scenario], label=label)

            # Obtain mases of isolated disks
            all_runs_isolated = []

            for r in range(runs):
                files_path = '{0}/{1}/{2}/{3}/{4}/{5}/'.format(path, "no_gas", N, alpha, Rc, r)
                initial_state = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, 0), 'hdf5',
                                                      close_file=True)
                masses = np.log10(disk_mass(initial_state, snapshot | units.yr).value_in(units.MJupiter))
                all_runs_isolated.append(masses)

            sorted_disk_masses_iso = np.sort(np.mean(all_runs_isolated, axis=0))
            cumulative_iso = np.array([float(x) for x in np.arange(sorted_disk_masses_iso.size + 1)])

            plt.plot(np.concatenate([sorted_disk_masses_iso, sorted_disk_masses_iso[[-1]]]),
                     cumulative_iso / len(cumulative_iso),
                     ls=line,
                     lw=1, color='black', label=label)

            if scenario == 'no_gas':
                low = np.concatenate([sorted_disk_masses, sorted_disk_masses[[-1]]]) - np.concatenate(
                    [sorted_disk_masses_errors, sorted_disk_masses_errors[[-1]]])
                high = np.concatenate([sorted_disk_masses, sorted_disk_masses[[-1]]]) + np.concatenate(
                    [sorted_disk_masses_errors, sorted_disk_masses_errors[[-1]]])
                plt.fill_betweenx(cumulative / len(cumulative), low, high, color=plot_colors[scenario], alpha='0.2')

    plt.xlim([-2.0, 3.5])
    plt.ylim([0.0, 1.0])

    plt.legend([NoGasObject(), GasObject(), GasExpObject(), IsolatedObject()],
               ['No gas', 'Gas', 'Gas expulsion', 'Isolated evolution,\nno gas'],
               handler_map={NoGasObject: NoGasObjectHandler(), GasObject: GasObjectHandler(),
                            GasExpObject: GasExpObjectHandler(), IsolatedObject: IsolatedObjectHandler()},
               loc='upper left', fontsize=20)

    plt.xlabel(r'$\log{M} \left[\mathrm{M_{Jup}}\right]$')
    plt.ylabel(r'$f < M$')
    if save:
        plt.savefig(
            '../viscous-disks/figures/redo/CDF_mass_R0{0}_N{1}_a0{2}_{3}Myr.png'.format(str(Rvir).split('.')[1], N,
                                                                                        str(alpha).split('.')[1],
                                                                                        int(snapshot / 1000000)))


def CDF_accretion_rates(N, path, snapshot, runs, Rvir, title, gas_scenario, save=False):
    """ Plot the cumulative distribution of disk stellar accretion rates for different gas scenarios.
        Bottom panel of Figure 2 on paper.

    :param N: number of stars
    :param path: path to file to plot
    :param snapshot: moment in time in which to plot
    :param runs: number of runs to consider for mean
    :param Rvir: virial radius of the cluster
    :param title: plot title
    :param gas_scenario: gas scenario to plot
    :param save: True to save figure to png file
    """
    fig = plt.figure(figsize=(10, 8), dpi=90)
    ax = plt.subplot(111)
    global plot_colors

    gas_scenarios = gas_scenario.split(' ')
    plot_labels = ['No gas', 'Gas', 'Gas expulsion']

    alphas = [0.005, 0.01]
    line_styles = ['--', '-']
    Rc = 30.0

    for alpha, line in zip(alphas, line_styles):
        for scenario, label in zip(gas_scenarios, plot_labels):
            all_runs = []
            all_runs_isolated = []

            # Calculate mean of all runs
            for r in range(runs):
                file_path = '{0}/{1}/{2}/{3}/{4}/{5}/R{6}_{7}.hdf5'.format(path, scenario, N, alpha, Rc, r, Rvir,
                                                                           snapshot)
                stars = io.read_set_from_file(file_path, 'hdf5')
                rates = np.log10(disk_accretion_rate(stars, snapshot | units.yr).value_in(units.MJupiter / units.yr))
                sorted_acc_rates = np.sort(rates)
                all_runs.append(sorted_acc_rates)

            all_runs = np.array(all_runs)
            sorted_acc_rates = np.mean(all_runs, axis=0)
            sorted_acc_rates_errors = np.std(all_runs, axis=0)
            cumulative = np.array([float(x) for x in np.arange(sorted_acc_rates.size + 1)])
            plt.plot(np.concatenate([sorted_acc_rates, sorted_acc_rates[[-1]]]),
                     cumulative / len(cumulative), ls=line,
                     lw=2, color=plot_colors[scenario], label=label)

            if scenario == 'no_gas':
                low = np.concatenate([sorted_acc_rates, sorted_acc_rates[[-1]]]) - np.concatenate(
                    [sorted_acc_rates_errors, sorted_acc_rates_errors[[-1]]])
                high = np.concatenate([sorted_acc_rates, sorted_acc_rates[[-1]]]) + np.concatenate(
                    [sorted_acc_rates_errors, sorted_acc_rates_errors[[-1]]])
                plt.fill_betweenx(cumulative / len(cumulative), low, high, color=plot_colors[scenario], alpha='0.2')

            # Get values for isolated disks
            all_runs_isolated = []

            for r in range(runs):
                files_path = '{0}/{1}/{2}/{3}/{4}/{5}/'.format(path, "no_gas", N, alpha, Rc, r)
                initial_state = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, 0), 'hdf5',
                                                      close_file=True)
                accs = np.log10(
                    disk_accretion_rate(initial_state, snapshot | units.yr).value_in(units.MJupiter / units.yr))
                all_runs_isolated.append(accs)

            sorted_disk_accs_iso = np.sort(np.mean(all_runs_isolated, axis=0))
            cumulative_iso = np.array([float(x) for x in np.arange(sorted_disk_accs_iso.size + 1)])

            plt.plot(np.concatenate([sorted_disk_accs_iso, sorted_disk_accs_iso[[-1]]]),
                     cumulative_iso / len(cumulative_iso),
                     ls=line,
                     lw=1, color='black', label=label)

    plt.xlim([-8.0, -4])
    plt.ylim([0.0, 1.0])
    if title == 1:
        plt.title('Disk mass accretion rate')

    plt.legend([NoGasObject(), GasObject(), GasExpObject(), IsolatedObject()],
               ['No gas', 'Gas', 'Gas expulsion', 'Isolated evolution,\nno gas'],
               handler_map={NoGasObject: NoGasObjectHandler(), GasObject: GasObjectHandler(),
                            GasExpObject: GasExpObjectHandler(), IsolatedObject: IsolatedObjectHandler()},
               loc='upper left', fontsize=20)

    plt.xlabel(r'$\log{\dot{M_{\star}}} \left[\mathrm{M_{Jup} / yr}\right]$', labelpad=3)
    plt.ylabel(r'$f < \dot{M_{\star}}$')
    if save:
        plt.savefig(
            '../viscous-disks/figures/redo/CDF_acc_rate_R0{0}_N{1}_a0{2}_{3}Myr.png'.format(str(Rvir).split('.')[1], N,
                                                                                            str(alpha).split('.')[1],
                                                                                            int(snapshot / 1000000)))


def plot_normalized_disk_radii(N, path, runs, Rvir, title, gas_scenario, save=False):
    """ Plot the normalized disk sizes for different gas scenarios.
        Top panel of Figure 3 on paper.

    :param N: number of stars
    :param path: path to file to plot
    :param snapshot: moment in time in which to plot
    :param runs: number of runs to consider for mean
    :param Rvir: virial radius of the cluster
    :param title: plot title
    :param gas_scenario: gas scenario to plot
    :param save: True to save figure to png file
    """
    fig = plt.figure(figsize=(10, 8), dpi=90)
    ax = plt.subplot(111)

    times = np.arange(0, 2050000, 50000)
    gas_scenarios = gas_scenario.split(' ')
    plot_labels = ['No gas', 'Gas', 'Gas expulsion']
    line_styles = ['-', '--', ':', ':']
    markers = ['o', 'x', '^', 'p']

    alpha = 0.01
    Rc = 30.0

    for scenario, label, line_style, marker in zip(gas_scenarios, plot_labels, line_styles, markers):
        all_runs = []

        # Get mean of all runs
        for r in range(runs):
            files_path = '{0}/{1}/{2}/{3}/{4}/{5}/'.format(path, scenario, N, alpha, Rc, r)
            initial_state = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, 0), 'hdf5', close_file=True)

            all_times = []

            for t in times:
                actual_file = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, t), 'amuse',
                                                    close_file=True)
                time = t | units.yr

                # Isolated value is calculated with equations, actual value is read from simuation snapshot
                isolated_radii = disk_characteristic_radius(initial_state, time)
                actual_radii = disk_characteristic_radius(actual_file, time)
                all_times.append(np.mean(np.divide(actual_radii, isolated_radii)))

            all_runs.append(all_times)

        all_runs = np.array(all_runs)
        all_mean = np.mean(all_runs, axis=0)
        all_errors = np.std(all_runs, axis=0)

        plt.plot(times / 1E6, all_mean, line_styles[0], color=plot_colors[scenario], lw=3, label=label)

        if scenario == 'no_gas':
            low = all_mean - all_errors
            high = all_mean + all_errors
            plt.fill_between(times / 1E6, low, high, color=plot_colors[scenario], alpha='0.2')

    # Same thing for different alpha
    alpha = 0.005

    for scenario, label, line_style, marker in zip(gas_scenarios, plot_labels, line_styles, markers):
        all_runs = []
        for r in range(runs):
            files_path = '{0}/{1}/{2}/{3}/{4}/{5}/'.format(path, scenario, N, alpha, Rc, r)
            initial_state = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, 0), 'hdf5', close_file=True)

            all_times = []

            for t in times:
                actual_file = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, t), 'amuse',
                                                    close_file=True)
                time = t | units.yr
                isolated_radii = disk_characteristic_radius(initial_state, time)
                actual_radii = disk_characteristic_radius(actual_file, time)
                all_times.append(np.mean(np.divide(actual_radii, isolated_radii)))

            all_runs.append(all_times)

        all_runs = np.array(all_runs)
        all_mean = np.mean(all_runs, axis=0)
        all_errors = np.std(all_runs, axis=0)

        plt.plot(times / 1E6, all_mean, line_styles[1], color=plot_colors[scenario], lw=3)  # , label=label)

        if scenario == 'no_gas':
            low = all_mean - all_errors
            high = all_mean + all_errors
            plt.fill_between(times / 1E6, low, high, color=plot_colors[scenario], alpha='0.2')

    ax.set_xlim([0.0, 2.0])
    ax.set_ylim([0.88, 1.0])
    plt.plot((1.0, 1.0), (0.88, 1.0), 'k--')

    text(0.48, 0.28, r'Gas expulsion onset (1.0 Myr)',
         horizontalalignment='center',
         verticalalignment='center',
         transform=ax.transAxes,
         rotation=90,
         size=20)

    if title == 1:
        fig.subplots_adjust(top=0.8)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                  ncol=3, fancybox=True, shadow=True, numpoints=1)
        plt.suptitle(r'{\Large Normalized disk size}')

    plt.legend([NoGasObject(), GasObject(), GasExpObject()], ['No gas', 'Gas', 'Gas expulsion'],
               handler_map={NoGasObject: NoGasObjectHandler(), GasObject: GasObjectHandler(),
                            GasExpObject: GasExpObjectHandler()}, loc='lower left', fontsize=20)

    plt.xlabel('Time [Myr]')
    ax.set_ylabel(r'$R_{disk} / R_{isolated}$', fontsize=26)
    if save:
        plt.savefig(
            '../viscous-disks/figures/redo/Normalized_size_R0{0}_N{1}_a0{2}.png'.format(str(Rvir).split('.')[1], N,
                                                                                        str(alpha).split('.')[1]))


def plot_normalized_disk_mass(N, path, runs, Rvir, title, gas_scenario, save=False):
    """ Plot the normalized disk masses for different gas scenarios.
        Middle panel of Figure 3 on paper.

    :param N: number of stars
    :param path: path to file to plot
    :param snapshot: moment in time in which to plot
    :param runs: number of runs to consider for mean
    :param Rvir: virial radius of the cluster
    :param title: plot title
    :param gas_scenario: gas scenario to plot
    :param save: True to save figure to png file
    """
    fig = plt.figure(figsize=(10, 8), dpi=90)
    ax = plt.subplot(111)
    global plot_colors

    times = np.arange(0, 2050000, 50000)
    gas_scenarios = gas_scenario.split(' ')
    plot_labels = ['No gas', 'Gas', 'Gas expulsion']
    line_styles = ['-', '--', ':', ':']
    markers = ['o', 'x', '^', 'p']

    alpha = 0.01
    Rc = 30.0

    for scenario, label, line_style, marker in zip(gas_scenarios, plot_labels, line_styles, markers):
        all_runs = []

        # Get mean of all runs
        for r in range(runs):
            files_path = '{0}/{1}/{2}/{3}/{4}/{5}/'.format(path, scenario, N, alpha, Rc, r)
            initial_state = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, 0), 'hdf5', close_file=True)

            all_times = []

            for t in times:
                actual_file = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, t), 'amuse',
                                                    close_file=True)
                time = t | units.yr

                # Isolated values are calculated from equations, actual values are read from simulation snapshot
                isolated_mass = disk_mass(initial_state, time)
                actual_mass = disk_mass(actual_file, time)
                all_times.append(np.mean(np.divide(actual_mass, isolated_mass)))

            all_runs.append(all_times)

        all_runs = np.array(all_runs)
        all_mean = np.mean(all_runs, axis=0)
        all_errors = np.std(all_runs, axis=0)

        plt.plot(times / 1E6, all_mean, line_styles[0], color=plot_colors[scenario], lw=3, label=label)

        if scenario == 'no_gas':
            low = all_mean - all_errors
            high = all_mean + all_errors
            plt.fill_between(times / 1E6, low, high, color=plot_colors[scenario], alpha='0.2')

    # Same thing for different alpha
    alpha = 0.005

    for scenario, label, line_style, marker in zip(gas_scenarios, plot_labels, line_styles, markers):
        all_runs = []
        for r in range(runs):
            files_path = '{0}/{1}/{2}/{3}/{4}/{5}/'.format(path, scenario, N, alpha, Rc, r)
            initial_state = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, 0), 'hdf5', close_file=True)

            all_times = []

            for t in times:
                actual_file = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, t), 'amuse',
                                                    close_file=True)
                time = t | units.yr
                isolated_mass = disk_mass(initial_state, time)
                actual_mass = disk_mass(actual_file, time)
                all_times.append(np.mean(np.divide(actual_mass, isolated_mass)))

            all_runs.append(all_times)

        all_runs = np.array(all_runs)
        all_mean = np.mean(all_runs, axis=0)
        all_errors = np.std(all_runs, axis=0)

        plt.plot(times / 1E6, all_mean, line_styles[1], color=plot_colors[scenario], lw=3, label=label)

        if scenario == 'no_gas':
            low = all_mean - all_errors
            high = all_mean + all_errors
            plt.fill_between(times / 1E6, low, high, color=plot_colors[scenario], alpha='0.2')

    ax.set_xlim([0.0, 2.0])
    ax.set_ylim([0.88, 1.0])
    plt.plot((1.0, 1.0), (0.88, 1.0), 'k--')
    text(0.48, 0.28, r'Gas expulsion onset (1.0 Myr)',
         horizontalalignment='center',
         verticalalignment='center',
         transform=ax.transAxes,
         rotation=90,
         size=20)

    if title == 1:
        fig.subplots_adjust(top=0.8)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                  ncol=3, fancybox=True, shadow=True, numpoints=1)
        plt.suptitle(r'{\Large Normalized disk mass}')

    plt.legend([NoGasObject(), GasObject(), GasExpObject()], ['No gas', 'Gas', 'Gas expulsion'],
               handler_map={NoGasObject: NoGasObjectHandler(), GasObject: GasObjectHandler(),
                            GasExpObject: GasExpObjectHandler()}, loc='lower left', fontsize=20)

    plt.xlabel('Time [Myr]')
    ax.set_ylabel(r'$M_{disk} / M_{isolated}$', fontsize=26)

    if save:
        plt.savefig(
            '../viscous-disks/figures/redo/Normalized_mass_R0{0}_N{1}_a0{2}.png'.format(str(Rvir).split('.')[1], N,
                                                                                        str(alpha).split('.')[1]))


def plot_normalized_disk_acc_rate(N, path, runs, Rvir, title, gas_scenario, save=False):
    """ Plot the normalized disk stellar accretion rates for different gas scenarios.
        Bottom panel of Figure 3 on paper.

    :param N: number of stars
    :param path: path to file to plot
    :param snapshot: moment in time in which to plot
    :param runs: number of runs to consider for mean
    :param Rvir: virial radius of the cluster
    :param title: plot title
    :param gas_scenario: gas scenario to plot
    :param save: True to save figure to png file
    """
    fig = plt.figure(figsize=(10, 8), dpi=90)
    ax = plt.subplot(111)
    global plot_colors

    times = np.arange(0, 2050000, 50000)
    gas_scenarios = gas_scenario.split(' ')
    plot_labels = ['No gas', 'Gas', 'Gas expulsion']
    line_styles = ['-', '--', ':', ':']
    markers = ['o', 'x', '^', 'p']

    alpha = 0.01
    Rc = 30.0

    for scenario, label, line_style, marker in zip(gas_scenarios, plot_labels, line_styles, markers):
        all_runs = []

        # Get mean of all runs
        for r in range(runs):
            files_path = '{0}/{1}/{2}/{3}/{4}/{5}/'.format(path, scenario, N, alpha, Rc, r)
            initial_state = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, 0), 'hdf5', close_file=True)

            all_times = []

            for t in times:
                actual_file = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, t), 'amuse',
                                                    close_file=True)
                time = t | units.yr

                # Isolated values are calculated from equations, actual values are read from simulation snapshot
                isolated_acc_rate = disk_accretion_rate(initial_state, time)
                actual_acc_rate = disk_accretion_rate(actual_file, time)
                all_times.append(np.mean(np.divide(actual_acc_rate, isolated_acc_rate)))

            all_runs.append(all_times)

        all_runs = np.array(all_runs)
        all_mean = np.mean(all_runs, axis=0)
        all_errors = np.std(all_runs, axis=0)

        plt.plot(times / 1E6, all_mean, line_styles[0], color=plot_colors[scenario], lw=3, label=label)

        if scenario == 'no_gas':
            low = all_mean - all_errors
            high = all_mean + all_errors
            plt.fill_between(times / 1E6, low, high, color=plot_colors[scenario], alpha='0.2')

    # Same thing, different alpha
    alpha = 0.005

    for scenario, label, line_style, marker in zip(gas_scenarios, plot_labels, line_styles, markers):
        all_runs = []
        for r in range(runs):
            files_path = '{0}/{1}/{2}/{3}/{4}/{5}/'.format(path, scenario, N, alpha, Rc, r)
            initial_state = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, 0), 'hdf5', close_file=True)

            all_times = []

            for t in times:
                actual_file = io.read_set_from_file(files_path + 'R{0}_{1}.hdf5'.format(Rvir, t), 'amuse',
                                                    close_file=True)
                time = t | units.yr
                isolated_acc_rate = disk_accretion_rate(initial_state, time)
                actual_acc_rate = disk_accretion_rate(actual_file, time)
                all_times.append(np.mean(np.divide(actual_acc_rate, isolated_acc_rate)))

            all_runs.append(all_times)

        all_runs = np.array(all_runs)
        all_mean = np.mean(all_runs, axis=0)
        all_errors = np.std(all_runs, axis=0)

        plt.plot(times / 1E6, all_mean, line_styles[1], color=plot_colors[scenario], lw=3, label=label)

        if scenario == 'no_gas':
            low = all_mean - all_errors
            high = all_mean + all_errors
            plt.fill_between(times / 1E6, low, high, color=plot_colors[scenario], alpha='0.2')

    ax.set_xlim([0.0, 2.0])
    ax.set_ylim([0.88, 1.0])
    plt.plot((1.0, 1.0), (0.88, 1.0), 'k--')
    text(0.48, 0.28, r'Gas expulsion onset (1.0 Myr)',
         horizontalalignment='center',
         verticalalignment='center',
         transform=ax.transAxes,
         rotation=90,
         size=20)

    if title == 1:
        fig.subplots_adjust(top=0.8)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                  ncol=3, fancybox=True, shadow=True, numpoints=1)
        plt.suptitle(r'{\Large Normalized disk accretion rate}')

    plt.legend([NoGasObject(), GasObject(), GasExpObject()], ['No gas', 'Gas', 'Gas expulsion'],
               handler_map={NoGasObject: NoGasObjectHandler(), GasObject: GasObjectHandler(),
                            GasExpObject: GasExpObjectHandler()}, loc='lower left', fontsize=20)

    plt.xlabel('Time [Myr]')
    ax.set_ylabel(r'$\dot{M}_{\star} / \dot{M_{\star}}_{isolated}$', fontsize=26)

    if save:
        plt.savefig(
            '../viscous-disks/figures/redo/Normalized_acc_rate_R0{0}_N{1}_a0{2}.png'.format(str(Rvir).split('.')[1], N,
                                                                                            str(alpha).split('.')[1]))


def main(N, path, alpha, run, Rvir, gas_scenario, snapshot, title):
    snapshot = int(snapshot * 1000000)

    # Figure 1 on paper
    plot_initial_viscous_timescales(save=True)

    # Figure 2 on paper
    CDF_disk_size(N, path, snapshot, 5, Rvir, title, gas_scenario, save=True)
    CDF_disk_mass(N, path, snapshot, 5, Rvir, title, gas_scenario, save=True)
    CDF_accretion_rates(N, path, snapshot, 5, Rvir, title, gas_scenario, save=True)

    # Figure 3 on paper
    plot_normalized_disk_radii(N, path, 3, Rvir, title, gas_scenario, save=False)
    plot_normalized_disk_mass(N, path, 3, Rvir, title, gas_scenario, save=False)
    plot_normalized_disk_acc_rate(N, path, 3, Rvir, title, gas_scenario, save=False)

    plt.show()


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()

    # Simulation
    result.add_option("-p", dest="path", type="string", default="redo_runs2/",
                      help="path to files [%default]")
    result.add_option("-N", dest="N", type="int", default="1250",
                      help="number of stars [%default]")
    result.add_option("-a", dest="alpha", type="float", default="0.01",
                      help="alpha [%default]")
    result.add_option("-r", dest="run", type="int", default=0,
                      help="Run to be plotted [%default]")
    result.add_option("-R", dest="Rvir", type="float", default=0.5,
                      help="Virial radius to be plotted [%default]")
    result.add_option("-s", dest="snapshot", type="float", default=2.0,
                      help="Time to plot [%default]")
    result.add_option("-g", dest="gas_scenario", type="string", default="no_gas gas gas_exp",
                      help="Gas scenario [%default]")
    result.add_option("-t", dest="title", type="int", default="0",
                      help="Title and legends for plot or not (0=no, 1=yes) [%default]")
    return result


if __name__ in ('__main__', '__plot__'):
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)
