""" Script used to generate figures 4, 5, and 6 on the paper
    For the rest of the figures, see plot_disks.py
"""

from amuse.lab import *
from amuse.datamodel.particle_attributes import LagrangianRadii
from amuse import io

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import matplotlib.patches as patches
from math import pi
from scipy import stats
import matplotlib.lines as mlines

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 24, })
rc('text', usetex=True)
rc('axes', labelsize=26)    # fontsize of the x and y labels
mpl.rcParams['hatch.linewidth'] = 6

global plot_colors
plot_colors = {'Trapezium': "#638ccc", 'Lupus': "#ca5670", 'Cham': "#c57c3c", 'Orionis': "#72a555",
               'UpperSco': "#ab62c0"}


# To manage plot legends
class ViscousObject(object):
    pass


class RamPressureObject(object):
    pass


class DynTruncObject(object):
    pass


class SupernovaeObject(object):
    pass


class WindsObject(object):
    pass


class EPhotoevapObject(object):
    pass


class TrapeziumObject(object):
    pass


class LupusObject(object):
    pass


class ChamObject(object):
    pass


class OrionisObject(object):
    pass


class UpperScoObject(object):
    pass


class ViscousObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5], [0.5 * height, 0.5 * height], lw=4, color="#E0F3F8")
        handlebox.add_artist(l1)
        return [l1]


class RamPressureObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = patches.Rectangle(  # External photoevap
            (x0 - 1, y0 + width - 32),  # (x,y)
            1.2 * width,  # width
            1.2 * height,  # height
            fill=False,
            facecolor="#FEE090",
            edgecolor="#FEE090",
            alpha=1.0,  # 0.2,
            label="External photoevaporation",
            hatch="\\",
        )
        handlebox.add_artist(l1)
        return [l1]

class DynTruncObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5], [0.5 * height, 0.5 * height], lw=4, color="#4575B4")
        handlebox.add_artist(l1)
        return [l1]


class SupernovaeObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5], [0.5 * height, 0.5 * height], lw=4, color="#D73027")
        handlebox.add_artist(l1)
        return [l1]


class WindsObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5], [0.5 * height, 0.5 * height], lw=4, color="#FC8D59")
        handlebox.add_artist(l1)
        return [l1]


class EPhotoevapObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = patches.Rectangle(  # External photoevap
        (x0 - 1, y0 + width - 32),  # (x,y)
        1.2 * width,  # width
        1.2 * height,  # height
        fill=False,
        facecolor="#91bfdb",
        edgecolor="#91bfdb",
        alpha=1.0,  # 0.2,
        label="External photoevaporation",
        hatch="/",
    )
        handlebox.add_artist(l1)
        return [l1]


class TrapeziumObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5], [0.7 * height, 0.7 * height], lw=3, color=plot_colors['Trapezium'])
        l2 = mlines.Line2D([x0, y0 + width + 5], [0.2 * height, 0.2 * height], linestyle='--', lw=3,
                           color=plot_colors['Trapezium'])
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class LupusObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5], [0.7 * height, 0.7 * height], lw=3, color=plot_colors['Lupus'])
        l2 = mlines.Line2D([x0, y0 + width + 5], [0.2 * height, 0.2 * height], linestyle='--', lw=3,
                           color=plot_colors['Lupus'])
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class ChamObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5], [0.7 * height, 0.7 * height], lw=3, color=plot_colors['Cham'])
        l2 = mlines.Line2D([x0, y0 + width + 5], [0.2 * height, 0.2 * height], linestyle='--', lw=3,
                           color=plot_colors['Cham'])
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class OrionisObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5], [0.7 * height, 0.7 * height], lw=3, color=plot_colors['Orionis'])
        l2 = mlines.Line2D([x0, y0 + width + 5], [0.2 * height, 0.2 * height], linestyle='--', lw=3,
                           color=plot_colors['Orionis'])
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


class UpperScoObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 5], [0.7 * height, 0.7 * height], lw=3, color=plot_colors['UpperSco'])
        l2 = mlines.Line2D([x0, y0 + width + 5], [0.2 * height, 0.2 * height], linestyle='--', lw=3,
                           color=plot_colors['UpperSco'])
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        return [l1, l2]


def disk_mass(star, t, gamma):
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


def n_stars(stars, Rv):
    """ Return number of stars inside give radius.

    :param stars: all stars
    :param Rv: radius
    :return: number of stars inside Rv
    """
    x = stars.x.value_in(units.parsec)
    y = stars.y.value_in(units.parsec)
    z = stars.z.value_in(units.parsec)

    N = 0

    for sx, sy, sz in zip(x, y, z):
        r = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
        if r < Rv:
            N += 1
    return N


def KStest(data1, data2):
    """ Return scipy's kstest result between data1 and data2 """
    return stats.ks_2samp(data1, data2)


def find_model_size(path, nruns, data_ages, data_densities, log=True):
    """ For each observed region, find the closest simulation regarding disk size.
        See section 3.4 of paper for details.

    :param path: path to simulation files
    :param nruns: number of runs to be used for mean
    :param data_ages: estimated ages of the observed regions
    :param data_densities: estimated stellar densities of the observed regions
    :param log: log scale for data
    :return: best N and alpha for each observed region
    """
    best_fitsN = {}
    best_fitsAlpha = {}

    Ns = [25, 50, 100, 125, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    alphas = [0.01, 0.002, 0.005, 0.007, 0.0001]
    Rvir = 0.5
    Rc = 30.0

    original_path = path

    for cluster in data_ages:
        a = 0.0001
        print("{0}, d={1}".format(cluster, data_densities[cluster]))
        similar = []
        t = data_ages[cluster]
        if data_ages[cluster] <= 2:
            Ns = [25, 50, 100, 125, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
            path = original_path
        else:
            Ns = [25, 50, 100, 125, 250, 500, 750, 1000, 1250, 1500]#, 1750, 2000]
            path = 'long_runs'
        d = data_densities[cluster]

        bestN = 0
        smallest_diff = 1000

        for N in Ns:  # Find N for closest stellar density
            if data_ages[cluster] <= 2:
                file_path = '{0}/no_gas/{1}/{2}/{3}/0/R0.5_{4}.hdf5'.format(path, N, a, Rc, int(t * 1E6))
            else:
                file_path = '{0}/no_gas/{1}/{2}/0/R0.5_{3}.hdf5'.format(path, N, a, int(t * 1E6))
            stars = io.read_set_from_file(file_path, 'hdf5')
            converter = nbody_system.nbody_to_si(stars.stellar_mass.sum() + stars.initial_disk_mass.sum(),
                                                 Rvir | units.parsec)
            radii = LagrangianRadii(stars, converter, mf=[0.5])
            hmr = radii[0].value_in(units.parsec)
            n = float(n_stars(stars, hmr)) / ((4 / 3) * np.pi * hmr ** 3)[0]
            diff = np.absolute(n - data_densities[cluster])

            if diff < smallest_diff:
                bestN = N
                smallest_diff = diff

        best_fitsN[cluster] = bestN

        bestp = 1E-50
        bestAlpha = 0

        for alpha in alphas:
            if data_ages[cluster] <= 2:
                file_path = '{0}/no_gas/{1}/{2}/{3}/0/R0.5_{4}.hdf5'.format(path, N, alpha, Rc, int(t * 1E6))
            else:
                file_path = '{0}/no_gas/{1}/{2}/0/R0.5_{3}.hdf5'.format(path, N, alpha, int(t * 1E6))

            stars = io.read_set_from_file(file_path, 'hdf5')

            if log:
                disk_size = np.log10(2.95 * 2 * stars.initial_characteristic_disk_radius.value_in(units.AU))
            else:
                disk_size = 2.95 * 2 * stars.initial_characteristic_disk_radius.value_in(units.AU)

            sorted_disk_size = np.sort(disk_size)

            if cluster == 'Trapezium':
                # Trapezium data (Vicente & Alves 2005)
                lines = open('data/Trapezium Cluster/Vicente_Alves_2005_Table3.txt', 'r').readlines()
                trapezium_sizes = []

                for line in (line for line in lines if not line.startswith('#')):
                    data = line.split()[8]
                    trapezium_sizes.append(float(data))

                trapezium_sizes = np.array(trapezium_sizes)
                if log:
                    sorted_trapezium_sizes = np.sort(np.log10(trapezium_sizes[trapezium_sizes > 100.]))
                else:
                    sorted_trapezium_sizes = np.sort(trapezium_sizes[trapezium_sizes > 100.])
                p = KStest(sorted_disk_size, sorted_trapezium_sizes)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            elif cluster == 'Lupus':
                lines = open('data/Lupus Complex/Ansdell_et_al_2018', 'r').readlines()
                lupus_sizes = []

                for line in (line for line in lines if not line.startswith('#')):
                    data = line.split()[0]
                    lupus_sizes.append(2 * float(data))#.split('+')[0]))  # Radius in AU

                sorted_lupus_disk_sizes_au = np.sort(np.log10(lupus_sizes))

                p = KStest(sorted_disk_size, sorted_lupus_disk_sizes_au)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            elif cluster == 'Cham':
                lines = open('data/ChamaeleonI/size', 'r').readlines()
                cham_sizes_arsec = []

                for line in (line for line in lines if not line.startswith('#')):
                    a = line.split()[7]
                    b = line.split()[8]
                    if a > b:
                        cham_sizes_arsec.append(float(a))
                    else:
                        cham_sizes_arsec.append(float(b))

                cham_sizes_arsec = np.array(cham_sizes_arsec)
                cham_sizes_arsec = cham_sizes_arsec[cham_sizes_arsec > 0.0]

                cham_distance_pc = 160
                cham_distance_au = 2.0626 * pow(10, 5) * cham_distance_pc
                cham_sizes_au = (pi / 180) * (cham_sizes_arsec / 3600.) * cham_distance_au
                if log:
                    cham_sorted_disk_sizes = np.sort(np.log10(cham_sizes_au))
                else:
                    cham_sorted_disk_sizes = np.sort(cham_sizes_au)

                p = KStest(sorted_disk_size, cham_sorted_disk_sizes)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            elif cluster == 'Orionis':
                lines = open('data/sigmaOrionis', 'r').readlines()
                sOrionis_sizes_au = []

                for line in (line for line in lines if not line.startswith('#')):
                    a = line.split()[1]
                    sOrionis_sizes_au.append(float(a))

                if log:
                    sOrionis_sorted_disk_sizes = np.sort(np.array(np.log10(sOrionis_sizes_au)))
                else:
                    sOrionis_sorted_disk_sizes = np.sort(np.array(np.log10(sOrionis_sizes_au)))
                p = KStest(sorted_disk_size, sOrionis_sorted_disk_sizes)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            elif cluster == 'UpperSco':
                lines = open('data/UpperSco/size', 'r').readlines()
                uppersco_sizes_arsec, errors_arsec = [], []

                for line in (line for line in lines if not line.startswith('#')):
                    a = line.split()[7]
                    b = line.split()[8]
                    uppersco_sizes_arsec.append(float(a))
                    errors_arsec.append(float(b))

                uppersco_sizes_arsec = np.array(uppersco_sizes_arsec)
                uppersco_sizes_arsec = uppersco_sizes_arsec[uppersco_sizes_arsec > 0.0]

                uppersco_distance_pc = 145
                uppersco_distance_au = 2.0626 * pow(10, 5) * uppersco_distance_pc
                uppersco_sizes_au = (pi / 180) * (uppersco_sizes_arsec / 3600.) * uppersco_distance_au
                if log:
                    uppersco_sorted_disk_sizes = np.sort(np.log10(uppersco_sizes_au))
                else:
                    uppersco_sorted_disk_sizes = np.sort(uppersco_sizes_au)
                p = KStest(sorted_disk_size, uppersco_sorted_disk_sizes)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

        best_fitsAlpha[cluster] = bestAlpha

    return best_fitsN, best_fitsAlpha


def disk_mass_within_radius(star, t, R, gamma):
    """ Return the mass of a circumstellar disk on a given moment inside a given radius.

    :param star: star with the disk
    :param t: moment of time at which the disk mass is to be evaluated
    :param R: radius inside which the mass is to be evaluated
    :return: disk mass at moment t inside radius R
    """
    R /= star.initial_characteristic_disk_radius
    R **= 2-gamma
    R /= 1+(t-star.last_encounter)/star.viscous_timescale
    return -disk_mass(star, t, gamma)*np.expm1(-R)


def find_model_mass(path, nruns, data_ages, data_densities, log=True):
    """ For each observed region, find the closest simulation regarding disk mass.
        See section 3.4 of paper for details.

    :param path: path to simulation files
    :param nruns: number of runs to be used for mean
    :param data_ages: estimated ages of the observed regions
    :param data_densities: estimated stellar densities of the observed regions
    :param log: log scale for data
    :return: best N and alpha for each observed region
    """
    best_fitsN = {}
    best_fitsAlpha = {}

    Ns = [25, 50, 100, 125, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    alphas = [0.01, 0.002, 0.005, 0.007, 0.0001]
    Rvir = 0.5
    Rc = 30.0

    original_path = path

    for cluster in data_ages:
        a = 0.01
        print("{0}, d={1}".format(cluster, data_densities[cluster]))
        similar = []
        t = data_ages[cluster]
        if data_ages[cluster] <= 2:
            Ns = [25, 50, 100, 125, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
            path = original_path
        else:
            Ns = [25, 50, 100, 125, 250, 500, 750, 1000, 1250, 1500]#, 1750, 2000]
            path = 'long_runs'
        d = data_densities[cluster]

        bestN = 0
        smallest_diff = 1000

        for N in Ns:  # Find N for closest stellar density
            if data_ages[cluster] <= 2:
                file_path = '{0}/no_gas/{1}/{2}/{3}/0/R0.5_{4}.hdf5'.format(path, N, a, Rc, int(t * 1E6))
            else:
                file_path = '{0}/no_gas/{1}/{2}/0/R0.5_{3}.hdf5'.format(path, N, a, int(t * 1E6))
            stars = io.read_set_from_file(file_path, 'hdf5')
            converter = nbody_system.nbody_to_si(stars.stellar_mass.sum() + stars.initial_disk_mass.sum(),
                                                 Rvir | units.parsec)
            radii = LagrangianRadii(stars, converter, mf=[0.5])
            hmr = radii[0].value_in(units.parsec)
            n = float(n_stars(stars, hmr)) / ((4 / 3) * np.pi * hmr ** 3)[0]
            diff = np.absolute(n - data_densities[cluster])

            if diff < smallest_diff:
                bestN = N
                smallest_diff = diff

        best_fitsN[cluster] = bestN

        bestp = 1E-70
        bestAlpha = 0.0001

        for alpha in alphas:
            if data_ages[cluster] <= 2:
                file_path = '{0}/no_gas/{1}/{2}/{3}/0/R0.5_{4}.hdf5'.format(path, N, alpha, Rc, int(t * 1E6))
            else:
                file_path = '{0}/no_gas/{1}/{2}/0/R0.5_{3}.hdf5'.format(path, N, alpha, int(t * 1E6))
            stars = io.read_set_from_file(file_path, 'hdf5')

            disk_size = 2.95 * 2 * stars.initial_characteristic_disk_radius.value_in(units.AU)

            disk_masses = []
            star_count = 0

            for s in stars:
                this_90mass = np.log10(disk_mass_within_radius(s, t * 1E6 | units.yr, disk_size[star_count] | units.AU, 1.).value_in(units.MJupiter))
                star_count += 1
                disk_masses.append(this_90mass)

            sorted_disk_masses = np.sort(disk_masses)

            if cluster == 'Trapezium':
                # Trapezium data
                r_path = 'data/Trapezium Cluster/disk_masses'  # Mann & Williams 2009
                lines = open(r_path, 'r').readlines()

                masses = []
                masses_errors = []

                for line in (l for l in lines if not l.startswith('#')):
                    mass = 100 * float(line.split()[7]) * 1E-2 * 1E3  # MJup
                    mass_error = float(line.split()[9]) * 1E-2 * 1E3
                    masses.append(mass)
                    masses_errors.append(mass_error)

                trapezium_sorted_masses = np.sort(np.log10(masses))
                p = KStest(sorted_disk_masses, trapezium_sorted_masses)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            elif cluster == 'Lupus':
                # Lupus data
                lines = open('data/Tazzari_et_al_2017', 'r').readlines()
                lupus_masses = []

                for line in (line for line in lines if not line.startswith('#')):
                    data = line.split()[7]
                    lupus_masses.append(float(data.split('+')[0]) / 317.8)  # MEarth to MJup conversion

                lupus_sorted_masses = np.sort(np.log10(lupus_masses))

                p = KStest(sorted_disk_masses, lupus_sorted_masses)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            elif cluster == 'Cham':
                # Chamaeleon I data (Long et al 2017)
                lines = open('data/ChamaeleonI/Mulders_et_al_2017', 'r').readlines()
                cham_masses = []

                for line in (line for line in lines if not line.startswith('#')):  # DUST masses
                    if len(line.split()) <= 11:
                        a = line.split()[4]
                    else:
                        a = line.split()[8]
                    cham_masses.append(float(a) / np.log10(317.8))  # MEarth to MJup conversion

                cham_sorted_masses = np.sort(100 * cham_masses)  # data already in log

                p = KStest(sorted_disk_masses, cham_sorted_masses)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            elif cluster == 'UpperSco':
                # UpperSco data (Barenfeld et al 2016)
                lines = open('data/UpperSco/mass', 'r').readlines()
                uppersco_masses = []

                for line in (line for line in lines if not line.startswith('#')):
                    a = line.split()[2]
                    uppersco_masses.append(
                        (float(a) * 100) / 317.8)  # Dust mass to gas mass and MEarth to MJup conversion

                uppersco_sorted_masses = np.sort(np.log10(uppersco_masses))

                p = KStest(sorted_disk_masses, uppersco_sorted_masses)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            elif cluster == 'Orionis':
                # sigma Orionis data (Ansdell+ 2017)
                lines = open('data/Ansdell_et_al_2017_Ori', 'r').readlines()
                sOrionis_masses = []

                for line in (line for line in lines if not line.startswith('#')):
                    a = line.split()[15]
                    sOrionis_masses.append((100 * float(a)) / 317.8)  # MEarth to MJup conversion

                sOrionis_sorted_masses = np.sort(np.array(np.log10(sOrionis_masses)))

                p = KStest(sorted_disk_masses, sOrionis_sorted_masses)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            best_fitsAlpha[cluster] = bestAlpha

    return best_fitsN, best_fitsAlpha


def find_model_acc(path, nruns, data_ages, data_densities, log=True):
    """ For each observed region, find the closest simulation regarding disk stellar accretion rate.
        See section 3.4 of paper for details.

    :param path: path to simulation files
    :param nruns: number of runs to be used for mean
    :param data_ages: estimated ages of the observed regions
    :param data_densities: estimated stellar densities of the observed regions
    :param log: log scale for data
    :return: best N and alpha for each observed region
    """
    best_fitsN = {}
    best_fitsAlpha = {}

    Ns = [25, 50, 100, 125, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]  # , 2250, 2500, 5000]
    # Ns = [125, 250, 500, 1000, 1250, 2500]
    alphas = [0.01, 0.002, 0.005, 0.007, 0.0001]
    Rvir = 0.5
    Rc = 30.0

    original_path = path

    for cluster in data_ages:
        a = 0.01
        print("{0}, d={1}".format(cluster, data_densities[cluster]))
        similar = []
        t = data_ages[cluster]
        if data_ages[cluster] <= 2:
            Ns = [25, 50, 100, 125, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
            path = original_path
        else:
            Ns = [25, 50, 100, 125, 250, 500, 750, 1000, 1250, 1500]#, 1750, 2000]
            path = 'long_runs'
        d = data_densities[cluster]

        bestN = 0
        smallest_diff = 1000  # Ns[0]/((4/3) * np.pi * Rvir**3)

        for N in Ns:
            if data_ages[cluster] <= 2:
                file_path = '{0}/no_gas/{1}/{2}/{3}/0/R0.5_{4}.hdf5'.format(path, N, a, Rc, int(t * 1E6))
            else:
                file_path = '{0}/no_gas/{1}/{2}/0/R0.5_{3}.hdf5'.format(path, N, a, int(t * 1E6))
            stars = io.read_set_from_file(file_path, 'hdf5')
            converter = nbody_system.nbody_to_si(stars.stellar_mass.sum() + stars.initial_disk_mass.sum(),
                                                 Rvir | units.parsec)
            radii = LagrangianRadii(stars, converter, mf=[0.5])
            hmr = radii[0].value_in(units.parsec)
            n = float(n_stars(stars, hmr)) / ((4 / 3) * np.pi * hmr ** 3)[0]
            diff = np.absolute(n - data_densities[cluster])

            if diff < smallest_diff:
                bestN = N
                smallest_diff = diff

        best_fitsN[cluster] = bestN

        bestp = 1E-70
        bestAlpha = 0.0001

        for alpha in alphas:
            if data_ages[cluster] <= 2:
                file_path = '{0}/no_gas/{1}/{2}/{3}/0/R0.5_{4}.hdf5'.format(path, N, alpha, Rc, int(t * 1E6))
            else:
                file_path = '{0}/no_gas/{1}/{2}/0/R0.5_{3}.hdf5'.format(path, N, alpha, int(t * 1E6))

            stars = io.read_set_from_file(file_path, 'hdf5')

            rates = np.log10(disk_accretion_rate(stars, t * 1E6 | units.yr).value_in(units.MJupiter / units.yr))
            sorted_acc_rates = np.sort(rates)

            if cluster == 'Trapezium':
                # Trapezium data
                r_path = 'data/Trapezium Cluster/accretion_rates_Rv31'
                lines = open(r_path, 'r').readlines()

                acc_rates = []

                for line in (l for l in lines if not l.startswith('#')):
                    acc_rate = float(line.split()[8]) * 1E-9 / (9.5E-4)  # MJup / yr
                    acc_rates.append(acc_rate)

                trapezium_sorted_acc_rates = np.array(np.sort(np.log10(acc_rates)))

                p = KStest(sorted_acc_rates, trapezium_sorted_acc_rates)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            elif cluster == 'Lupus':

                # Lupus data
                lupus_names, lupus_rates = [], []

                i = 0
                with open("data/Lupus Complex/acc_rates", 'r') as filepointer:
                    for row in filepointer:
                        if i < 1:
                            i += 1
                        else:
                            n = row.split()[0]
                            lupus_names.append(n)
                            a = row.split()[6]
                            lupus_rates.append(10 ** float(a) / (9.5E-4))  # MSun/yr to Mjup/yr
                        # print(a)
                filepointer.close()

                lupus_sorted_acc_rates = np.log10(np.sort([float(x) for x in lupus_rates]))
                p = KStest(sorted_acc_rates, lupus_sorted_acc_rates)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            elif cluster == 'Cham':
                # Chamaeleon I data (Pascucci et al 2016)
                lines = open('data/ChamaeleonI/acc', 'r').readlines()
                cham_acc_rates = []

                for line in (line for line in lines if not line.startswith('#')):
                    a = line.split()[0]
                    cham_acc_rates.append(10 ** float(a) / (9.5E-4))  # MSun to MJup conversion

                cham_sorted_acc_rates = np.sort(np.log10(cham_acc_rates))

                p = 1. * np.arange(len(cham_sorted_acc_rates)) / (len(cham_sorted_acc_rates) - 1)
                p = KStest(sorted_acc_rates, cham_sorted_acc_rates)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            elif cluster == 'Orionis':
                # sigma Orionis data (Mauco et al 2016)
                lines = open('data/sigmaOrionis2', 'r').readlines()
                sOrionis_acc_rates = []

                for line in (line for line in lines if not line.startswith('#')):
                    a = line.split()[9]
                    sOrionis_acc_rates.append(float(a) / (9.5E-4))  # MSun to MJup

                sOrionis_sorted_acc_rates = np.sort(np.array(np.log10(sOrionis_acc_rates)))
                p = KStest(sorted_acc_rates, sOrionis_sorted_acc_rates)[1]
                if p > bestp:
                    bestp = p
                    bestAlpha = alpha

            best_fitsAlpha[cluster] = bestAlpha

    return best_fitsN, best_fitsAlpha


def plot_models(best_fitsN, best_fitsAlpha, path, nruns, data_ages, data_ages_errors, data_densities,
                save=False):
    """ Figure 4 on paper:
        Plot observed region's ages vs stellar densities, along with their most similar simulation.
        See section 3.4 of paper for details.

    :param best_fitsN: dictionary with best N for each observed region
    :param best_fitsAlpha: dictionary with best alpha for each observed region
    :param path: path to simulation snapshots
    :param nruns: number of runs to use for mean
    :param data_ages: estimated ages of observed regions
    :param data_ages_errors: error of the estimated ages
    :param data_densities: estimated stellar densities of observed regions
    :param save: True to save Figure
    """
    times = np.arange(0, 2050000, 50000)
    alpha = 0.0001
    Rvir = 0.5
    Rc = 30.0

    plt.clf()
    fig = plt.figure(figsize=(10, 8), dpi=90)
    ax = plt.subplot(111)
    global plot_colors

    labels = {'Trapezium': 'Trapezium', 'Lupus': 'Lupus clouds', 'Orionis': r'$\sigma$ Orionis',
              'UpperSco': 'Upper Scorpio', 'Cham': 'Chamaeleon I'}

    for cluster in best_fitsN:
        densities = []
        if cluster == 'Lupus' or cluster == 'Cham':
            times = np.arange(0, 2050000, 50000)
        elif cluster == 'Orionis':
            times = np.arange(0, 5050000, 50000)
        elif cluster == 'UpperSco':
            times = np.arange(0, 11050000, 50000)
        else:
            times = np.arange(0, 2050000, 50000)

        # Calculate mean and plot simulations at correct snapshot
        for t in times:
            all_runs = []
            for r in range(nruns):
                file_path = './{0}/no_gas/{1}/{2}/{3}/{4}/R0.5_{5}.hdf5'.format(path,
                                                                                best_fitsN[cluster],
                                                                                best_fitsAlpha[cluster],
                                                                                Rc, r, t)
                stars = io.read_set_from_file(file_path, 'hdf5', close_file=True)
                converter = nbody_system.nbody_to_si(stars.stellar_mass.sum() + stars.initial_disk_mass.sum(),
                                                     Rvir | units.parsec)
                hmr = 0.5
                n = float(n_stars(stars, hmr)) / ((4 / 3) * np.pi * hmr ** 3)  # [0]
                all_runs.append(n)
            densities.append(np.mean(all_runs, axis=0))

        ax.plot(times / 1E6, densities, '--', color=plot_colors[cluster], lw=3)

    # Plot observations
    ax.errorbar(data_ages['Trapezium'], data_densities['Trapezium'], xerr=data_ages_errors['Trapezium'], marker='x',
                markersize=12, mew=6, c=plot_colors['Trapezium'], ls='-', lw=2, label=labels['Trapezium'], capsize=4)

    ax.errorbar(data_ages['Lupus'], data_densities['Lupus'], xerr=data_ages_errors['Lupus'], marker='x', markersize=12,
                mew=6, c=plot_colors['Lupus'], ls='-', lw=2, label=labels['Lupus'], capsize=4)

    ax.errorbar(data_ages['Orionis'], data_densities['Orionis'], xerr=data_ages_errors['Orionis'], marker='x',
                markersize=12, mew=6, c=plot_colors['Orionis'], ls='-', lw=2, label=labels['Orionis'], capsize=4)

    ax.errorbar(data_ages['UpperSco'], data_densities['UpperSco'], xerr=data_ages_errors['UpperSco'], marker='x',
                markersize=12, mew=6, c=plot_colors['UpperSco'], ls='-', lw=2, label=labels['UpperSco'], capsize=4)

    ax.errorbar(data_ages['Cham'], data_densities['Cham'], xerr=data_ages_errors['Cham'], marker='x', markersize=12,
                mew=6, c=plot_colors['Cham'], ls='-', lw=2, label=labels['Cham'], capsize=4)

    # Making the legend look good
    handles, labels2 = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax.legend(handles, labels2, loc='upper right', numpoints=1, fontsize=20)

    ax.set_yscale('log')
    plt.xlim([0.0, 11.0])

    plt.xlabel('Time [Myr]')
    plt.ylabel(r'$\log$ Number density [pc$^{-3}$]')
    if save:
        plt.savefig('../viscous-disks/figures/redo/data_densities.png')
    plt.show()


def plot_cdfs(best_fitsN_size, best_fitsAlpha_size, best_fitsN_mass, best_fitsAlpha_mass, best_fitsN_acc, best_fitsAlpha_acc,
              path, nruns, data_ages, data_ages_errors, data_densities,
              save=False):
    """ Figure 5 on paper:
        Plot observed region's ages vs stellar densities, along with their most similar simulation.
        See section 3.4 of paper for details.

    :param best_fitsN_size: dictionary of best N for disk size for each observed region
    :param best_fitsAlpha_size: dictionary of best alpha for disk size for each observed region
    :param best_fitsN_mass: dictionary of best N for disk mass for each observed region
    :param best_fitsAlpha_mass: dictionary of best alpha for disk mass for each observed region
    :param best_fitsN_acc: dictionary for best N for disk stellar accretion rate for each observed region
    :param best_fitsAlpha_acc: dictionary for best alpha for disk stellar accretion rate for each observed region
    :param path: path to simulation snapshots
    :param nruns: number of runs to use for mean
    :param data_ages: estimated ages of observed regions
    :param data_ages_errors: errors of estimated ages
    :param data_densities: estimated stellar densities of observed regions
    :param save: True to save figure
    """
    times = np.arange(0, 2050000, 50000)
    alpha = 0.0001
    Rvir = 0.5
    Rc = 30.0

    fig_size = plt.figure(figsize=(10, 8), dpi=90)
    ax_size = plt.subplot(111)
    fig_mass = plt.figure(figsize=(10, 8), dpi=90)
    ax_mass = plt.subplot(111)
    fig_acc = plt.figure(figsize=(10, 8), dpi=90)
    ax_acc = plt.subplot(111)
    global plot_colors

    labels = {'Trapezium': 'Trapezium', 'Lupus': 'Lupus clouds',
              'Orionis': 'sigma Orionis',
              'UpperSco': 'Upper Scorpio', 'Cham': 'Chamaeleon I'}
    data_ages = {'Trapezium': 0.3, 'Lupus': 2, 'Cham': 2.5, 'Orionis': 4, 'UpperSco': 8}

    for cluster in best_fitsN_size:
        densities = []
        if cluster == 'Lupus' or cluster == 'Cham':
            times = np.arange(0, 2050000, 50000)
        elif cluster == 'Orionis':
            times = np.arange(0, 5050000, 50000)
        elif cluster == 'UpperSco':
            times = np.arange(0, 11050000, 50000)
        else:
            times = np.arange(0, 2050000, 50000)

        snapshot = int(data_ages[cluster] * 1E6)

        all_runs_size = []
        all_runs_mass = []
        all_runs_acc = []

        for r in range(nruns):
            file_path = './{0}/no_gas/{1}/{2}/{3}/{4}/R0.5_{5}.hdf5'.format(path, best_fitsN_size[cluster],
                                                                                           best_fitsAlpha_size[cluster], Rc,
                                                                                           r, snapshot)
            stars = io.read_set_from_file(file_path, 'hdf5')
            disk_size = np.log10(2.95 * 2 * stars.initial_characteristic_disk_radius.value_in(units.AU))
            sorted_disk_size = np.sort(disk_size)
            all_runs_size.append(sorted_disk_size)

            file_path = './{0}/no_gas/{1}/{2}/{3}/{4}/R0.5_{5}.hdf5'.format(path, best_fitsN_mass[cluster],
                                                                                           best_fitsAlpha_mass[cluster], Rc,
                                                                                           r, snapshot)
            stars2 = io.read_set_from_file(file_path, 'hdf5')
            disk_size2 = np.log10(2.95 * 2 * stars2.initial_characteristic_disk_radius.value_in(units.AU))

            disk_masses = []
            star_count = 0

            for s in stars2:
                this_90mass = np.log10(disk_mass_within_radius(s, snapshot | units.yr, disk_size2[star_count] | units.AU, 1.).value_in(units.MJupiter))
                star_count += 1
                disk_masses.append(this_90mass)

            sorted_disk_mass = np.sort(disk_masses)
            all_runs_mass.append(sorted_disk_mass)

            file_path = './{0}/no_gas/{1}/{2}/{3}/{4}/R0.5_{5}.hdf5'.format(path, best_fitsN_acc[cluster],
                                                                                           best_fitsAlpha_acc[cluster], Rc,
                                                                                           r, snapshot)
            stars3 = io.read_set_from_file(file_path, 'hdf5')

            rates = np.log10(disk_accretion_rate(stars3, snapshot | units.yr).value_in(units.MJupiter / units.yr))
            sorted_acc_rates = np.sort(rates)
            all_runs_acc.append(sorted_acc_rates)

        all_runs = np.array(all_runs_size)
        all_runs_mass = np.array(all_runs_mass)
        all_runs_acc = np.array(all_runs_acc)

        sorted_disk_sizes = np.mean(all_runs_size, axis=0)
        sorted_disk_size_errors = np.std(all_runs_size, axis=0)
        cumulative_size = np.array([float(x) for x in np.arange(sorted_disk_sizes.size)])

        sorted_disk_masses = np.mean(all_runs_mass, axis=0)
        sorted_disk_masses_errors = np.std(all_runs_mass, axis=0)
        cumulative_mass = np.array([float(x) for x in np.arange(sorted_disk_masses.size)])

        sorted_disk_accs = np.mean(all_runs_acc, axis=0)
        sorted_disk_accs_errors = np.std(all_runs_acc, axis=0)
        cumulative_acc = np.array([float(x) for x in np.arange(sorted_disk_accs.size)])

        # Plot best simulations' sizes
        p = 1. * np.arange(len(sorted_disk_sizes)) / (len(sorted_disk_sizes) - 1)
        ax_size.plot(sorted_disk_sizes, p, c=plot_colors[cluster], ls='--', lw=3)
        ax_size.set_xlabel(r'$\log{d} \left[\mathrm{au}\right]$')
        ax_size.set_ylabel(r'$f < d$')

        # Plot best simulations' masses
        p = 1. * np.arange(len(sorted_disk_masses)) / (len(sorted_disk_masses) - 1)
        ax_mass.plot(sorted_disk_masses, p, c=plot_colors[cluster], ls='--', lw=3)
        ax_mass.set_xlabel(r'$\log{M} \left[\mathrm{M_{Jup}}\right]$')
        ax_mass.set_ylabel(r'$f < M$')

        # Plot best simulations' stellar accretion rate
        p = 1. * np.arange(len(sorted_disk_accs)) / (len(sorted_disk_accs) - 1)
        if cluster is not 'UpperSco':
            ax_acc.plot(sorted_disk_accs, p, c=plot_colors[cluster], ls='--', lw=3)
        ax_acc.set_xlabel(r'$\log{\dot{M_{\star}}} \left[\mathrm{M_{Jup} / yr}\right]$', labelpad=3)
        ax_acc.set_ylabel(r'$f < \dot{M_{\star}}$')

        savefile = open('../viscous-disks/plot_data/{0}_CDF_plot_sim_data.txt'.format(cluster), 'w')
        savefile.write('# {0} Cluster\n'.format(labels[cluster]))
        savefile.write('#\tLog10[Disk size] (AU)\tLog10[Disk mass] (MJup)\tLog10[Star acc rate] (MJup/yr)\n')
        for s, m, a in zip(sorted_disk_sizes, sorted_disk_masses, sorted_disk_accs):
            savefile.write('{0}\t{1}\t{2}\n'.format(s, m, a))
        savefile.close()

    ######### PLOT OBSERVED DISK SIZES #########
    # Trapezium data (Vicente & Alves 2005)
    lines = open('data/Trapezium Cluster/Vicente_Alves_2005_Table3.txt', 'r').readlines()
    trapezium_sizes = []

    for line in (line for line in lines if not line.startswith('#')):
        data = line.split()[8]
        trapezium_sizes.append(float(data))

    trapezium_sizes = np.array(trapezium_sizes)
    sorted_trapezium_sizes = np.sort(np.log10(trapezium_sizes[trapezium_sizes > 100.]))
    p = 1. * np.arange(len(sorted_trapezium_sizes)) / (len(sorted_trapezium_sizes) - 1)
    ax_size.plot(sorted_trapezium_sizes, p, c=plot_colors['Trapezium'], ls='-', lw=3)

    # Lupus data (Ansdell et al 2016)
    lines = open('data/Lupus Complex/Ansdell_et_al_2018', 'r').readlines()
    lupus_sizes = []

    for line in (line for line in lines if not line.startswith('#')):
        data = line.split()[0]
        lupus_sizes.append(2 * float(data))#.split('+')[0]))  # Radius in AU

    sorted_lupus_disk_sizes_au = np.sort(np.log10(lupus_sizes))
    p = 1. * np.arange(len(sorted_lupus_disk_sizes_au)) / (len(sorted_lupus_disk_sizes_au) - 1)
    ax_size.plot(sorted_lupus_disk_sizes_au, p, c=plot_colors['Lupus'], ls='-', lw=3)

    # Chamaeleon I data (Pascucci et al 2016)
    lines = open('data/ChamaeleonI/size', 'r').readlines()
    cham_sizes_arsec = []

    for line in (line for line in lines if not line.startswith('#')):
        a = line.split()[7]
        b = line.split()[8]
        if a > b:
            cham_sizes_arsec.append(float(a))
        else:
            cham_sizes_arsec.append(float(b))

    cham_sizes_arsec = np.array(cham_sizes_arsec)
    cham_sizes_arsec = cham_sizes_arsec[cham_sizes_arsec > 0.0]

    cham_distance_pc = 160
    cham_distance_au = 2.0626 * pow(10, 5) * cham_distance_pc
    cham_sizes_au = (pi / 180) * (cham_sizes_arsec / 3600.) * cham_distance_au
    cham_sorted_disk_sizes = np.sort(np.log10(cham_sizes_au))
    p = 1. * np.arange(len(cham_sorted_disk_sizes)) / (len(cham_sorted_disk_sizes) - 1)
    ax_size.plot(cham_sorted_disk_sizes, p, c=plot_colors['Cham'], ls='-', lw=3)

    # UpperSco data (Barenfeld et al 2016)
    lines = open('data/UpperSco/Barenfeld_et_al_2017', 'r').readlines()
    uppersco_sizes = []

    for line in (line for line in lines if not line.startswith('#')):
        data = line.split()[8]
        uppersco_sizes.append(2 * float(data))#.split('+')[0]))  # Radius in AU

    uppersco_sorted_disk_sizes = np.sort(np.log10(uppersco_sizes))

    p = 1. * np.arange(len(uppersco_sorted_disk_sizes)) / (len(uppersco_sorted_disk_sizes) - 1)
    ax_size.plot(uppersco_sorted_disk_sizes, p, c=plot_colors['UpperSco'], ls='-', lw=3)

    # sigma Orionis data (Mauco et al 2016)
    lines = open('data/sigmaOrionis', 'r').readlines()
    sOrionis_sizes_au = []

    for line in (line for line in lines if not line.startswith('#')):
        a = line.split()[1]
        sOrionis_sizes_au.append(2 * float(a))

    sOrionis_sorted_disk_sizes = np.sort(np.array(np.log10(sOrionis_sizes_au)))
    p = 1. * np.arange(len(sOrionis_sorted_disk_sizes)) / (len(sOrionis_sorted_disk_sizes) - 1)
    ax_size.plot(sOrionis_sorted_disk_sizes, p, c=plot_colors['Orionis'], ls='-', lw=3)

    ######### PLOT OBSERVED DISK MASSES #########
    # Trapezium data
    r_path = 'data/Trapezium Cluster/disk_masses'  # Mann & Williams 2009
    lines = open(r_path, 'r').readlines()

    masses = []
    masses_errors = []

    for line in (l for l in lines if not l.startswith('#')):
        mass = 100 * float(line.split()[7]) * 1E-2 * 1E3  # MJup
        mass_error = float(line.split()[9]) * 1E-2 * 1E3
        masses.append(mass)
        masses_errors.append(mass_error)

    trapezium_sorted_masses = np.sort(np.log10(masses))

    p = 1. * np.arange(len(trapezium_sorted_masses)) / (len(trapezium_sorted_masses) - 1)
    ax_mass.plot(trapezium_sorted_masses, p, c=plot_colors['Trapezium'], ls='-', lw=3)

    # Lupus data
    lines = open('data/Tazzari_et_al_2017', 'r').readlines()
    lupus_masses = []

    for line in (line for line in lines if not line.startswith('#')):
        data = line.split()[7]
        lupus_masses.append(float(data.split('+')[0]) / 317.8)  # MEarth to MJup conversion

    lupus_sorted_masses = np.sort(np.log10(lupus_masses))
    p = 1. * np.arange(len(lupus_sorted_masses)) / (len(lupus_sorted_masses) - 1)
    ax_mass.plot(lupus_sorted_masses, p, c=plot_colors['Lupus'], ls='-', lw=3)

    # Chamaeleon I data (Long et al 2017)
    lines = open('data/ChamaeleonI/Mulders_et_al_2017', 'r').readlines()
    cham_masses = []

    for line in (line for line in lines if not line.startswith('#')):  # DUST masses
        if len(line.split()) <= 11:
            a = line.split()[4]
        else:
            a = line.split()[8]
        cham_masses.append(float(a)/ np.log10(317.8))  # MEarth to MJup conversion

    cham_sorted_masses = np.sort(100 * cham_masses)  #data already in log
    p = 1. * np.arange(len(cham_sorted_masses)) / (len(cham_sorted_masses) - 1)
    ax_mass.plot(cham_sorted_masses, p, c=plot_colors['Cham'], ls='-', lw=3)

    # UpperSco data (Barenfeld et al 2016)
    lines = open('data/UpperSco/mass', 'r').readlines()
    uppersco_masses = []

    for line in (line for line in lines if not line.startswith('#')):
        a = line.split()[2]
        uppersco_masses.append((float(a)*100) / 317.8)  # Dust mass to gas mass and MEarth to MJup conversion

    uppersco_sorted_masses = np.sort(np.log10(uppersco_masses))
    p = 1. * np.arange(len(uppersco_sorted_masses)) / (len(uppersco_sorted_masses) - 1)
    ax_mass.plot(uppersco_sorted_masses, p, c=plot_colors['UpperSco'], ls='-', lw=3)

    # sigma Orionis data (Ansdell+ 2017)
    lines = open('data/Ansdell_et_al_2017_Ori', 'r').readlines()
    sOrionis_masses = []

    for line in (line for line in lines if not line.startswith('#')):
        a = line.split()[15]
        sOrionis_masses.append((100 * float(a)) / 317.8)  # MEarth to MJup conversion

    sOrionis_sorted_masses = np.sort(np.array(np.log10(sOrionis_masses)))
    p = 1. * np.arange(len(sOrionis_sorted_masses)) / (len(sOrionis_sorted_masses) - 1)
    ax_mass.plot(sOrionis_sorted_masses, p, c=plot_colors['Orionis'], ls='-', lw=3)

    ######### PLOT OBSERVED DISK STELLAR ACCRETION RATES #########

    # Trapezium data
    r_path = 'data/Trapezium Cluster/accretion_rates_Rv31'
    lines = open(r_path, 'r').readlines()

    acc_rates = []

    for line in (l for l in lines if not l.startswith('#')):
        acc_rate = float(line.split()[8]) * 1E-9 / (9.5E-4)  # MJup / yr
        acc_rates.append(acc_rate)

    trapezium_sorted_acc_rates = np.array(np.sort(np.log10(acc_rates)))
    p = 1. * np.arange(len(trapezium_sorted_acc_rates)) / (len(trapezium_sorted_acc_rates) - 1)
    ax_acc.plot(trapezium_sorted_acc_rates, p, c=plot_colors['Trapezium'], ls='-', lw=3)
    # ax.text(-6.9, 0.15, 'Trapezium', fontsize=18, color=plot_colors[0], rotation=75)

    # Lupus data
    lupus_names, lupus_rates = [], []

    i = 0
    with open("data/Lupus Complex/acc_rates", 'r') as filepointer:
        for row in filepointer:
            if i < 1:
                i += 1
            else:
                n = row.split()[0]
                lupus_names.append(n)
                a = row.split()[6]
                lupus_rates.append(10 ** float(a) / (9.5E-4))  # MSun/yr to Mjup/yr
                #print(a)
    filepointer.close()

    lupus_sorted_acc_rates = np.log10(np.sort([float(x) for x in lupus_rates]))
    p = 1. * np.arange(len(lupus_sorted_acc_rates)) / (len(lupus_sorted_acc_rates) - 1)
    ax_acc.plot(lupus_sorted_acc_rates, p, c=plot_colors['Lupus'], ls='-', lw=3)

    # Chamaeleon I data (Pascucci et al 2016)
    lines = open('data/ChamaeleonI/acc', 'r').readlines()
    cham_acc_rates = []

    for line in (line for line in lines if not line.startswith('#')):
        a = line.split()[0]
        cham_acc_rates.append(10 ** float(a) / (9.5E-4))  # MSun to MJup conversion

    cham_sorted_acc_rates = np.sort(np.log10(cham_acc_rates))

    p = 1. * np.arange(len(cham_sorted_acc_rates)) / (len(cham_sorted_acc_rates) - 1)
    ax_acc.plot(cham_sorted_acc_rates, p, c=plot_colors['Cham'], ls='-', lw=3)

    # sigma Orionis data (Mauco et al 2016)
    lines = open('data/sigmaOrionis2', 'r').readlines()
    sOrionis_acc_rates = []

    for line in (line for line in lines if not line.startswith('#')):
        a = line.split()[9]
        sOrionis_acc_rates.append(float(a) / (9.5E-4))  # MSun to MJup

    sOrionis_sorted_acc_rates = np.sort(np.array(np.log10(sOrionis_acc_rates)))
    p = 1. * np.arange(len(sOrionis_sorted_acc_rates)) / (len(sOrionis_sorted_acc_rates) - 1)
    ax_acc.plot(sOrionis_sorted_acc_rates, p, c=plot_colors['Orionis'], ls='-', lw=3)

    # Legend
    ax_size.legend([TrapeziumObject(), LupusObject(), OrionisObject(), UpperScoObject(), ChamObject()],
                   ['Trapezium', 'Lupus clouds', r'$\sigma$ Orionis', 'Upper Scorpio', 'Chamaeleon I'],
                   handler_map={TrapeziumObject: TrapeziumObjectHandler(), LupusObject: LupusObjectHandler(),
                                OrionisObject: OrionisObjectHandler(),
                                UpperScoObject: UpperScoObjectHandler(), ChamObject: ChamObjectHandler()},
                   loc='upper left', fontsize=20)
    ax_mass.legend([TrapeziumObject(), LupusObject(), OrionisObject(), UpperScoObject(), ChamObject()],
                   ['Trapezium', 'Lupus clouds', r'$\sigma$ Orionis', 'Upper Scorpio', 'Chamaeleon I'],
                   handler_map={TrapeziumObject: TrapeziumObjectHandler(), LupusObject: LupusObjectHandler(),
                                OrionisObject: OrionisObjectHandler(),
                                UpperScoObject: UpperScoObjectHandler(), ChamObject: ChamObjectHandler()},
                   loc='upper left', fontsize=20)
    ax_acc.legend([TrapeziumObject(), LupusObject(), OrionisObject(), ChamObject()],
                  ['Trapezium', 'Lupus clouds', r'$\sigma$ Orionis', 'Chamaeleon I'],
                  handler_map={TrapeziumObject: TrapeziumObjectHandler(), LupusObject: LupusObjectHandler(),
                               OrionisObject: OrionisObjectHandler(),
                               ChamObject: ChamObjectHandler()},
                  loc='upper left', fontsize=20)

    ax_size.set_xlim([0.4, 3.1])
    ax_mass.set_xlim([-8.0, 4.0])
    ax_acc.set_xlim([-10.0, -3.5])
    ax_size.set_ylim([0.0, 1.0])
    ax_mass.set_ylim([0.0, 1.0])
    ax_acc.set_ylim([0.0, 1.0])

    if save:
        fig_size.savefig('../viscous-disks/figures/redo/bestfit_sizes.png')
        fig_mass.savefig('../viscous-disks/figures/redo/bestfit_masses.png')
        fig_acc.savefig('../viscous-disks/figures/redo/bestfit_accrates.png')

    plt.show()


def plot_density(path, N, alpha, nruns, gas_scenario, data_ages, data_ages_errors, data_densities, save=False):
    """ Figure 6 on paper.
        See section 4 for details.

    :param path: path to simulation snapshots
    :param N: number of stars for plotted simulation
    :param alpha: alpha for plotted simulation
    :param nruns: number of runs to use for mean
    :param gas_scenario: gas scenario of simulation to plot
    :param data_ages: estimated ages of observed regions
    :param data_ages_errors: errors of estimated ages
    :param data_densities: estimated stellar densities of observed regions
    :param save: True to save figure
    """
    times = np.arange(0, 11050000, 50000)

    Rvir = 0.5
    Rc = 30.0

    plt.clf()
    fig = plt.figure(figsize=(11, 9), dpi=90)
    ax = plt.subplot(111)

    densities = []
    labels = {'Trapezium': 'Trapezium', 'Lupus': 'Lupus clouds', 'Orionis': r'$\sigma$ Orionis',
              'UpperSco': 'Upper Scorpio', 'Cham': 'Chamaeleon I'}

    for t in times:
        all_runs = []
        for r in range(nruns):
            file_path = './{0}/{1}/{2}/{3}/{4}/{5}/R0.5_{6}.hdf5'.format(path, gas_scenario, N, alpha,
                                                                                        Rc, r, t)
            stars = io.read_set_from_file(file_path, 'hdf5', close_file=True)
            converter = nbody_system.nbody_to_si(stars.stellar_mass.sum() + stars.initial_disk_mass.sum(),
                                                 Rvir | units.parsec)
            hmr = 0.5
            n = float(n_stars(stars, hmr)) / ((4 / 3) * np.pi * hmr ** 3)  # [0]
            all_runs.append(n)
        densities.append(np.mean(all_runs, axis=0))

    savefile = open('../viscous-disks/plot_data/processes_plot_data.txt', 'w')
    savefile.write('# {0}\n'.format(file_path))
    for l in densities:
        savefile.write('{0}\n'.format(l))
    savefile.close()

    ax.plot(times / 1E6, densities, '-', color='k', lw=2)

    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Rescale to values between 0 and 1
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    # These are the rectangles for each effect mentioned in the figure.
    # See Figure 6 on paper for references.
    ax.add_patch(patches.Rectangle(  # Visc growth
        (0, 0),  # (x,y)
        5,  # width
        2486,  # height
        facecolor="#e0f3f8",#"#6E97C9",#"#B6DBFF",#tableau20[15],  # "#d0d0c9",
        alpha=1,  # 0.2,
        # hatch="/",
    ))

    ax.add_patch(patches.Rectangle(  # Visc growth
        (0.04, 14),  # (x,y)
        4.96,  # width
        2472,  # height
        fill=False,
        edgecolor="#e0f3f8",#"#6E97C9",#"#B6DBFF",#tableau20[15],  # "#d0d0c9",
        linewidth=4,
        joinstyle="round",
        label="Viscous growth"))

    # ax.text(4, 149, 'Viscous growth', fontsize=26, color="#d0d0c9")#, rotation=80)

    ax.add_patch(patches.Rectangle(  # Dynamical truncations
        (0.8, 1000),  # (x,y)
        1.0,  # width
        3000,  # height
        facecolor="#4575b4",#"#79A34C",#"#006DDB",#tableau20[19],  # "#9bcde5",
        alpha=1,  # 0.3,
        # hatch="\\\\",
    ))

    ax.add_patch(patches.Rectangle(  # Dynamical truncations
        (0.8, 1000),  # (x,y)
        1.0,  # width
        3000,  # height
        fill=False,
        edgecolor="#4575b4",#"#79A34C",#"#006DDB",#tableau20[19],  # "#9bcde5",
        linewidth=4,
        joinstyle="round",
        label="Dynamical truncations"))

    ax.add_patch(patches.Rectangle(  # Ram pressure
        (0.04, 14),  # (x,y)
        3,  # width
        10000,  # height
        fill=False,
        facecolor="#fee090",  # "#009292",#tableau20[5],  # "#add5c0",
        edgecolor="#fee090",  # "#009292",#tableau20[5],  # "#add5c0",
        alpha=1.0,  # 0.2,
        label="Ram pressure stripping",
        hatch="\\",
        ))

    ax.add_patch(patches.Rectangle(  # Winds
        (0.96, 1700),  # (x,y)
        7.4,  # width
        800,  # height
        facecolor="#fc8d59",#"#FFB677",#tableau20[7],  # "#e9b4b2",
        alpha=1,  # 0.2,
        # hatch="*",
    ))

    ax.add_patch(patches.Rectangle(  # Winds
        (0.96, 1700),  # (x,y)
        7.4,  # width
        786,  # height
        fill=False,
        edgecolor="#fc8d59",#"#FFB677",#tableau20[7],  # "#e9b4b2",
        linewidth=4,
        joinstyle="round",
        label="Stellar wind feedback"))

    ax.add_patch(patches.Rectangle(  # Supernovae
        (4, 1900),  # (x,y)
        6,  # width
        600,  # height
        facecolor="#d73027",#"#B66DFF",#tableau20[9],  # "#cdbedd",
        alpha=1,  # 0.2,
        # hatch=".",
    ))

    ax.add_patch(patches.Rectangle(  # Supernovae
        (4, 1900),  # (x,y)
        6,  # width
        586,  # height
        fill=False,
        edgecolor="#d73027",#"#B66DFF",#tableau20[9],  # "#cdbedd",
        linewidth=4,
        joinstyle="round",
        label="Supernovae feedback"))

    ax.add_patch(patches.Polygon(  # External photoevap
        [(0.5, 2500), (1.0, 100), (10.0, 100), (10.0, 2500)],  # (x,y)
        closed=True,
        fill=False,
        facecolor="#91bfdb",#"#009292",#tableau20[5],  # "#add5c0",
        edgecolor="#91bfdb",#"#009292",#tableau20[5],  # "#add5c0",
        alpha=1.0,  # 0.2,
        label="External photoevaporation",
        hatch="/",
    ))

    # Observational data
    ax.errorbar(data_ages['Trapezium'], data_densities['Trapezium'], xerr=data_ages_errors['Trapezium'], marker='x',
                markersize=12, mew=6, c=plot_colors['Trapezium'], ls='-', lw=2, label=labels['Trapezium'], capsize=4)
    ax.text(0.01, 1870, 'Trapezium', fontsize=20, color=plot_colors['Trapezium'])  # , rotation=80)

    ax.errorbar(data_ages['Lupus'], data_densities['Lupus'], xerr=data_ages_errors['Lupus'], marker='x', markersize=12,
                mew=6, c=plot_colors['Lupus'], ls='-', lw=2, label=labels['Lupus'], capsize=4)
    ax.text(1.6, 555, 'Lupus', fontsize=20, color=plot_colors['Lupus'])  # , rotation=70)

    ax.errorbar(data_ages['Orionis'], data_densities['Orionis'], xerr=data_ages_errors['Orionis'], marker='x',
                markersize=12, mew=6, c=plot_colors['Orionis'], ls='-', lw=2, label=labels['Orionis'], capsize=4)
    ax.text(3.6, 45, r'$\sigma$Orionis', fontsize=20, color=plot_colors['Orionis'])  # , rotation=90)

    ax.errorbar(data_ages['UpperSco'], data_densities['UpperSco'], xerr=data_ages_errors['UpperSco'], marker='x',
                markersize=12, mew=6, c=plot_colors['UpperSco'], ls='-', lw=2, label=labels['UpperSco'], capsize=4)
    ax.text(7.5, 70, 'UpperSco', fontsize=20, color=plot_colors['UpperSco'])  # , rotation=85)

    ax.errorbar(data_ages['Cham'], data_densities['Cham'], xerr=data_ages_errors['Cham'], marker='x', markersize=12,
                mew=6, c=plot_colors['Cham'], ls='-', lw=2, label=labels['Cham'], capsize=4)
    ax.text(1.7, 45, 'Chamaeleon I', fontsize=20, color=plot_colors['Cham'])  # , rotation=80)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(
        [ViscousObject(), RamPressureObject(), DynTruncObject(), SupernovaeObject(), WindsObject(), EPhotoevapObject()],
        ['Viscous growth', 'Ram pressure stripping', 'Dynamical truncations', 'Supernovae feedback',
         'Stellar winds feedback', 'External photoevaporation'],
        handler_map={ViscousObject: ViscousObjectHandler(), RamPressureObject: RamPressureObjectHandler(),
                     DynTruncObject: DynTruncObjectHandler(),
                     SupernovaeObject: SupernovaeObjectHandler(), WindsObject: WindsObjectHandler(),
                     EPhotoevapObject: EPhotoevapObjectHandler()},
        loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=3, fontsize=16)

    plt.xlim([0.0, 10.0])
    plt.ylim([-0.5, 2500])
    plt.xlabel('Time [Myr]')
    plt.ylabel(r'Number density [pc$^{-3}$]')
    if save:
        plt.savefig('../viscous-disks/figures/redo/processes_solid2.png')
    plt.show()


def main(N, path, alpha, gas_scenario):
    # Observational data
    data_ages = {'Trapezium': 0.3, 'Lupus': 2, 'Cham': 2.5, 'Orionis': 4, 'UpperSco': 8}
    data_densities = {'Trapezium': 2000, 'Lupus': 500, 'Cham': 0.9, 'Orionis': 3, 'UpperSco': 0.05}  # pc^-3
    data_ages_errors = {'Trapezium': 0.5, 'Lupus': 1, 'Cham': 0.5, 'Orionis': 1, 'UpperSco': 3}
    data_densities_errors = {'Trapezium': 250, 'Lupus': 320, 'Cham': 0.9, 'Orionis': 3, 'UpperSco': 0.05}

    # Find closest simulation to each observed region
    best_fitsN_size, best_fitsAlpha_size = find_model_size(path, 3, data_ages, data_densities)
    best_fitsN_mass, best_fitsAlpha_mass = find_model_mass(path, 3, data_ages, data_densities)
    best_fitsN_acc, best_fitsAlpha_acc = find_model_acc(path, 3, data_ages, data_densities)

    # Figure 4 on paper
    plot_models(best_fitsN_size, best_fitsAlpha_size, path, 3, data_ages, data_ages_errors, data_densities, save=True)

    # Figure 5 on paper
    plot_cdfs(best_fitsN_size, best_fitsAlpha_size, best_fitsN_mass, best_fitsAlpha_mass,
              best_fitsN_acc, best_fitsAlpha_acc,
              path, 3, data_ages, data_ages_errors, data_densities, save=True)

    # Figure 6 on paper
    plot_density(path, N, alpha, 3, gas_scenario, data_ages, data_ages_errors, data_densities, save=True)


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()

    # Simulation
    result.add_option("-p", dest="path", type="string", default="redo_runs2/",
                      help="path to files [%default]")
    result.add_option("-N", dest="N", type="int", default=1250,
                      help="number of stars [%default]")
    result.add_option("-a", dest="alpha", type="float", default=0.001,
                      help="alpha [%default]")
    result.add_option("-g", dest="gas_scenario", type="string", default="no_gas",
                      help="Gas scenario [%default]")
    return result


if __name__ in ('__main__', '__plot__'):
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)
