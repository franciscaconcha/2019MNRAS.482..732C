# coding=utf-8
from amuse.lab import *
from amuse.couple.bridge import Bridge
import os
import numpy as np
import datetime


class PlummerSpherePotential(object):
    """ Plummer sphere potential for the gas in the cluster. """

    def __init__(self, a=1.0 | units.parsec, M=1e3 | units.MSun):
        self.a = a
        self.M = M

    def get_gravity_at_point(self, eps, x, y, z):
        fr = -constants.G * self.M / (x ** 2 + y ** 2 + z ** 2 + self.a ** 2) ** 1.5
        return fr * x, fr * y, fr * z

    def get_potential_at_point(self, eps, x, y, z):
        return -constants.G * self.M / (x ** 2 + y ** 2 + z ** 2 + self.a ** 2) ** 0.5


def disk_accretion_rate(star, t, gamma=1):
    """Return the disk accretion rate of a circumstellar disk at a given moment.

    :param star: the star with the disk
    :param t: time at which the accretion rate is to be evaluated
    :param gamma: radial viscosity dependence exponent
    :return: disk accretion rate.
    """
    T = 1 + (t - star.last_encounter) / star.viscous_timescale
    return T ** ((5 - 2 * gamma) / (2 * gamma - 4)) * star.initial_disk_mass / (4 - 2 * gamma) / star.viscous_timescale


def disk_characteristic_radius(star, t, gamma=1):
    """Return the characteristic radius of a circumstellar disk at a given moment.

    :param star: the star with the disk
    :param t: time at which the characteristic radius is to be evaluated
    :param gamma: radial viscosity dependence exponent
    """
    T = 1 + (t - star.last_encounter) / star.viscous_timescale
    return (T ** (1 / (2 - gamma))) * star.initial_characteristic_disk_radius


def disk_mass(star, t, gamma=1):
    """Return the total mass of a circumstellar disk at a given moment.

    :param star: the star with the disk
    :param t: time at which the disk mass is to be evaluated
    :param gamma: radial viscosity dependence exponent
    :return: total disk mass at moment t
    """
    T = 1 + (t - star.last_encounter) / star.viscous_timescale
    return T ** (1 / (2 * gamma - 4)) * star.initial_disk_mass


def disk_mass_within_radius(star, t, R, gamma=1):
    """Return the mass of a circumstellar disk at a given moment inside a given radius.

    :param star: the star with the disk
    :param t: time at which the disk mass is to be evaluated
    :param R: radius inside which the mass is to be evaluated
    :param gamma: radial viscosity dependence exponent
    :returns: disk mass at moment t inside radius R
    """
    R /= star.initial_characteristic_disk_radius
    R **= 2 - gamma
    R /= 1 + (t - star.last_encounter) / star.viscous_timescale
    return -disk_mass(star, t, gamma) * np.expm1(-R)


def periastron_distance(stars):
    """ Return the periastron distance of two encountering stars.

    :param stars: pair of encountering stars
    :return: periastron distance of the encounter
    """

    # Standard gravitational parameter
    mu = constants.G * stars.mass.sum()

    # Position vector from one star to the other
    r = stars[0].position - stars[1].position

    # Relative velocity between the stars
    v = stars[0].velocity - stars[1].velocity

    # Energy
    E = (v.length()) ** 2 / 2 - mu / r.length()

    # Semi-major axis
    a = -mu / 2 / E

    # Semi-latus rectum
    p = (np.cross(r.value_in(units.AU),
                  v.value_in(units.m / units.s)) | units.AU * units.m / units.s).length() ** 2 / mu

    # Eccentricity
    e = np.sqrt(1 - p / a)

    # Periastron distance
    return p / (1 + e)


def resolve_encounter(stars, time, mass_factor_exponent=0.2, truncation_parameter=1. / 3, gamma=1, verbose=False):
    """Resolve encounter between two stars.
    Changes radius and mass of the stars' disks according to eqs. in paper.

    :param stars: pair of encountering stars
    :param time: time at which encounter occurs
    :param mass_factor_exponent: exponent characterizing truncation mass dependence in a stellar encounter (eq. 13)
    :param truncation_parameter: factor characterizing the size of circumstellar disks after an encounter (eq. 13)
    :param gamma: radial viscosity dependence exponent
    :param verbose: verbose option for debugging
    """

    # For debugging
    if verbose:
        print(time.value_in(units.yr), stars.mass.value_in(units.MSun))

    closest_approach = periastron_distance(stars)

    # Check each star
    for i in range(2):
        truncation_radius = closest_approach * truncation_parameter * \
                            ((stars[i].mass / stars[1 - i].mass) ** mass_factor_exponent)

        if stars[i].closest_encounter > closest_approach:  # This is the star's closest encounter so far
            stars[i].closest_encounter = closest_approach

        if stars[i].strongest_truncation > truncation_radius:  # This is the star's strongest truncation so far
            stars[i].strongest_truncation = truncation_radius

        R_disk = disk_characteristic_radius(stars[i], time, gamma)
        stars[i].radius = 0.49 * closest_approach  # So that we don't detect this encounter in the next time step

        if truncation_radius < R_disk:
            stars[i].stellar_mass += stars[i].initial_disk_mass - disk_mass(stars[i], time, gamma)
            stars[i].initial_disk_mass = 1.58 * disk_mass_within_radius(stars[i], time, truncation_radius, gamma)
            stars[i].viscous_timescale *= (truncation_radius
                                           / stars[i].initial_characteristic_disk_radius) ** (2 - gamma)
            stars[i].initial_characteristic_disk_radius = truncation_radius
            stars[i].last_encounter = time


def set_output_directory(name):
    global output_directory
    if name[-1] != '/':
        name += '/'
    output_directory = name
    if not os.path.isdir(name):
        os.makedirs(name)


def viscous_timescale(star, alpha, temperature_profile, Rref, Tref, mu, gamma):
    """Compute the viscous timescale of the circumstellar disk.

    :param star: star with the circumstellar disk
    :param alpha: turbulence mixing strenght
    :param temperature_profile: negative of the temperature profile exponent, q in eq. (8)
    :param Rref: reference distance from the star at which the disk temperature is given
    :param Tref: disk temperature at the reference distance for a star with solar luminosity
    :param mu: molar mass of the gas in g/mol
    :param gamma: radial viscosity dependence exponent
    :return: viscous timescale in Myr
    """
    # To calculate luminosity
    stellar_evolution = SeBa()
    stellar_evolution.particles.add_particles(Particles(mass=star.stellar_mass))
    stellar_luminosity = stellar_evolution.particles.luminosity.value_in(units.LSun)
    stellar_evolution.stop()

    R = star.initial_characteristic_disk_radius
    T = Tref * (stellar_luminosity ** 0.25)
    q = temperature_profile
    M = constants.G * star.stellar_mass

    return mu * (R ** (0.5 + q)) * (M ** 0.5) / 3 / alpha / ((2 - gamma) ** 2) \
           / constants.molar_gas_constant / T / (Rref ** q)


def main(N, Rvir, Qvir, alpha, R, gas_presence, gas_expulsion, gas_expulsion_onset, gas_expulsion_timescale,
         t_ini, t_end, save_interval, run_number, save_path,
         gamma=1,
         mass_factor_exponent=0.2,
         truncation_parameter=1. / 3,
         gas_to_stars_mass_ratio=2.0,
         gas_to_stars_plummer_radius_ratio=1.0,
         plummer_radius=0.5 | units.parsec,
         dt=2000 | units.yr,
         temp_profile=0.5,
         Rref=1.0 | units.AU,
         Tref=280 | units.K,
         mu=2.3 | units.g / units.mol,
         filename=''):
    """Runs simulation with given parameters. Handles viscous growth (parametrized) and dynamical truncations.

    :param N: number of stars
    :param Rvir: virial radius of cluster
    :param Qvir: virial ratio of cluster
    :param alpha: turbulence parameter
    :param R: factor for initial characteristic radius of disks (see eq. 16)
    :param gas_presence: True if gas is present constantly through the evolution of the cluster
    :type gas_presence: bool
    :param gas_expulsion: True if gas presence with subsequent gas expulsion
    :type gas_expulsion: bool
    :param gas_expulsion_onset: time at which gas starts to be dissipated
    :param gas_expulsion_timescale: time that takes for the cluster to lose half its gas
    :param t_ini: initial time of simulation
    :param t_end: end time of simulation
    :param save_interval: interval to save simulation snapshots
    :param run_number: run number
    :param save_path: path to save results
    :param gamma: viscosity exponent
    :param mass_factor_exponent: mass dependence for disks in encounter
    :param truncation_parameter: disk truncantion dependence for disks in encounter
    :param gas_to_stars_mass_ratio: ratio of gas and stellar masses in cluster
    :param gas_to_stars_plummer_radius_ratio: ratio of Plummer radius of gas and stellar components
    :param plummer_radius: Plummer radius of stellar component
    :param dt: dt for recomputing disk sizes and checking energy error
    :param temp_profile: negative of temperature profile exponent (see eq. 8)
    :param Rref: reference distance from star at which Tref is given
    :param Tref: disk temperature at Rref
    :param mu: molar mass of gas in g/mol
    :param filename:
    """

    # This is necessary sometimes, depending on the input... uncomment if needed
    #gas_expulsion_onset = gas_expulsion_onset | units.yr
    #t_end = t_end | units.yr
    #Rvir = Rvir | units.parsec

    # Create path for results, if it does not exist
    path = "{0}/{1}/{2}/{3}/".format(save_path, str(N), str(alpha), str(run_number))
    try:
        os.makedirs(path)
    except OSError, e:
        if e.errno != 17:
            raise
        pass

    # Initialize
    if filename == '':  # Starting from scratch
        max_stellar_mass = 100 | units.MSun
        stellar_masses = new_kroupa_mass_distribution(N, max_stellar_mass)  # , random=False)
        disk_masses = 0.1 * stellar_masses
        converter = nbody_system.nbody_to_si(stellar_masses.sum() + disk_masses.sum(), Rvir)

        stars = new_plummer_model(N, converter)
        stars.scale_to_standard(converter, virial_ratio=Qvir)

        stars.stellar_mass = stellar_masses
        stars.initial_characteristic_disk_radius = (stars.stellar_mass.value_in(units.MSun) ** 0.5) * R | units.AU
        stars.initial_disk_mass = disk_masses
        stars.mass = stars.initial_disk_mass + stars.stellar_mass
        stars.viscous_timescale = viscous_timescale(stars, alpha, temp_profile, Rref, Tref, mu, gamma)
        stars.last_encounter = 0.0 | units.yr

        # Initializing the closest encounter distance and strongest truncation radius
        stars.closest_encounter = plummer_radius
        stars.strongest_truncation = plummer_radius

    else:  # Starting from saved snapshot
        stars = read_set_from_file(output_directory + filename + '.hdf5', 'amuse', close_file=True)
        converter = nbody_system.nbody_to_si(stars.mass.sum(), 1.695 * plummer_radius)

    write_set_to_file(stars,
                      '{0}/R{1}_{2}.hdf5'.format(path, Rvir.value_in(units.parsec),
                                                 int(t_ini.value_in(units.yr))),
                      'amuse')

    # The maximum truncation factors for the stars in the cluster.
    truncation_factor = truncation_parameter * (stars.mass / max(stars.mass)) ** mass_factor_exponent

    # Initializing the N-body code.
    gravity = ph4(converter)
    gravity.parameters.timestep_parameter = 0.01
    gravity.parameters.epsilon_squared = (100 | units.AU) ** 2
    gravity.particles.add_particles(stars)

    channel_from_gravity_to_framework = gravity.particles.new_channel_to(stars)
    channel_from_framework_to_gravity = stars.new_channel_to(gravity.particles)

    # Setting stopping conditions
    stopping_condition = gravity.stopping_conditions.collision_detection
    stopping_condition.enable()

    # If needed, add gas as a background Plummer potential
    if gas_presence or gas_expulsion:
        initial_gas_mass = gas_to_stars_mass_ratio * stars.mass.sum()
        gas = PlummerSpherePotential(a=plummer_radius * gas_to_stars_plummer_radius_ratio, M=initial_gas_mass)

        gas_gravity = Bridge(use_threading=False)
        gas_gravity.add_system(gravity, (gas,))
        cluster_gas_expulsion_timescale = gas_expulsion_timescale * plummer_radius.value_in(units.parsec)

        if t_ini == 0 | units.yr:
            gravity.particles.scale_to_standard(converter, virial_ratio=Qvir * (1.0 + gas_to_stars_mass_ratio))
            channel_from_gravity_to_framework.copy()

    # Get initial energy values
    t = 0 | units.yr
    E_ini = gravity.kinetic_energy + gravity.potential_energy

    if gas_presence or gas_expulsion:
        E_ini += np.sum(stars.mass * gas.get_potential_at_point(0, stars.x, stars.y, stars.z))

    # Save energy values to check for conservation
    E_handle = file('{0}/energy.txt'.format(path), 'a')
    Q_handle = file('{0}/virial.txt'.format(path), 'a')

    E_list = []
    Q_list = []

    # Evolve!
    while t < t_end - t_ini:
        # If gas_expulsion, reduce gas mass according to equation (11)
        if gas_expulsion and t > gas_expulsion_onset - t_ini:
            gas.M = initial_gas_mass / (1 + (t + t_ini - gas_expulsion_onset) / cluster_gas_expulsion_timescale)

        # Dealing with energy
        E_kin = gravity.kinetic_energy
        E_pot = gravity.potential_energy

        if gas_presence or gas_expulsion:
            E_pot += np.sum(stars.mass * gas.get_potential_at_point(0, stars.x, stars.y, stars.z))

        E_list.append([(E_kin + E_pot) / E_ini - 1])
        Q_list.append([-1.0 * E_kin / E_pot])

        # Evolve gas for half a time step
        if gas_presence or gas_expulsion:
            gas_gravity.kick_codes(dt / 2)

        # Update the collision radii of the stars based on the truncation factors and viscous spreading.
        gravity.particles.radius = disk_characteristic_radius(stars, t, gamma) / truncation_factor
        t += dt

        # Evolve gravity code for a full time step
        while gravity.model_time < t:
            gravity.evolve_model(t)

            if stopping_condition.is_set():
                # Dealing with encounter
                channel_from_gravity_to_framework.copy()
                encountering_stars = Particles(particles=[stopping_condition.particles(0)[0],
                                                          stopping_condition.particles(1)[0]])
                resolve_encounter(encountering_stars.get_intersecting_subset_in(stars), gravity.model_time + t_ini,
                                  mass_factor_exponent, truncation_parameter, gamma)
                channel_from_framework_to_gravity.copy_attributes(['radius'])

        # Evolve gas for half a time step
        if gas_presence or gas_expulsion:
            gas_gravity.kick_codes(dt / 2)

        channel_from_gravity_to_framework.copy()

        # Save on the corresponding intervals only
        if (t + t_ini).value_in(units.yr) % save_interval.value_in(units.yr) == 0:
            channel_from_gravity_to_framework.copy()

            np.savetxt(E_handle, E_list)
            np.savetxt(Q_handle, Q_list)

            E_list = []
            Q_list = []

            new_mass = disk_mass(stars, t + t_ini, gamma)
            new_radius = disk_characteristic_radius(stars, t + t_ini, gamma)

            stars.stellar_mass += stars.initial_disk_mass - new_mass
            stars.initial_disk_mass = new_mass
            stars.viscous_timescale *= (np.divide(new_radius, stars.initial_characteristic_disk_radius)) ** (2 - gamma)
            stars.initial_characteristic_disk_radius = new_radius
            stars.last_encounter = t + t_ini

            write_set_to_file(stars,
                              '{0}/R{1}_{2}.hdf5'.format(path, Rvir.value_in(units.parsec),
                                                         int((t + t_ini).value_in(units.yr))),
                              'amuse')

    if gas_presence or gas_expulsion:
        gas_gravity.stop()

    gravity.stop()
    E_handle.close()
    Q_handle.close()


def new_option_parser():
    """ Parses command line input for code.
    """
    from amuse.units.optparse import OptionParser

    result = OptionParser()

    # Cluster parameters
    result.add_option("-N", dest="N", type="int", default=2000,
                      help="number of stars [%default]")
    result.add_option("-R", dest="Rvir", type="float",
                      unit=units.parsec, default=0.5,
                      help="cluster virial radius [%default]")
    result.add_option("-Q", dest="Qvir", type="float", default=0.5,
                      help="virial ratio [%default]")
    result.add_option("-n", dest="run_number", type="int", default=0,
                      help="run number [%default]")
    result.add_option("-s", dest="save_path", type="string", default='.',
                      help="path to save the results [%default]")
    result.add_option("-i", dest="save_interval", type="int", default=50000 | units.yr,
                      help="time interval of saving a snapshot of the cluster [%default]")

    # Disk parameters
    result.add_option("-a", dest="alpha", type="float", default=1E-2,
                      help="turbulence parameter [%default]")
    result.add_option("-c", dest="R", type="float", default=30.0,
                      help="Initial disk radius [%default]")

    result.add_option("-e", dest="gas_expulsion_onset", type="float", default=0.6 | units.Myr,
                      help="the moment when the gas starts dispersing [%default]")
    result.add_option("-E", dest="gas_expulsion_timescale", type="float", default=0.1 | units.Myr,
                      help="the time after which half of the initial gas is expulsed assuming gas Plummer radius of 1 parsec [%default]")

    # Time parameters
    result.add_option("-I", dest="t_ini", type="int", default=0 | units.yr,
                      help="initial time [%default]")
    result.add_option("-t", dest="dt", type="int", default=2000 | units.yr,
                      help="time interval of recomputing circumstellar disk sizes and checking for energy conservation [%default]")
    result.add_option("-x", dest="t_end", type="float", default=2000000 | units.yr,
                      help="end time of the simulation [%default]")

    # Gas behaviour
    result.add_option("-l", dest="gas_presence", action="store_false", default=False,
                      help="gas presence [%default]")
    result.add_option("-k", dest="gas_expulsion", action="store_false", default=False,
                      help="gas expulsion [%default]")

    return result


# The name of the directory where the output will be directed. Does not have to exist before running the code
output_directory = '.'

if __name__ in ('__main__', '__plot__'):
    o, arguments = new_option_parser().parse_args()

    start = datetime.datetime.now()

    print(
        'Starting run #{0} for cluster with {1} stars, Rvir {2} at {3}/{4}/{5} {6}:{7}:{8}'.format(
            o.__dict__['run_number'],
            o.__dict__['N'],
            o.__dict__['Rvir'],
            start.day, start.month,
            start.year, start.hour,
            start.minute,
            start.second))

    main(**o.__dict__)

    elapsed = datetime.datetime.now()

    print(
        'Finished at {0}/{1}/{2} {3}:{4}:{5}'.format(elapsed.day, elapsed.month, elapsed.year, elapsed.hour,
                                                     elapsed.minute,
                                                     elapsed.second))
    print("******************************************************************************")
