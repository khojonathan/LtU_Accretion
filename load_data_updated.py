import numpy as np
import os
import h5py
from multiprocessing import Pool
from LtU_get_property import get_particle_property_LTU
import arepo_package as arepo_package

# -------------------------------------------------------
# Paths and box definitions
# -------------------------------------------------------

Filepath = '/project/torrey-group/jkho/LtU_accretion'
Boxes = ['Zooms', 'Constrained', 'Low_mass_seeds']
outputpath = 'output'

# Redshifts
redshifts = np.arange(20, 5, -1)
a = 1.0 / (1.0 + redshifts)
h = 0.6774

# Mapping Simpaths index → HDF5 group
group_labels = ["AGN_S", "AGN_NS", "NAGN_S", "NAGN_NS"]
subnames = ["Bondi", "FF", "modFF"]

# Simpaths should be the same for all boxes now
Simpaths = [
        ['Bondi_AGN_stellar_strict_Repo',     'FF_AGN_stellar_strict_Repo',    'modFF_AGN_stellar_strict_Repo'],
        ['Bondi_AGN_lowstellar_strict_Repo',  'FF_AGN_lowstellar_strict_Repo', 'modFF_AGN_lowstellar_strict_Repo'],
        ['Bondi_noAGN_stellar_strict_Repo',   'FF_noAGN_stellar_strict_Repo',  'modFF_noAGN_stellar_strict_Repo'],
        ['Bondi_noAGN_lowstellar_strict_Repo','FF_noAGN_lowstellar_strict_Repo',
         'modFF_noAGN_lowstellar_strict_Repo']
    ]

# -------------------------------------------------------
# Load one simulation across redshifts (runs in a worker)
# -------------------------------------------------------

def process_sim(Box, label, subname, sim):

    print(f"  Loading: {Box}/{sim}", flush = True)

    basePath = f"{Filepath}/{Box}/Repositioning/{sim}/{outputpath}/"

    # Lists per redshift
    Masses = []
    Mdots = []
    Accretion_growth = []
    Edd = []
    Progs = []
    HostHaloStellarMasses = []
    HostHaloSFR = []
    # rho = []

    for i, z in enumerate(redshifts):

        BH_Mass,zout  = get_particle_property_LTU(basePath,'BH_Mass',p_type=5, desired_redshift=z)
        BH_Mdot,zout  = get_particle_property_LTU(basePath,'BH_Mdot',p_type=5, desired_redshift=z)
        BH_QM,zout    = get_particle_property_LTU(basePath,'BH_CumMassGrowth_QM',p_type=5, desired_redshift=z)
        BH_RM,zout    = get_particle_property_LTU(basePath,'BH_CumMassGrowth_RM',p_type=5, desired_redshift=z)
        BH_Edd,zout   = get_particle_property_LTU(basePath,'BH_MdotEddington',p_type=5, desired_redshift=z)
        BH_Progs,zout = get_particle_property_LTU(basePath,'BH_Progs',p_type=5, desired_redshift=z)
        # BH_rho,zout   = get_particle_property_LTU(basePath,'SubfindDensity',p_type=5, desired_redshift=z)

        if (BH_Mass.shape == (1,)) and (BH_Mass[0]) == 0 or (np.abs(zout - z) > 1):

            print(f"{Box}/{sim}: Appending NaNs at z = {z}!", flush = True)

            Masses.append(np.nan)
            Mdots.append(np.nan)
            Accretion_growth.append(np.nan)
            Edd.append(np.nan)
            Progs.append(0)
            HostHaloStellarMasses.append(np.nan)
            HostHaloSFR.append(np.nan)

            continue

        most_massive_ind = np.argmax(BH_Mass)

        # Host = the subhalo containing the BH (each particle belongs to at most one subhalo)
        SubhaloLenType = arepo_package.get_subhalo_property(basePath,"SubhaloLenType",z)[0]

        if isinstance(SubhaloLenType, dict): # No subhalos in this snapshot
            host_subhalos = np.array([], dtype=int)
        else:
            host_subhalos = np.flatnonzero(SubhaloLenType[:, 5] > 0)

        if len(host_subhalos) == 0:
            print(f"{Box}/{sim}: No subhalo contains a BH at z = {z}, appending NaNs for host properties!", flush = True)
            HostHaloStellarMasses.append(np.nan)
            HostHaloSFR.append(np.nan)
        else:
            if len(host_subhalos) > 1:
                print(f"{Box}/{sim}: {len(host_subhalos)} subhalos contain BHs at z = {z}, using the one with the largest BH mass!", flush = True)
                SubhaloBHMass = arepo_package.get_subhalo_property(basePath,"SubhaloBHMass",z)[0]
                host_index = host_subhalos[np.argmax(SubhaloBHMass[host_subhalos])]
            else:
                host_index = host_subhalos[0]

            SubhaloMassType = arepo_package.get_subhalo_property(basePath,"SubhaloMassType",z)[0]
            SubhaloSFR = arepo_package.get_subhalo_property(basePath,"SubhaloSFR",z)[0]
            HostHaloStellarMasses.append(SubhaloMassType[host_index, 4] * 1e10 / h)
            HostHaloSFR.append(SubhaloSFR[host_index])

        Masses.append(BH_Mass[most_massive_ind] * 1e10 / h)
        Mdots.append(BH_Mdot[most_massive_ind] * 1e10 / 0.978)
        Accretion_growth.append((BH_QM[most_massive_ind] + BH_RM[most_massive_ind]) * 1e10 / h)
        Edd.append(BH_Edd[most_massive_ind] * 1e10 / 0.978)
        Progs.append(BH_Progs[most_massive_ind])
        # rho.append(BH_rho[most_massive_ind] * (1e10/h) / (a[i]/h)**3)

    data = {
        "Masses": np.array(Masses),
        "Mdots": np.array(Mdots),
        "Accretion_growth": np.array(Accretion_growth),
        "Edd": np.array(Edd),
        "Progs": np.array(Progs),
        # Names kept for compatibility, but these are now the BH's host subhalo values
        "HostHaloStellarMasses": np.array(HostHaloStellarMasses),
        "HostHaloSFR": np.array(HostHaloSFR),
        # "rho": np.array(rho),
    }

    return Box, label, subname, data

def process_sim_star(task):
    return process_sim(*task)

# -------------------------------------------------------
# Run all simulations in parallel, then write output
# -------------------------------------------------------

if __name__ == "__main__":

    # One task per (Box, group, accretion model) simulation
    tasks = [(Box, label, subname, sim)
             for Box in Boxes
             for label, simset in zip(group_labels, Simpaths)
             for subname, sim in zip(subnames, simset)]

    # Size the pool from the Slurm allocation, not the whole node
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))
    print(f"### Running {len(tasks)} simulations on {nproc} processes", flush = True)

    results = {Box: {label: {} for label in group_labels} for Box in Boxes}

    with Pool(processes=nproc) as pool:
        for n, (Box, label, subname, data) in enumerate(pool.imap_unordered(process_sim_star, tasks), 1):
            results[Box][label][subname] = data
            print(f"### Finished {Box}/{label}/{subname} ({n}/{len(tasks)})", flush = True)

    # -------------------------------------------------------
    # Write output HDF5 files (parent process only)
    # -------------------------------------------------------

    for Box in Boxes:

        outname = f"output/{Box}_correctd_repo.hdf5" # f"output/{Box}_25.hdf5"
        print(f"\n### Writing output for {Box} → {outname}",flush = True)

        with h5py.File(outname, "w") as hf:
            for label in group_labels:
                label_group = hf.create_group(label)
                for subname in subnames:
                    subgroup = label_group.create_group(subname)
                    for name, values in results[Box][label][subname].items():
                        subgroup.create_dataset(name, data=values)

        print(f"### Finished writing {outname}\n", flush = True)

    print("\nAll boxes completed successfully.", flush = True)
