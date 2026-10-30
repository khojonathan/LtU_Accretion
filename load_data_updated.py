import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LogNorm
import numpy as np
import os
import h5py
from astropy import constants as c
from astropy import units as u
import matplotlib as mpl
from torreylabtools import helpers
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

# Simpaths should be the same for all boxes now
Simpaths = [
        ['Bondi_AGN_stellar_strict_Repo',     'FF_AGN_stellar_strict_Repo',    'modFF_AGN_stellar_strict_Repo'],
        ['Bondi_AGN_lowstellar_strict_Repo',  'FF_AGN_lowstellar_strict_Repo', 'modFF_AGN_lowstellar_strict_Repo'],
        ['Bondi_noAGN_stellar_strict_Repo',   'FF_noAGN_stellar_strict_Repo',  'modFF_noAGN_stellar_strict_Repo'],
        ['Bondi_noAGN_lowstellar_strict_Repo','FF_noAGN_lowstellar_strict_Repo',
         'modFF_noAGN_lowstellar_strict_Repo']
    ]

# -------------------------------------------------------
# Loop through boxes
# -------------------------------------------------------

for Box in Boxes:

    # -------------------------------------------------------
    # Prepare output HDF5 file
    # -------------------------------------------------------

    outname = f"output/{Box}_correctd_repo.hdf5" # f"output/{Box}_25.hdf5"
    print(f"\n### Writing output for {Box} → {outname}",flush = True)

    hf = h5py.File(outname, "w")

    # -------------------------------------------------------
    # Loop through the 4 categories (AGN_S, AGN_NS, ...)
    # -------------------------------------------------------
    for gi, label in enumerate(group_labels):

        print(f"\n### Processing group: {label}",flush = True)
        label_group = hf.create_group(label)

        # Each element: [Bondi_path, FF_path, modFF_path]
        bondi_path, ff_path, modff_path = Simpaths[gi]

        subnames = ["Bondi", "FF", "modFF"]
        simset = [bondi_path, ff_path, modff_path]

        # ---------------------------------------------------
        # Loop over Bondi, FF, modFF
        # ---------------------------------------------------
        for subname, sim in zip(subnames, simset):

            print(f"  Loading: {sim}", flush = True)
            subgroup = label_group.create_group(subname)

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

            # ---------------------------------------------------
            # Load data across redshifts
            # ---------------------------------------------------
            for i, z in enumerate(redshifts):

                BH_Mass,zout  = get_particle_property_LTU(basePath,'BH_Mass',p_type=5, desired_redshift=z)
                BH_Mdot,zout  = get_particle_property_LTU(basePath,'BH_Mdot',p_type=5, desired_redshift=z)
                BH_QM,zout    = get_particle_property_LTU(basePath,'BH_CumMassGrowth_QM',p_type=5, desired_redshift=z)
                BH_RM,zout    = get_particle_property_LTU(basePath,'BH_CumMassGrowth_RM',p_type=5, desired_redshift=z)
                BH_Edd,zout   = get_particle_property_LTU(basePath,'BH_MdotEddington',p_type=5, desired_redshift=z)
                BH_Progs,zout = get_particle_property_LTU(basePath,'BH_Progs',p_type=5, desired_redshift=z)
                # BH_rho,zout   = get_particle_property_LTU(basePath,'SubfindDensity',p_type=5, desired_redshift=z)

                if (BH_Mass.shape == (1,)) and (BH_Mass[0]) == 0 or (np.abs(zout - z) > 1):

                    print("Appending NaNs!")
                                        
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
                    print(f"No subhalo contains a BH at z = {z}, appending NaNs for host properties!", flush = True)
                    HostHaloStellarMasses.append(np.nan)
                    HostHaloSFR.append(np.nan)
                else:
                    if len(host_subhalos) > 1:
                        print(f"{len(host_subhalos)} subhalos contain BHs at z = {z}, using the one with the largest BH mass!", flush = True)
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

            # ---------------------------------------------------
            # Store datasets
            # ---------------------------------------------------
            subgroup.create_dataset("Masses", data=np.array(Masses))
            subgroup.create_dataset("Mdots",  data=np.array(Mdots))
            subgroup.create_dataset("Accretion_growth", data=np.array(Accretion_growth))
            subgroup.create_dataset("Edd", data=np.array(Edd))
            subgroup.create_dataset("Progs", data=np.array(Progs))
            # Names kept for compatibility, but these are now the BH's host subhalo values
            subgroup.create_dataset("HostHaloStellarMasses",data=np.array(HostHaloStellarMasses))
            subgroup.create_dataset("HostHaloSFR",data=np.array(HostHaloSFR))
            # subgroup.create_dataset("rho", data=np.array(rho))

    hf.close()
    print(f"### Finished writing {outname}\n", flush = True)

print("\nAll boxes completed successfully.", flush = True)
