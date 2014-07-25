# -*- coding: utf-8 -*-
import __main__
__main__.pymol_argv = ['pymol','-qc'] # Pymol: quiet and no GUI
import logging
import re
from collections import defaultdict
from functools import partial
import numpy as np
import pandas as pd
from pandas.core.reshape import melt
from numba import jit, autojit, double, void, int_
from Bio.PDB.Polypeptide import CaPPBuilder
from rdkit import Chem
from MMTK import surfm

@autojit
def pairwise_dist(X):
    """ 
    Calculates the pairwise distance of a M x N matrix

    Args:
        X: a numpy M X N array
    Returns:
        A M X M numpy array with pairwise distances
    """
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D   

def extractSequenceFromPDBFile(pdbfile):
    """ 
    Parses the sequences (SEQRES records) from a pdbfile.

    Args:
        pdbfile: Path to a pdb file

    Returns:
        A dictionary Chain: Sequence for all sequences of all chains
    """
    f = open(pdbfile)
    chains = defaultdict(list)
    for line in f:
        # in seqres block
        if line.startswith('SEQRES'):
            tokens = [tok for tok in re.split(r'[\s\|]+', line) if tok]
            chain,resns = tokens[2], tokens[4:]
            chains[chain].extend(resns)            
    return {k: ''.join(chains[k]) for k in chains}



class _Atom(object):
    """
    Dummy Atom class to wrap the neccessary properties for MMTK
    """
    def __init__(self, idx, vdw, pos):
        self.vdW_radius = vdw
        self.pos = pos
        self.idx = idx
    def position(self):
        return self.pos
    def __str__(self):
        return "Atom #%s [%s - %s]" % (self.idx, self.pos, self.vdW_radius)
    def __repr__(self):
        return self.__str__()    
        
class PPIBitstrings(object):
    """ 
    Class for creating surface ppi bitstrings.
        
    Processes the pdb, calculates the surface residues, the distance matrix and the bitstrings for the surface residues on init.       
    """

    # Amino acid dictionary
    AADICT = {
			'ALA': (0, 'A'),
            'ARG': (1, 'R'),
            'ASN': (2, 'N'),
            'ASP': (3, 'D'),
            'CYS': (4, 'C'),
            'GLU': (5, 'E'),
            'GLN': (6, 'Q'),
            'GLY': (7, 'G'),
            'HIS': (8, 'H'),
            'ILE': (9, 'I'),
            'LEU': (10, 'L'),
            'LYS': (11, 'K'),
            'MET': (12, 'M'),
            'PHE': (13, 'F'),
            'PRO': (14, 'P'),
            'SER': (15, 'S'),
            'THR': (16, 'T'),
            'TRP': (17, 'W'),
            'TYR': (18, 'Y'),
            'VAL': (19, 'V')
        }

    def __init__(self, pdb, pdbpath, fastaSequences, interactionRadius=4.5, surfMethod='mmtk', surfaceArea=2.5, solventRadius=1.4, pointDensity=1026, logger=logging.getLogger()):
        """ 
        Creates a new PPIBitstrings instance. 
        Processes the pdb, calculates the surface residues, the distance matrix and the bitstrings for the surface residues.
        
        Args:
            pdb: A BioPython structure object
            fastaSequences: Sequences as returned by PPI.extractSequenceFromPDBFile(pdbfile)
            interactionRadius: Interaction radius used for PPI/NPPI determination (default: 4.5)
            surfMethod: The method used to calculate the surface residues. Can be mmtk or pymol (default: mmtk)
            surfaceArea: Minimum surface area required to be treated as surface residue (default: 2.5)
            solventRadius: Solvent radius used by MMTK (default: 1.4)
            pointDensity: Point density used by MMTK (default: 1026)
            logger: Logging instance to use as logger (default: logging.getLogger())
        """
        self._df = None
        self._distMatrix = None
        self._peptideChains = None
        self._phosphateChains = None

        self._logger = logger

        if surfMethod == 'pymol':
            surfFunc = self._getSurfaceResiduesPyMOL
            logger.info('Loading PyMol')
            import pymol
            pymol.finish_launching()
            self._cmd = pymol.cmd
            self._stored = pymol.stored
        else:
            surfFunc = self._getSurfaceResiduesMMTK

        self._pdbpath = pdbpath
        self._interactionRadius = interactionRadius
        self._surfaceArea = surfaceArea
        self._solventRadius = solventRadius
        self._pointDensity = pointDensity
        self._pdb = pdb
        self._periodicTable = Chem.GetPeriodicTable()
        self._sequences = fastaSequences
        self._processPDB()   
        self._logger.info('Using %s for surface calculation' % surfMethod)
        self._surfaceResidues = surfFunc()
        self._calcDistMatrix()
        self._calcNeighbours()
    
    def _processPDB(self):
        """ Processes the PDB file, i.e. adds all relevant atoms to a dataframe and determines the peptide an phosphate chains"""
        self._logger.info("Processing PDB")
        ppb=CaPPBuilder()
        d = []
        peptide_chains = {}
        phosphate_chains = set()
        # Loop over all chains
        for chain_idx, chain in enumerate(self._pdb[0]):
            isPeptideChain = False
            isPhosphateChain = False
            # try to create peptide sequence
            pp = ppb.build_peptides(chain)
            if pp:
                # tag chain as peptide chain
                isPeptideChain = True
                peptide_chains[chain.get_id()] = pp[0].get_sequence().tostring()
            # loop over residues in chain
            for residue in chain:
                resn = residue.get_resname()                
                if resn in ['PTR', 'TPO', 'SEP']:
                    # Chain contains a phospho-residue; tag as phosphateChain
                    isPhosphateChain = True
                    phosphate_chains.add(chain.get_id())
                # process atoms only if residue is not water and is part of a peptide or phospho chain
                if residue.get_id()[0] != 'W' and (isPeptideChain or isPhosphateChain):
                    resi = residue.get_id()[1]
                    inscode = residue.get_id()[2].strip()
                    hasPhosphate = False

                    for atom in residue:
                        vdw = -1
                        if isPeptideChain:
                            elem = atom.element
                            if elem:
                                elem = elem if len(elem)==1 else elem[0]+elem[1].lower()
                                vdw = self._periodicTable.GetRvdw(self._periodicTable.GetAtomicNumber(elem))
                        coords = atom.get_coord()
                        sn = atom.get_serial_number()
                        # append to dataframe
                        d.append((chain.get_id().strip(), chain_idx, resn, resi, inscode, sn, atom.get_name(), isPeptideChain, isPhosphateChain, coords[0], coords[1], coords[2], vdw))

                        if resn in ['PTR', 'TPO', 'SEP'] and atom.get_name().strip() == 'P' and len(coords)==3:
                            hasPhosphate = True

                if resn in ['PTR', 'TPO', 'SEP'] and not hasPhosphate:
                    # del residue because no annotated phosphate
                    d = d[:-len(residue)]


        if len(d) == 0:
            raise Exception('No amino acids found.')
        # save list to dataframe
        data = np.zeros((len(d), ), dtype=[('chain', 'a1'), ('chain_idx', 'a1'), ('resn', 'a3'), ('resi', 'i4'), ('inscode', 'a1'), ('sn', 'i4'), ('an', 'a4'), ('peptideChain', 'b'), ('phosphateChain', 'b'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('vdw', 'f4')])
        data[:] = d
        idf = pd.DataFrame(data)
        idf[['peptideChain', 'phosphateChain']] = idf[['peptideChain', 'phosphateChain']].astype('bool')
        self._df = idf
        self._peptideChains = peptide_chains
        self._phosphateChains = phosphate_chains

    def _getSurfaceAtomIndices(self):
        """ Calculates the surface atoms """
        self._logger.info("Calculating surface")
        surface_atoms = [] 

        if 'surfaceArea' not in self._df.columns:
            sa = np.zeros((len(self._df),), dtype=np.float)
            saS = pd.Series(index=self._df.index, data=sa, name='surfaceArea')
            self._df = self._df.join(saS)

        # calculate surface for each peptide chain separately
        for chain in self._peptideChains:
            atoms = []
            # indices of all atoms in peptide chains
            atms = self._df.loc[(self._df.peptideChain==True) & ( self._df.chain==chain), ['vdw', 'x', 'y', 'z']]

            # wrap in _Atom class for MMTK
            for tpl in atms.itertuples(): 
                row_idx, vdw, x, y, z = tpl
                atoms.append(_Atom(row_idx, vdw, (x, y, z)))

            # run MMTK to calculate surface
            smap = surfm.surface_atoms(atoms, solvent_radius=self._solventRadius, point_density=self._pointDensity, ret_fmt=1)  
            for i in xrange(len(atoms)):
                area = smap[atoms[i]]
                self._df.loc[atoms[i].idx, 'surfaceArea'] = area
                # add atoms to list if surface area is over the threshold
                if area >= self._surfaceArea:
                    surface_atoms.append(atoms[i].idx)
        self._logger.info("...Found %s surface atoms" % len(surface_atoms))
        if not 'surfaceAtom' in self._df.columns:
            sf = np.zeros((len(self._df),), dtype=np.bool)
            sfS = pd.Series(index=self._df.index, name='surfaceAtom')
            self._df = self._df.join(sfS)
        self._df.loc[surface_atoms, 'surfaceAtom'] = True
        return surface_atoms

    def _findSurfaceResidues(self, objSel="(all)", cutoff=2.5):
        """
	    findSurfaceResidues
		    finds those residues on the surface of a protein
		    that have at least 'cutoff' exposed A**2 surface area.
 
	    PARAMS
		    objSel (string)
			    the object or selection in which to find
			    exposed residues
			    DEFAULT: (all)
 
		    cutoff (float)
			    your cutoff of what is exposed or not. 
			    DEFAULT: 2.5 Ang**2
 
		    asSel (boolean)
			    make a selection out of the residues found
 
	    RETURNS
		    (list: (chain, resv ) )
			    A Python list of residue numbers corresponding
			    to those residues w/more exposure than the cutoff.
 
	    """
        # load pdb
        self._cmd.reinitialize()
        self._cmd.load(self._pdbpath)

        tmpObj = "__tmp"
        self._cmd.create(tmpObj, objSel + " and polymer")
        self._cmd.set("dot_solvent")
        self._cmd.get_area(selection=tmpObj, load_b=1)

        # threshold on what one considers an "exposed" atom (in A**2):
        self._cmd.remove(tmpObj + " and b < " + str(cutoff))

        self._stored.tmp_dict = {}
        self._cmd.iterate(tmpObj, "stored.tmp_dict[(chain,resi,b)]=1")
        exposed = self._stored.tmp_dict.keys()
        exposed.sort()

        self._cmd.delete(tmpObj)

        return exposed

    def _getSurfaceResiduesMMTK(self):
        """ 
        Finds the according residues for a list of surface atoms using MMTK.

        Args:
            surfaceAtoms: List of indices of surface atoms
        """
        self._logger.info("Searching surface residues")
        surfaceAtoms = self._getSurfaceAtomIndices() 
        # add surface column if it doesn't exist
        if not 'surface' in self._df.columns:
            sf = np.zeros((len(self._df),), dtype=np.bool)
            sfS = pd.Series(index=self._df.index, data=sf, name='surface')
            self._df = self._df.join(sfS)

        # group found surface atoms by chain, residue id and insertion code
        surfResis = self._df.loc[surfaceAtoms].groupby(['chain', 'resi', 'inscode']).groups.keys()
        # flag matching groups as surface residue
        for v in surfResis:
            self._df.loc[(self._df['chain'] == v[0]) & (self._df['resi'] == v[1]) & (self._df['inscode'] == v[2]), 'surface'] = True
        self._logger.info("...%s surface atoms resolved to %s residues in %s chains" % (len(surfaceAtoms), len(surfResis), len({x[0] for x in surfResis})))
        return surfResis

    def _getSurfaceResiduesPyMOL(self):
        """ 
        Finds the according residues for a list of surface atoms using PyMOL.

        Args:
            surfaceAtoms: List of indices of surface atoms
        """
        self._logger.info("Searching surface residues")
        surfResis = self._findSurfaceResidues(cutoff=self._surfaceArea)
        # add surface column if it doesn't exist
        if not 'surface' in self._df.columns:
            sf = np.zeros((len(self._df),), dtype=np.bool)
            sfS = pd.Series(index=self._df.index, data=sf, name='surface')
            self._df = self._df.join(sfS)
        if 'surfaceArea' not in self._df.columns:
            sa = np.zeros((len(self._df),), dtype=np.float)
            saS = pd.Series(index=self._df.index, data=sa, name='surfaceArea')
            self._df = self._df.join(saS)

        # flag matching groups as surface residue
        for v in surfResis:
            chain, resn, b = v
            src = re.search("(\d+)(\D)", resn)
            resi = resn
            inscode = ''
            if src:
                resi = src.groups()[0]
                inscode = src.groups()[1]            
            self._df.loc[(self._df['chain'] == chain.strip()) & (self._df['resi'] == int(resi)) & (self._df['inscode'] == inscode), ['surface', 'surfaceArea']] = [True, b]
        self._logger.info("... found %s residues in %s chains" % (len(surfResis), len({x[0] for x in surfResis})))
        return surfResis

    def _calcDistMatrix(self):
        """ Calculates the distance matrix for all surface residues, c alpha atoms or phospho residues. """
        self._logger.info("Calculating distance matrix")

        # coordinates of all surface residues, c alpha atoms or phospho residues
        # bitstrings are calculated only for surface residues but for neighbourhood determination it is neccessary to include the c alphas
        surf = self._df.loc[(self._df['surface'] == True) | (self._df['an'] == 'CA') | (self._df['resn'].isin(['PTR', 'TPO', 'SEP'])), ['x', 'y', 'z']]
        surfIndex = surf.index

        # calculate distance matrix
        self._distMatrix = pd.DataFrame(pairwise_dist(np.array(surf)), index=surfIndex, columns=surfIndex)

    def _calcNeighbours(self):
        """ Calculates the neighbours and creates the bitstrings """
        self._logger.info("Creating neighbourhood bitstrings")  

        bitstrings = []

        # apply BS func on all c alphas in peptide chains
        cas = self._df.loc[(self._df['an'] == 'CA') & (self._df['peptideChain'] == True) & (self._df['surface'] == True), ['chain', 'resi', 'resn', 'inscode']]
        for tpl in cas.itertuples():
            idx, chain, resi, resn, inscode = tpl

            # other c alphas in same chain, including insertion mutants (i.e. same resi but different inscode)
            others = self._df.loc[(self._df['an']=='CA') & (-(self._df['resi']==resi) | -(self._df['inscode']==inscode)) & (self._df['chain']==chain)].index
            first = second = None            
            # copy distance values from matrix and sort
            distances = self._distMatrix.loc[idx, others].copy()
            distances.sort()
            # first two entries
            minEntries = self._df.loc[distances.iloc[:2].index, 'resn']
            first, second = minEntries

            bs = np.NaN
            # set the bits            
            if resn in PPIBitstrings.AADICT:
                bs = np.zeros(60, dtype=np.int)
                bs[PPIBitstrings.AADICT[resn][0]] = np.int(1)
                for k,v in enumerate([first, second]):
                    if v in PPIBitstrings.AADICT:
                        bs[(k+1)*20 + PPIBitstrings.AADICT[v][0]] = np.int(1)   
            bitstrings.append(bs)
        
        # create series
        s = pd.Series(bitstrings, index=cas.index, name='bitstring', dtype=np.object)
        # join to our df
        self._df = self._df.join(pd.DataFrame(s))       

    def _getPhosphoAtomIndices(self):
        """ 
        Finds all atoms within a phosphate resdiue.

        Returns:
            A dataframe containing all phosphate residue atoms
        """
        self._logger.info("Searching Phosphate residues")
        phosphoAtoms = self._df.loc[self._df['resn'].isin(['PTR', 'TPO', 'SEP'])]
        self._logger.info("...Found %s atoms" % len(phosphoAtoms))
        return phosphoAtoms
         
    def _getHotSpotResidues(self, phosphoAtoms):
        """ 
        Calculates the hotspot residues, i.e. surface residues within a certain distance to the phospho atoms

        Args:
            phosphoAtoms: Dataframe of all phospho residues
        """
        self._logger.info("Calculating hotspot residues")
        hotspots = []
        # Group the phospho atoms by chain, residue id and insertion code to get the distinct residues
        phosphoResis = phosphoAtoms.groupby(['chain', 'resi', 'inscode']).groups
        
        # key is a tuple (chain, residue id, insertion code)
        for key in phosphoResis:
            chain, resi, inscode = key
            # get atoms for this residue
            pAtomIndices = phosphoResis[key]
            # get all distances
            pDistances = self._distMatrix.loc[pAtomIndices]
            # get distances <= 4.5
            matchingAtomIndicesNP = np.argwhere(pDistances <= self._interactionRadius)[:,1]
            matchingAtomLabels = self._distMatrix.index[matchingAtomIndicesNP]
            # only surface residues within a different chain
            matchingResis = self._df.loc[matchingAtomLabels].loc[(self._df['surface'] == True) & (self._df['chain'] != chain) & (self._df['peptideChain'] == True)].groupby(['chain', 'resi', 'inscode']).groups.keys()
            hotspots.extend([(key, matchingResis)]) 
        self._logger.info("...found %s residues in %s chains" % (sum([len(x[1]) for x in hotspots]), len({x[0] for x in hotspots})))
        return hotspots
            
    def _calculatePhosphoInteractions(self):
        """ Calculates the phosphate interactions and flags the according entries in the dataframe. """

        self._logger.info("Calculating interactions")
        # Find phospho atoms
        phosphoAtoms = self._getPhosphoAtomIndices()
        # Find hotspot resis
        hotspotResis = self._getHotSpotResidues(phosphoAtoms)

        # Create neccessary columns if the don't exist
        if not 'PPI' in self._df.columns:
            ppi = np.zeros((len(self._df),), dtype=np.bool)
            ppiS = pd.Series(index=self._df.index, data=ppi, name='PPI')
            self._df = self._df.join(ppiS)
        if not 'interactingChain' in self._df.columns:
            ic = np.zeros((len(self._df),), dtype=np.bool)
            icS = pd.Series(index=self._df.index, data=ic, name='interactingChain')
            self._df = self._df.join(icS)
        if not 'ptype' in self._df.columns:
            pt = np.zeros((len(self._df),), dtype=np.str)
            ptS = pd.Series(index=self._df.index, data=pt, name='ptype')
            self._df = self._df.join(ptS)            
        if not 'pres' in self._df.columns:
            pr = np.zeros((len(self._df),), dtype=np.str)
            prS = pd.Series(index=self._df.index, data=pr, name='pres')
            self._df = self._df.join(prS)      

        # flag entries as PPI/NPPI
        interactingChains = set()
        for entry in hotspotResis:
            p,rs  = entry     
            ptype = self._df.loc[(self._df['chain'] == p[0]) & (self._df['resi'] == p[1]) & (self._df['inscode'] == p[2]), 'resn'].unique()[0]
            pres = '%s %s %s' % (p[0], ptype, p[1])
            for r in rs:     
                interactingChains.add(r[0])
                self._df.loc[(self._df['chain'] == r[0]) & (self._df['resi'] == r[1]) & (self._df['inscode'] == r[2]), ['PPI', 'ptype', 'pres']] = (True, ptype, pres)
        # flag corresponding chains as interacting chains
        for c in interactingChains:
            self._df.loc[self._df['chain'] == c, 'interactingChain'] = True


    def _getBitStrings(self, interactingOnly=False):
        """ 
        Returns a dataframe with the bitstrings.

        Args:
            interactingOnly: Return only bitstrings with the interactingChain flag. (default: False)

        Returns:
            A dataframe with the bitstrings
        """
        valid = set()
        if interactingOnly:
            # get interacting chains
            interactingChains = self._df.loc[self._df.interactingChain].chain.unique()
        bsv = self._df.loc[-(self._df.bitstring.isnull())].sort(['chain', 'resi', 'inscode']).groupby('chain', sort=False).bitstring.values
        for chain in self._peptideChains:
            if chain not in bsv:
                invalid = True
                continue
            bs1 = np.concatenate(bsv[chain]).tostring()
            invalid = False
            # remove non-interacting chains if neccessary
            if interactingOnly and chain not in interactingChains:
                invalid = True
                continue
            # check other chains for sequence identity
            for other in valid:
                bs2 = np.concatenate(bsv[other]).tostring()
                # triple check if fasta sequence in pdb, experimental sequence and bitstrings are identical
                if self._sequences: # if fasta available
                    if self._sequences[chain] == self._sequences[other] and self._peptideChains.has_key(chain) and self._peptideChains.has_key(other) and self._peptideChains[chain] == self._peptideChains[other] and bs1 == bs2:
                        invalid = True
                else:
                    if self._peptideChains.has_key(chain) and self._peptideChains.has_key(other) and self._peptideChains[chain] == self._peptideChains[other] and bs1 == bs2:
                        invalid = True
            if not invalid:
                valid.add(chain)

        valid = list(valid)  
        if interactingOnly:
            return self._df.loc[(self._df.interactingChain==True) & (self._df.chain.isin(valid)) & (self._df['surface']==True) & (-self._df['bitstring'].isnull())]
        return self._df.loc[(self._df.chain.isin(valid)) & (self._df['surface']==True) & (-self._df['bitstring'].isnull())]

    def getTrainingBitstrings(self):
        """
        Returns the bitstrings for training

        Returns:
            The bitstrings used for training
        """
        self._logger.info("Calculating training subset")
        self._calculatePhosphoInteractions()  
        self._logger.info("Done")
        return self._getBitStrings(interactingOnly=True)

    def predictPPIs(self, classifier):
        """ 
        Predicts the classes for the bitstrings.

        Args:
            classifier: A sklearn classifier

        Returns:
            A dataframe with predictions assigned
        """
        
        self._logger.info("Predicting PPIs")
        # Get all bitstrings
        bitstringDF = self._getBitStrings()
        X_test = np.vstack(bitstringDF['bitstring'])
        bsidx = bitstringDF.index
        vals = np.zeros((len(self._df), 2)) # class, probPPI
        vals[:,1] = 1 # per default NPPI probability is 1
        proba = classifier.predict_proba(X_test)        
        vals[bsidx,:] = proba * 100

        if not 'probNPPI' in self._df.columns:
            df = pd.DataFrame(vals, columns=['probNPPI', 'probPPI'], index=self._df.index)
            self._df = self._df.join(df)

        # copy CA vals to remaining atoms of resi
        for key in self._df.loc[bsidx].groupby(['chain', 'resi', 'inscode']).groups.iterkeys():
            chain, resi, inscode = key
            v = self._df.loc[(self._df['chain']==chain) & (self._df['resi'] == resi) & (self._df['inscode'] == inscode) & (self._df['an'] == 'CA'), ['probNPPI', 'probPPI']]
            self._df.loc[(self._df['chain']==chain) & (self._df['resi'] == resi) & (self._df['inscode'] == inscode), ['probNPPI', 'probPPI']] = v
        return self._df

    def getGetSurfaceBins(self, gridspacing=10.0):
        """ 
        Bins the surface residues ans calculates the mean probability for each bin 
        
        Args:
            gridspacing: The grid spacing to use in Angstrom (default 10.0)

        Returns:
            A dataframe with mean PPI probability for each bin
        """
        self._logger.info("Calculating probabilities per bin")
        if not 'probNPPI' in self._df.columns:
            raise Exception('predictPPIs has to be called before getSurfaceBins')

        minMax = self._df.loc[self._df.surfaceAtom==True, ['x', 'y', 'z']].describe().loc[['min', 'max'],:].transpose()
        minVal = np.floor(minMax['min']).min()
        maxVal = np.ceil(minMax['max']).max()
        r = np.arange(minVal, maxVal+gridspacing, gridspacing)
        p = partial(np.digitize, bins=r)
        self._df[['bin_x', 'bin_y', 'bin_z']] = self._df.loc[self._df.surfaceAtom==True, ['x', 'y', 'z']].apply(p)
        self._df['bin'] = self._df[self._df.surfaceAtom==True].apply(lambda r: '%i_%i_%i' % (r['bin_x'], r['bin_y'], r['bin_z']), axis=1)
        # unify bins
        #tmpdf = self._df[self._df.surfaceAtom==True].sort('probPPI', ascending=False).groupby(['bin', 'chain', 'resi']).first().reset_index()
        melted = melt(self._df.loc[self._df.surfaceAtom==True, ['chain', 'resi', 'resn', 'probPPI', 'bin']], id_vars=['chain', 'resi', 'bin', 'resn'], value_name='probPPI')
        tmpdf = melted.groupby(['chain', 'resi', 'bin']).first().reset_index()

        # try to increase bin size
        for bin_x in xrange(1, len(r)):
            for bin_y in xrange(1, len(r)):
                for bin_z in xrange(1, len(r)):
                    meid = '%i_%i_%i' % (bin_x, bin_y, bin_z)
                    me = tmpdf[tmpdf.bin == meid]            
            
                    mesum = me['probPPI'].sum()
                    memean = me['probPPI'].mean()
                    if len(me) > 0 and memean > 0.0:
                        prevBins = [(bin_x -  1, bin_y, bin_z),
                                    (bin_x, bin_y -  1, bin_z),
                                    (bin_x, bin_y, bin_z -  1)]
                        for prevBin in prevBins:
                            binid = '%i_%i_%i' % prevBin
                            b = tmpdf[tmpdf.bin == binid]
                            bsum = b['probPPI'].sum()
                            if len(b) > 0:
                                combinedMean = (memean + b['probPPI'].mean()) / (len(me) + len(b))                    
                                if combinedMean >= memean:
                                    tmpdf.loc[(tmpdf.bin == binid), 'bin'] = meid
            
        bin_means = tmpdf.groupby(['chain', 'resi']).reset_index().groupby('bin').mean()['probPPI'].copy()
        bin_means.sort(ascending=False)
        return bin_means