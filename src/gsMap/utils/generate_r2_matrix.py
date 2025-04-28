import bitarray as ba
import numpy as np
import pandas as pd
from tqdm import tqdm


def ID_List_Factory(colnames, keepcol, fname_end, header=None, usecols=None):
    """
    Factory function to create an IDContainer class for reading and storing data.
    """

    class IDContainer:
        """
        A class to read data from a file and store it as a DataFrame.
        """

        def __init__(self, fname):
            """
            Initialize the IDContainer with the given filename and reading options.
            """
            self.usecols = usecols
            self.colnames = colnames
            self.keepcol = keepcol
            self.fname_end = fname_end
            self.header = header
            self.read(fname)
            self.n = len(self.df)

        def read(self, fname):
            """
            Read data from the given file and store it as a DataFrame.
            """
            end = self.fname_end
            if end and not fname.endswith(end):
                raise ValueError(f"{end} filename must end in {end}")
            self.df = pd.read_csv(
                fname,
                header=self.header,
                usecols=self.usecols,
                sep=r"\s+",
            )
            if self.colnames:
                self.df.columns = self.colnames
            if self.keepcol is not None:
                self.IDList = self.df.iloc[:, [self.keepcol]].astype("object")

    return IDContainer


def getBlockLefts(coords, max_dist):
    """
    Converts coordinates + max block length to a list of coordinates of the leftmost
    SNPs to be included in blocks.
    """
    M = len(coords)
    j = 0
    block_left = np.zeros(M)
    for i in range(M):
        while j < M and abs(coords[j] - coords[i]) > max_dist:
            j += 1

        block_left[i] = j
    return block_left


def block_left_to_right(block_left):
    """
    Converts block lefts to block rights.
    """
    M = len(block_left)
    j = 0
    block_right = np.zeros(M)
    for i in range(M):
        while j < M and block_left[j] <= i:
            j += 1
        block_right[i] = j

    return block_right


class PlinkBEDFile:
    """
    Interface for Plink .bed format for reading and processing genotype data.
    """

    def __init__(self, fname, n, snp_list):
        """
        Initialize the PlinkBEDFile with the given parameters, pre-calculating MAF for all SNPs.

        Parameters
        ----------
        fname : str
            Path to the .bed file.
        n : int
            Number of individuals.
        snp_list : object
            Object containing SNP information.
        """
        # Initialize bitarray for bed code mapping
        self._bedcode = {
            2: ba.bitarray("11"),
            9: ba.bitarray("10"),
            1: ba.bitarray("01"),
            0: ba.bitarray("00"),
        }

        # Store original data
        self.m_original = len(snp_list.IDList)
        self.n_original = n
        self.snp_list = snp_list
        self.df_original = np.array(snp_list.df[["CHR", "SNP", "BP", "CM"]])
        self.colnames = ["CHR", "SNP", "BP", "CM"]

        # Read the bed file
        (self.nru_original, self.geno_original) = self._read(fname, self.m_original, n)

        # Initialize current state variables
        self._currentSNP = 0
        self.m = self.m_original
        self.n = self.n_original
        self.nru = self.nru_original
        self.geno = self.geno_original
        self.df = self.df_original

        # Pre-calculate MAF for all SNPs
        self.all_snp_info = self._calculate_all_snp_info()

        # Initialize as unfiltered
        self.kept_snps = np.arange(self.m_original)
        self.freq = self.all_snp_info['freq']
        self.maf = np.minimum(self.freq, 1 - self.freq)
        self.sqrtpq = np.sqrt(self.freq * (1 - self.freq))

        # Update dataframe with MAF
        self.df = np.c_[self.df, self.maf]
        self.colnames = ["CHR", "SNP", "BP", "CM", "MAF"]

    def _read(self, fname, m, n):
        """
        Read the bed file and return the genotype data.
        """
        if not fname.endswith(".bed"):
            raise ValueError(".bed filename must end in .bed")

        fh = open(fname, "rb")
        magicNumber = ba.bitarray(endian="little")
        magicNumber.fromfile(fh, 2)
        bedMode = ba.bitarray(endian="little")
        bedMode.fromfile(fh, 1)
        e = (4 - n % 4) if n % 4 != 0 else 0
        nru = n + e

        # Check magic number
        if magicNumber != ba.bitarray("0011011011011000"):
            raise OSError("Magic number from Plink .bed file not recognized")

        if bedMode != ba.bitarray("10000000"):
            raise OSError("Plink .bed file must be in default SNP-major mode")

        # Check file length
        geno = ba.bitarray(endian="little")
        geno.fromfile(fh)
        self._test_length(geno, m, nru)
        return (nru, geno)

    def _test_length(self, geno, m, nru):
        """
        Test if the genotype data has the expected length.
        """
        exp_len = 2 * m * nru
        real_len = len(geno)
        if real_len != exp_len:
            s = "Plink .bed file has {n1} bits, expected {n2}"
            raise OSError(s.format(n1=real_len, n2=exp_len))

    def _calculate_all_snp_info(self):
        """
        Pre-calculate MAF and other information for all SNPs.

        Returns
        -------
        dict
            Dictionary containing information for all SNPs
        """
        nru = self.nru_original
        n = self.n_original
        m = self.m_original
        geno = self.geno_original

        snp_info = {
            'freq': np.zeros(m),  # Allele frequencies
            'het_miss_count': np.zeros(m),  # Count of het or missing genotypes
            'valid_snp': np.zeros(m, dtype=bool)  # Whether SNP passes basic criteria
        }

        # For each SNP, calculate statistics
        for j in range(m):
            z = geno[2 * nru * j: 2 * nru * (j + 1)]
            A = z[0::2]
            a = A.count()
            B = z[1::2]
            b = B.count()
            c = (A & B).count()
            major_ct = b + c  # number of copies of the major allele
            n_nomiss = n - a + c  # number of individuals with nonmissing genotypes
            f = major_ct / (2 * n_nomiss) if n_nomiss > 0 else 0
            het_miss_ct = a + b - 2 * c  # count of SNPs that are het or missing

            snp_info['freq'][j] = f
            snp_info['het_miss_count'][j] = het_miss_ct
            snp_info['valid_snp'][j] = (het_miss_ct < n)  # Basic validity check

        return snp_info

    def apply_filters(self, keep_snps=None, keep_indivs=None, mafMin=None):
        """
        Apply filters to the genotype data without reloading the bed file.

        Parameters
        ----------
        keep_snps : array-like, optional
            Indices of SNPs to keep.
        keep_indivs : array-like, optional
            Indices of individuals to keep.
        mafMin : float, optional
            Minimum minor allele frequency.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        # Reset to original state first
        self.geno = self.geno_original.copy()
        self.m = self.m_original
        self.n = self.n_original
        self.nru = self.nru_original
        self.df = self.df_original.copy()
        self._currentSNP = 0

        # Initialize with all SNPs
        kept_snps = np.arange(self.m_original)

        # Apply MAF filter using pre-calculated values
        if mafMin is not None and mafMin > 0:
            maf_values = np.minimum(self.all_snp_info['freq'], 1 - self.all_snp_info['freq'])
            maf_mask = (maf_values > mafMin) & self.all_snp_info['valid_snp']
            kept_snps = kept_snps[maf_mask]
            print(f"After MAF filtering (>{mafMin}), {len(kept_snps)} SNPs remain")

        # Apply SNP filter if specified
        if keep_snps is not None:
            keep_snps = np.array(keep_snps, dtype="int")
            if np.any(keep_snps > self.m_original):
                raise ValueError("keep_snps indices out of bounds")

            # Intersect with current kept_snps
            kept_snps = np.intersect1d(kept_snps, keep_snps)
            print(f"After keep_snps filtering, {len(kept_snps)} SNPs remain")

        # Filter SNPs in the genotype data
        if len(kept_snps) < self.m_original:
            # Create new genotype data with only the kept SNPs
            new_geno = ba.bitarray()
            for j in kept_snps:
                new_geno += self.geno_original[2 * self.nru * j: 2 * self.nru * (j + 1)]
            self.geno = new_geno
            self.m = len(kept_snps)

        # Filter individuals if specified
        if keep_indivs is not None:
            keep_indivs = np.array(keep_indivs, dtype="int")
            if np.any(keep_indivs > self.n):
                raise ValueError("keep_indivs indices out of bounds")

            (self.geno, self.m, self.n) = self._filter_indivs(
                self.geno, keep_indivs, self.m, self.n
            )

            if self.n > 0:
                print(f"After filtering, {self.n} individuals remain")
            else:
                raise ValueError("After filtering, no individuals remain")

        # Update kept_snps and other attributes
        self.kept_snps = kept_snps
        self.freq = self.all_snp_info['freq'][kept_snps]
        self.maf = np.minimum(self.freq, 1 - self.freq)
        self.sqrtpq = np.sqrt(self.freq * (1 - self.freq))

        # Update dataframe with kept SNPs
        self.df = self.df_original[kept_snps]
        self.df = np.c_[self.df, self.maf]
        self.colnames = ["CHR", "SNP", "BP", "CM", "MAF"]

        return self

    def _filter_indivs(self, geno, keep_indivs, m, n):
        """
        Filter individuals based on the keep_indivs parameter.
        """
        n_new = len(keep_indivs)
        e = (4 - n_new % 4) if n_new % 4 != 0 else 0
        nru_new = n_new + e
        nru = self.nru
        z = ba.bitarray(m * 2 * nru_new, endian="little")
        z.setall(0)
        for e, i in enumerate(keep_indivs):
            z[2 * e:: 2 * nru_new] = geno[2 * i:: 2 * nru]
            z[2 * e + 1:: 2 * nru_new] = geno[2 * i + 1:: 2 * nru]
        self.nru = nru_new
        return (z, m, n_new)

    def get_snps_by_maf(self, mafMin):
        """
        Get the list of SNPs that pass the MAF threshold.

        Parameters
        ----------
        mafMin : float
            Minimum MAF threshold

        Returns
        -------
        list
            List of SNP IDs that pass the MAF threshold
        """
        # Use the pre-calculated MAF values
        maf_values = np.minimum(self.all_snp_info['freq'], 1 - self.all_snp_info['freq'])
        maf_mask = (maf_values > mafMin) & self.all_snp_info['valid_snp']

        # Get SNP names from the SNP list
        snp_pass_maf = self.snp_list.IDList.iloc[maf_mask]

        return snp_pass_maf.SNP.to_list()

    def restart(self):
        """
        Reset the current SNP index to 0.
        """
        self._currentSNP = 0

    def nextSNPs(self, b, minorRef=None):
        """
        Unpacks the binary array of genotypes and returns an n x b matrix of floats of
        normalized genotypes for the next b SNPs.
        """
        try:
            b = int(b)
            if b <= 0:
                raise ValueError("b must be > 0")
        except TypeError as e:
            raise TypeError("b must be an integer") from e

        if self._currentSNP + b > self.m:
            s = "{b} SNPs requested, {k} SNPs remain"
            raise ValueError(s.format(b=b, k=(self.m - self._currentSNP)))

        c = self._currentSNP
        n = self.n
        nru = self.nru
        slice = self.geno[2 * c * nru: 2 * (c + b) * nru]
        X = np.array(slice.decode(self._bedcode), dtype="float64").reshape((b, nru)).T
        X = X[0:n, :]
        Y = np.zeros(X.shape)

        # Normalize the SNPs and impute the missing ones with the mean
        for j in range(0, b):
            newsnp = X[:, j]
            ii = newsnp != 9
            avg = np.mean(newsnp[ii])
            newsnp[np.logical_not(ii)] = avg
            denom = np.std(newsnp)
            if denom == 0:
                denom = 1

            if minorRef is not None and self.freq[self._currentSNP + j] > 0.5:
                denom = denom * -1

            Y[:, j] = (newsnp - avg) / denom

        self._currentSNP += b
        return Y

    def _l2_unbiased(self, x, n):
        """
        Calculate the unbiased estimate of L2.
        """
        denom = n - 2 if n > 2 else n  # allow n<2 for testing purposes
        sq = np.square(x)
        return sq - (1 - sq) / denom

    def ldScoreVarBlocks(self, block_left, c, annot=None):
        """
        Computes an unbiased estimate of L2(j) for j=1,..,M.
        """

        def func(x):
            return self._l2_unbiased(x, self.n)

        snp_getter = self.nextSNPs
        return self._corSumVarBlocks(block_left, c, func, snp_getter, annot)

    def _corSumVarBlocks(self, block_left, c, func, snp_getter, annot=None):
        """
        Calculate the sum of correlation coefficients.
        """
        m, n = self.m, self.n
        block_sizes = np.array(np.arange(m) - block_left)
        block_sizes = np.ceil(block_sizes / c) * c
        if annot is None:
            annot = np.ones((m, 1))
        else:
            annot_m = annot.shape[0]
            if annot_m != self.m:
                raise ValueError("Incorrect number of SNPs in annot")

        n_a = annot.shape[1]  # number of annotations
        cor_sum = np.zeros((m, n_a))
        # b = index of first SNP for which SNP 0 is not included in LD Score
        b = np.nonzero(block_left > 0)
        if np.any(b):
            b = b[0][0]
        else:
            b = m
        b = int(np.ceil(b / c) * c)  # round up to a multiple of c
        if b > m:
            c = 1
            b = m

        l_A = 0  # l_A := index of leftmost SNP in matrix A
        A = snp_getter(b)
        rfuncAB = np.zeros((b, c))
        rfuncBB = np.zeros((c, c))
        # chunk inside of block
        for l_B in np.arange(0, b, c):  # l_B := index of leftmost SNP in matrix B
            B = A[:, l_B: l_B + c]
            # ld matrix
            np.dot(A.T, B / n, out=rfuncAB)
            # ld matrix square
            rfuncAB = func(rfuncAB)
            cor_sum[l_A: l_A + b, :] += np.dot(rfuncAB, annot[l_B: l_B + c, :])

        # chunk to right of block
        b0 = b
        md = int(c * np.floor(m / c))
        end = md + 1 if md != m else md
        for l_B in tqdm(np.arange(b0, end, c), desc="Compute SNP Gene Weight"):
            # check if the annot matrix is all zeros for this block + chunk
            # this happens w/ sparse categories (i.e., pathways)
            # update the block
            old_b = b
            b = int(block_sizes[l_B])
            if l_B > b0 and b > 0:
                # block_size can't increase more than c
                # block_size can't be less than c unless it is zero
                # both of these things make sense
                A = np.hstack((A[:, old_b - b + c: old_b], B))
                l_A += old_b - b + c
            elif l_B == b0 and b > 0:
                A = A[:, b0 - b: b0]
                l_A = b0 - b
            elif b == 0:  # no SNPs to left in window, e.g., after a sequence gap
                A = np.array(()).reshape((n, 0))
                l_A = l_B
            if l_B == md:
                c = m - md
                rfuncAB = np.zeros((b, c))
                rfuncBB = np.zeros((c, c))
            if b != old_b:
                rfuncAB = np.zeros((b, c))

            B = snp_getter(c)
            p1 = np.all(annot[l_A: l_A + b, :] == 0)
            p2 = np.all(annot[l_B: l_B + c, :] == 0)
            if p1 and p2:
                continue

            np.dot(A.T, B / n, out=rfuncAB)
            rfuncAB = func(rfuncAB)
            cor_sum[l_A: l_A + b, :] += np.dot(rfuncAB, annot[l_B: l_B + c, :])
            cor_sum[l_B: l_B + c, :] += np.dot(annot[l_A: l_A + b, :].T, rfuncAB).T
            np.dot(B.T, B / n, out=rfuncBB)
            rfuncBB = func(rfuncBB)
            cor_sum[l_B: l_B + c, :] += np.dot(rfuncBB, annot[l_B: l_B + c, :])

        return cor_sum


def initialize_bed_file(bfile_chr_prefix):
    """
    Initialize a PlinkBEDFile without applying any filters, but pre-calculating MAF info.

    Parameters
    ----------
    bfile_chr_prefix : str
        Prefix of the PLINK binary file.

    Returns
    -------
    geno_array : PlinkBEDFile
        Initialized PlinkBEDFile object
    array_snps : IDContainer
        Object containing SNP information
    array_indivs : IDContainer
        Object containing individual information
    """
    PlinkBIMFile = ID_List_Factory(
        ["CHR", "SNP", "CM", "BP", "A1", "A2"], 1, ".bim", usecols=[0, 1, 2, 3, 4, 5]
    )
    PlinkFAMFile = ID_List_Factory(["IID"], 0, ".fam", usecols=[1])

    # Load SNP info
    snp_file = bfile_chr_prefix + ".bim"
    array_snps = PlinkBIMFile(snp_file)

    # Load individual info
    ind_file = bfile_chr_prefix + ".fam"
    array_indivs = PlinkFAMFile(ind_file)
    n = len(array_indivs.IDList)

    # Load genotype data without applying filters
    array_file = bfile_chr_prefix + ".bed"
    geno_array = PlinkBEDFile(array_file, n, array_snps)

    return geno_array, array_snps, array_indivs


def get_filtered_bed_file(bed_file, keep_snps=None, keep_indivs=None, mafMin=None):
    """
    Get a filtered PlinkBEDFile object by applying filters to an existing object.

    Parameters
    ----------
    bed_file : PlinkBEDFile
        An initialized PlinkBEDFile object
    keep_snps : array-like, optional
        Indices of SNPs to keep.
    keep_indivs : array-like, optional
        Indices of individuals to keep.
    mafMin : float, optional
        Minimum minor allele frequency.

    Returns
    -------
    PlinkBEDFile
        Filtered PlinkBEDFile object
    """
    # Apply filters and return
    return bed_file.apply_filters(keep_snps=keep_snps, keep_indivs=keep_indivs, mafMin=mafMin)


def load_bfile(bfile_chr_prefix, keep_snps=None, keep_indivs=None, mafMin=None):
    """
    Load a binary PLINK file with optional filtering.

    This version maintains backward compatibility with the original function.

    Parameters
    ----------
    bfile_chr_prefix : str
        Prefix of the PLINK binary file.
    keep_snps : array-like, optional
        Indices of SNPs to keep.
    keep_indivs : array-like, optional
        Indices of individuals to keep.
    mafMin : float, optional
        Minimum minor allele frequency.

    Returns
    -------
    array_snps : IDContainer
        Object containing SNP information.
    array_indivs : IDContainer
        Object containing individual information.
    geno_array : PlinkBEDFile
        Object containing filtered genotype data.
    """
    # Initialize without filters
    geno_array, array_snps, array_indivs = initialize_bed_file(bfile_chr_prefix)

    # Apply filters if any are specified
    if keep_snps is not None or keep_indivs is not None or mafMin is not None:
        geno_array.apply_filters(keep_snps=keep_snps, keep_indivs=keep_indivs, mafMin=mafMin)

    return array_snps, array_indivs, geno_array